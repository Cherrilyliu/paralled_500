#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mineru PDF 多进程并行处理脚本 (Mineru PDF Processing Parallel Pipeline)
=====================================================================

这是一个用于大规模 PDF 处理的工程化脚本。它基于 Python 的多进程 (Multiprocessing) 模块构建，
能够自动将任务分发到多张 GPU 上并行运行。

主要功能：
1. 多显卡并行：自动在指定的 GPU 之间进行负载均衡。
2. 进程隔离：每个 Worker 进程独立运行，避免 CUDA 上下文冲突。
3. 断点续传：支持跳过输出目录中已经存在的文件。
4. 结构保持：输出文件会尝试保持输入文件夹的层级结构。

使用示例 (Usage Examples):
------------------------

1. 基本用法 (使用 0 号卡，启动 2 个进程):
   python process_pdf.py --input-dir ./my_pdfs --output-dir ./output --gpus 0 --num-workers 2

2. 多卡并行 (使用 0,1,2,3 四张卡，每张卡跑 2 个进程，共 8 个进程):
   python process_pdf.py --input-dir ./data --output-dir ./results --gpus 0,1,2,3 --num-workers 8

3. 指定后端与语言 (使用 VLM 后端处理中文):
   python process_pdf.py --input-dir ./pdfs --output-dir ./json --backend vlm-siliconcloud --lang zh

4. 跳过已处理文件 (用于中断后恢复):
   python process_pdf.py --input-dir ./data --output-dir ./results --skip-processed

依赖安装:
   pip install loguru mineru

Author: relic-yuexi
License: Apache-2.0
"""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import warnings
import queue
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Set

from loguru import logger

# --- 配置与常量 ---

# Loguru 日志格式配置
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[worker_id]}</cyan> | "
    "<level>{message}</level>"
)

# 抑制第三方库的冗余警告
warnings.filterwarnings("ignore", message=".*Cannot set gray non-stroke color.*")
warnings.filterwarnings("ignore", message=".*FontBBox.*")
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pypdfium2").setLevel(logging.ERROR)


def setup_worker_env(gpu_id: int, vram_size: int) -> None:
    """
    设置 Worker 进程的环境变量。
    
    关键说明:
    必须在导入任何 torch 或 mineru 相关库之前调用此函数。
    这通过设置 CUDA_VISIBLE_DEVICES 确保该进程只能看到被分配的那张显卡。
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['MINERU_MODEL_SOURCE'] = "modelscope"
    # 注意: 因为 CUDA_VISIBLE_DEVICES 已经屏蔽了其他卡，所以对进程来说永远是 device 0
    os.environ['MINERU_DEVICE_MODE'] = "cuda:0"  
    os.environ['MINERU_VIRTUAL_VRAM_SIZE'] = str(vram_size)


def process_single_pdf(
    pdf_path: str,
    output_base_dir: str,
    lang: str,
    backend: str,
    gpu_id: int
) -> Tuple[str, bool, Optional[str]]:
    """
    处理单个 PDF 的核心逻辑。
    
    Args:
        pdf_path: PDF 文件绝对路径
        output_base_dir: 输出根目录
        lang: 语言代码 (如 'en', 'zh')
        backend: 解析后端 ('pipeline', 'vlm-xxx')
        gpu_id: 分配的物理 GPU ID (仅用于日志记录)
        
    Returns:
        (file_path, is_success, error_message)
    """
    try:
        # --- 关键：延迟导入 (Lazy Import) ---
        # 必须在子进程内部导入这些库，防止在主进程中过早初始化 CUDA 上下文导致冲突。
        from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
        from mineru.data.data_reader_writer import FileBasedDataWriter
        from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
        from mineru.utils.enum_class import MakeMode
        from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
        from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
        from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
        from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
        from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
        import copy

        pdf_path_obj = Path(pdf_path)
        
        # --- 计算输出路径 ---
        # 尝试保持相对目录结构。如果没有明确的层级，则保存到输出根目录。
        try:
            # 这里的逻辑是：如果 PDF 在 input_dir/A/B.pdf，我们希望输出到 output_dir/A/
            # 简单起见，这里取父文件夹名作为子目录
            rel_path = pdf_path_obj.parent.name 
        except Exception:
            rel_path = "."

        target_output_dir = Path(output_base_dir) / rel_path
        target_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = pdf_path_obj.stem
        
        # --- Mineru 处理逻辑 ---
        pdf_bytes = read_fn(str(pdf_path_obj))
        
        # 设置解析参数
        parse_method = "auto" if backend == "pipeline" else "vlm"
        formula_enable = True
        table_enable = True
        
        # 准备环境 (创建 output/images 和 output/md 目录)
        local_image_dir, local_md_dir = prepare_env(str(target_output_dir), file_name, parse_method)
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)
        
        # 根据后端选择处理流程
        if backend == "pipeline":
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                [pdf_bytes], [lang], 
                parse_method="auto", 
                formula_enable=formula_enable,
                table_enable=table_enable
            )
            
            # 提取 Pipeline 结果
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, lang_list[0], ocr_enabled_list[0], formula_enable
            )
            make_func = pipeline_union_make
            # model_output = copy.deepcopy(model_list) # 如果需要保存原始模型输出可解开注释
            
        else: # VLM 后端逻辑
            vlm_backend_name = backend.replace("vlm-", "") if backend.startswith("vlm-") else backend
            middle_json, model_output = vlm_doc_analyze(
                pdf_bytes, image_writer=image_writer, backend=vlm_backend_name, server_url=None
            )
            make_func = vlm_union_make

        # --- 生成最终结果文件 ---
        pdf_info = middle_json["pdf_info"]
        image_dir_name = os.path.basename(local_image_dir)
        
        # 1. 写入 Markdown
        md_content = make_func(pdf_info, MakeMode.MM_MD, image_dir_name)
        md_writer.write_string(f"{file_name}.md", md_content)
        
        # 2. 写入 Content List (JSON)
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir_name)
        md_writer.write_string(f"{file_name}_content_list.json", json.dumps(content_list, ensure_ascii=False, indent=4))
        
        # 3. 写入 Middle JSON (包含详细结构信息，方便调试)
        md_writer.write_string(f"{file_name}_middle.json", json.dumps(middle_json, ensure_ascii=False, indent=4))

        logger.info(f"Success: {local_md_dir}")
        return str(pdf_path), True, None

    except Exception:
        # 捕获所有异常，确保进程不崩溃
        err_msg = traceback.format_exc()
        # 只记录最后一行错误信息以保持日志整洁，详细堆栈已在 err_msg 中
        logger.error(f"Error processing {pdf_path}: {err_msg.splitlines()[-1]}")
        return str(pdf_path), False, err_msg


def worker_loop(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    gpu_id: int,
    worker_idx: int,
    vram_size: int
):
    """
    Worker 进程的主循环。
    不断从 task_queue 取任务，处理完后将结果放入 result_queue。
    """
    # 1. 初始化环境变量 (最重要的一步)
    setup_worker_env(gpu_id, vram_size)
    
    # 2. 配置当前进程的 Logger (带上 Worker ID 和 GPU ID)
    logger.configure(extra={"worker_id": f"W-{worker_idx}|GPU-{gpu_id}"})
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level="INFO")
    
    logger.info("Worker started.")
    
    while True:
        try:
            # 获取任务，设置超时防止死锁
            task = task_queue.get(timeout=3.0)
            
            # 如果收到 None，说明是终止信号
            if task is None:
                break 
                
            # 解包任务参数
            pdf_path, output_dir, lang, backend = task
            
            # 执行处理
            result = process_single_pdf(pdf_path, output_dir, lang, backend, gpu_id)
            
            # 返回结果
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            logger.critical(f"Worker crashed unexpectedly: {e}")
            break
            
    logger.info("Worker shutdown.")


def get_processed_files(output_dir: Path) -> Set[str]:
    """扫描输出目录，获取已经处理完成的 PDF 文件名集合 (基于 .md 文件存在与否)"""
    processed = set()
    if output_dir.exists():
        for f in output_dir.rglob("*.md"):
            processed.add(f.stem)
    return processed


def main():
    # --- 命令行参数定义 ---
    parser = argparse.ArgumentParser(description="Mineru 多进程 PDF 并行解析工具")
    parser.add_argument("--input-dir", type=str, required=True, help="输入 PDF 文件夹路径")
    parser.add_argument("--output-dir", type=str, required=True, help="结果输出文件夹路径")
    parser.add_argument("--num-workers", type=int, default=4, help="并行的 Worker 进程总数 (建议每张卡 2-4 个)")
    parser.add_argument("--gpus", type=str, default="0", help="使用的 GPU ID 列表，用逗号分隔 (例如: '0,1,2')")
    parser.add_argument("--backend", type=str, default="pipeline", choices=["pipeline", "vlm-siliconcloud", "vlm-openai"], help="解析后端选择")
    parser.add_argument("--lang", type=str, default="en", help="文档主要语言 (en 或 zh)")
    parser.add_argument("--skip-processed", action="store_true", help="是否跳过输出目录中已存在的文件")
    parser.add_argument("--vram", type=int, default=8, help="每个进程预设的虚拟显存大小 (GB)，仅用于 MinerU 内部配置")
    
    args = parser.parse_args()
    
    # --- 初始化检查 ---
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    gpu_list = [int(x.strip()) for x in args.gpus.split(",")]
    
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        sys.exit(1)

    # --- 文件扫描 ---
    logger.info(f"正在扫描目录: {input_path} ...")
    all_pdfs = list(input_path.rglob("*.pdf"))
    
    # 过滤已处理文件
    if args.skip_processed:
        processed_files = get_processed_files(output_path)
        pending_pdfs = [p for p in all_pdfs if p.stem not in processed_files]
        logger.info(f"总文件数: {len(all_pdfs)} | 跳过已处理: {len(processed_files)} | 待处理: {len(pending_pdfs)}")
    else:
        pending_pdfs = all_pdfs
        logger.info(f"待处理文件总数: {len(all_pdfs)}")

    if not pending_pdfs:
        logger.info("没有需要处理的文件。退出。")
        return

    # --- 多进程启动准备 ---
    # 必须使用 'spawn' 模式，否则 CUDA 上下文会在 fork 时被错误复制
    ctx = mp.get_context('spawn') 
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    
    # 1. 填充任务队列
    for pdf in pending_pdfs:
        task_queue.put((str(pdf), str(output_path), args.lang, args.backend))
    
    # 2. 填充终止信号 (每个 Worker 需要一个 None)
    for _ in range(args.num_workers):
        task_queue.put(None)
        
    # 3. 启动 Worker 进程
    processes = []
    logger.info(f"启动 {args.num_workers} 个 Worker 进程，分布在 GPU: {gpu_list}")
    
    for i in range(args.num_workers):
        # --- 核心数学逻辑：轮询分配 (Round-Robin) ---
        # 公式: assigned_gpu = gpu_list[i % len(gpu_list)]
        # 作用: 无论有多少个进程，都会均匀地依次分配给列表中的 GPU。
        assigned_gpu = gpu_list[i % len(gpu_list)]
        
        p = ctx.Process(
            target=worker_loop,
            args=(task_queue, result_queue, assigned_gpu, i, args.vram)
        )
        p.start()
        processes.append(p)

    # --- 结果收集与监控 ---
    success_count = 0
    fail_count = 0
    failed_files = []
    
    start_time = time.time()
    total_tasks = len(pending_pdfs)
    
    try:
        while success_count + fail_count < total_tasks:
            # 检查是否有僵尸进程 (所有 Worker 都挂了但任务没完)
            if not any(p.is_alive() for p in processes) and result_queue.empty():
                logger.error("所有 Worker 进程均已退出，但任务未完成。强制停止。")
                break
                
            try:
                # 获取结果，超时时间 5 秒
                path, success, msg = result_queue.get(timeout=5)
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    failed_files.append((path, msg))
                
                # 每处理 10 个文件或遇到错误时打印进度
                if (success_count + fail_count) % 10 == 0 or not success:
                    logger.info(f"进度: {success_count + fail_count}/{total_tasks} | 成功: {success_count} | 失败: {fail_count}")
                    
            except queue.Empty:
                continue
                
    except KeyboardInterrupt:
        logger.warning("用户中断 (Ctrl+C)。正在终止所有进程...")
        for p in processes:
            p.terminate()
            
    finally:
        # 清理与收尾
        logger.info("等待所有子进程退出...")
        for p in processes:
            p.join()
            
        duration = time.time() - start_time
        logger.info(f"任务结束。耗时: {duration:.2f}秒。成功: {success_count}, 失败: {fail_count}")
        
        # 保存失败列表
        if failed_files:
            log_file = output_path / f"failed_files_{int(time.time())}.txt"
            with open(log_file, "w", encoding="utf-8") as f:
                for path, msg in failed_files:
                    f.write(f"{path}\t{msg}\n")
            logger.warning(f"失败文件列表已保存至: {log_file}")

if __name__ == "__main__":
    # 强制设置全局启动方法为 spawn，确保兼容性
    mp.set_start_method('spawn', force=True)
    
    # 配置主进程日志
    logger.configure(extra={"worker_id": "MAIN"})
    logger.remove()
    logger.add(sys.stderr, format=LOG_FORMAT, level="INFO")
    
    main()
