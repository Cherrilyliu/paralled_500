#!/usr/bin/env python3
"""
自动分批导入脚本：每批解析指定数量的 PDF，从 SQLite 读元数据，写入7个新索引。
文件名即 lngid（如 1000004517957.pdf），用 lngid 去 db3 查元数据。
7个索引里用 file_id 字段（值 = lngid = 文件名）。
支持多个 MinerU 实例轮询负载均衡、单篇超时和断点续传。
每批完成后自动重启 MinerU 容器，服务恢复后继续下一批，直到处理完全部 PDF。
"""

import os
import sys
import json
import asyncio
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# 路径配置（部署环境）
PROJECT_ROOT = Path("/home/vscodeuser/yanjiushequ/Encyclopedia_project")
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "algorithm"))

import httpx
from algorithm.knowledge_base.file_parser.service import MinerUParserWithStructure
from RAG.datamerge import DataMergePipeline
from RAG.datamerge.data_types import Chunk, ChunkStrategy
from values.global_config_variable import (
    ES_URL,
    ES_TITLE_INDEX, ES_ABSTRACT_INDEX, ES_KEYWORD_INDEX, ES_FULLTEXT_INDEX,
    ES_PARAGRAPH_INDEX, ES_SENTENCE_INDEX, ES_REFERENCE_INDEX,
)
from algorithm.ai_tools.metadata_loader import load_metadata_by_lngid, extract_lngid_from_filename
from algorithm.ai_tools.sentence_chunker import paragraph_to_sentence_chunks


# =========================================================
# 配置
# =========================================================
PDF_DIR = PROJECT_ROOT / "data" / "500pdf"
DB3_PATH = r"/home/vscodeuser/yanjiushequ/Encyclopedia_project/zkyrjyjs_50w_20260707.db3"
STATS_FILE = PROJECT_ROOT / "data" / "500pdf_vec_result" / "batch_import_stats.json"
# 断点续传状态文件（放在用户有写权限的目录）
PROCESSED_FILE = Path("/tmp/500pdf_processed_ids.txt")  # 或 Path.home() / "500pdf_processed_ids.txt"
STATS_FILE.parent.mkdir(parents=True, exist_ok=True)    # 若权限不足，会抛出异常，可忽略

MAX_FULL_TOKENS = 8000
MINERU_PORTS = [8060]
MINERU_TASK_TIMEOUT = 600
CONTAINER_READY_TIMEOUT = 180
CONTAINER_POLL_INTERVAL = 3
BATCH_SIZE = 2000
DOCKER_COMMAND = "docker"
_port_index = 0
# 全局锁，用于保护状态文件并发写入
processed_lock = asyncio.Lock()


def get_next_mineru_port() -> int:
    global _port_index
    port = MINERU_PORTS[_port_index % len(MINERU_PORTS)]
    _port_index += 1
    return port


# =========================================================
# 向量化工具（复用 DataMerge Embedder）
# =========================================================
_pipeline = None

def get_pipeline() -> DataMergePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DataMergePipeline(chunk_strategy="structure")
    return _pipeline


def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_pipeline().embedder.embed_texts(texts)


def embed_one(text: str) -> List[float]:
    return embed_texts([text])[0] if text else []


# =========================================================
# ES 写入助手（file_id 字段 = lngid = 文件名）
# =========================================================
async def es_index(es_client: httpx.AsyncClient, index_name: str, doc_id: str, body: Dict[str, Any]):
    url = f"{ES_URL.rstrip('/')}/{index_name}/_doc/{doc_id}"
    await es_client.put(url, json=body, headers={"Content-Type": "application/json"})


async def import_one_pdf_to_es(es_client: httpx.AsyncClient, lngid: str, raw_chunks: List[Chunk], meta: Dict[str, Any]) -> Dict[str, int]:
    """把一篇PDF的解析结果写入7个索引。返回各库写入数量。"""
    counts = {"title": 0, "abstract": 0, "keyword": 0, "fulltext": 0, "paragraph": 0, "sentence": 0, "reference": 0}

    title_c = meta.get("title_c", "")
    title_e = meta.get("title_e", "")
    keyword_c = meta.get("keyword_c", "")
    keyword_e = meta.get("keyword_e", "")
    remark_c = meta.get("remark_c", "")
    remark_e = meta.get("remark_e", "")
    firstwriter = meta.get("firstwriter", "")
    showwriter = meta.get("showwriter", "")
    doi = meta.get("doi", "")

    common = {
        "file_id": lngid,
        "doi": doi,
        "firstwriter": firstwriter,
        "showwriter": showwriter,
        "delete_flag": 0,
    }

    # 1. 标题库
    try:
        title_text = (title_c + " " + title_e).strip()
        if title_text:
            await es_index(es_client, ES_TITLE_INDEX, f"{lngid}_title", {
                **common, "title_c": title_c, "title_e": title_e,
                "title_text": title_text, "title_embedding": embed_one(title_text),
            })
            counts["title"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 标题库失败: {e}")

    # 2. 摘要库
    try:
        abstract_text = (remark_c + " " + remark_e).strip()
        if abstract_text:
            await es_index(es_client, ES_ABSTRACT_INDEX, f"{lngid}_abstract", {
                **common, "remark_c": remark_c, "remark_e": remark_e,
                "abstract_embedding": embed_one(abstract_text),
            })
            counts["abstract"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 摘要库失败: {e}")

    # 3. 关键词库
    try:
        keyword_text = (keyword_c + " " + keyword_e).strip()
        if keyword_text:
            await es_index(es_client, ES_KEYWORD_INDEX, f"{lngid}_keyword", {
                **common, "keyword_c": keyword_c, "keyword_e": keyword_e,
                "keyword_text": keyword_text, "keyword_embedding": embed_one(keyword_text),
            })
            counts["keyword"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 关键词库失败: {e}")

    # 4. 段落/句子/全文/参考文献（需要 MinerU 解析结果）
    paragraph_chunks: List[Chunk] = []
    reference_chunks: List[Chunk] = []
    for c in raw_chunks:
        etype = getattr(c, "element_type", "")
        if etype in ("参考文献", "参考文献条目") or "参考文献" in getattr(c, "section_path", ""):
            reference_chunks.append(c)
        else:
            paragraph_chunks.append(c)

    # 4a. 全文库
    try:
        full_text = "\n".join([getattr(c, "text", "") for c in raw_chunks if getattr(c, "text", "")])
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        estimated_tokens = len(full_text) // 2
        if estimated_tokens <= MAX_FULL_TOKENS and full_text:
            await es_index(es_client, ES_FULLTEXT_INDEX, f"{lngid}_fulltext", {
                **common, "full_text": full_text,
                "full_embedding": embed_one(full_text), "token_count": estimated_tokens,
            })
            counts["fulltext"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 全文库失败: {e}")

    # 4b. 段落库
    try:
        para_texts = [getattr(c, "text", "") for c in paragraph_chunks if getattr(c, "text", "")]
        if para_texts:
            para_vecs = embed_texts(para_texts)
            for idx, (c, vec) in enumerate(zip(paragraph_chunks, para_vecs)):
                p_text = getattr(c, "text", "")
                if not p_text:
                    continue
                await es_index(es_client, ES_PARAGRAPH_INDEX, f"{lngid}_p{idx}", {
                    **common, "para_index": idx, "para_text": p_text, "para_embedding": vec,
                    "section_path": getattr(c, "section_path", ""),
                    "element_type": getattr(c, "element_type", ""),
                    "page_num": getattr(c, "page_num", 1),
                })
                counts["paragraph"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 段落库失败: {e}")

    # 4c. 句子库（2逗号规则）
    try:
        sent_docs = []
        for p_idx, c in enumerate(paragraph_chunks):
            p_text = getattr(c, "text", "")
            if not p_text:
                continue
            s_chunks = paragraph_to_sentence_chunks(p_text)
            for s_seq, s_text in enumerate(s_chunks):
                if s_text and len(s_text.strip()) >= 2:
                    sent_docs.append((p_idx, s_seq, s_text, getattr(c, "section_path", ""),
                                      getattr(c, "page_num", 1), getattr(c, "element_type", "")))
        if sent_docs:
            sent_texts = [d[2] for d in sent_docs]
            sent_vecs = embed_texts(sent_texts)
            for (p_idx, s_seq, s_text, sec, page, etype), vec in zip(sent_docs, sent_vecs):
                await es_index(es_client, ES_SENTENCE_INDEX, f"{lngid}_p{p_idx}s{s_seq}", {
                    **common, "para_index": p_idx, "sent_seq": s_seq,
                    "sent_text": s_text, "sent_embedding": vec,
                    "section_path": sec, "element_type": etype, "page_num": page,
                })
                counts["sentence"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 句子库失败: {e}")

    # 4d. 参考文献库
    try:
        ref_texts = [getattr(c, "text", "") for c in reference_chunks if getattr(c, "text", "")]
        if ref_texts:
            ref_vecs = embed_texts(ref_texts)
            for idx, (c, vec) in enumerate(zip(reference_chunks, ref_vecs)):
                r_text = getattr(c, "text", "")
                if not r_text:
                    continue
                await es_index(es_client, ES_REFERENCE_INDEX, f"{lngid}_r{idx}", {
                    **common, "ref_seq": idx, "ref_text": r_text, "ref_embedding": vec,
                })
                counts["reference"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 参考文献库失败: {e}")

    return counts


# =========================================================
# 单篇处理
# =========================================================
async def process_single_pdf(
    semaphore,
    pdf_path: Path,
    idx: int,
    attempted_ids: Optional[set] = None,
) -> Dict[str, Any]:
    async with semaphore:
        lngid = extract_lngid_from_filename(pdf_path.name)
        if attempted_ids is not None:
            attempted_ids.add(lngid)
        result = {
            "index": idx, "filename": pdf_path.name, "lngid": lngid,
            "success": False, "timeout": False, "error": None,
            "counts": {}, "time": 0,
        }
        start = time.time()
        try:
            port = get_next_mineru_port()
            parser = MinerUParserWithStructure(mineru_api_base=f"http://127.0.0.1:{port}")
            raw_chunks, _ = await asyncio.wait_for(
                parser.parse_pdf_with_structure(str(pdf_path), pdf_path.name),
                timeout=MINERU_TASK_TIMEOUT,
            )

            meta = load_metadata_by_lngid(lngid, db3_path=DB3_PATH)

            async with httpx.AsyncClient(trust_env=False, timeout=120.0) as es_client:
                counts = await import_one_pdf_to_es(es_client, lngid, raw_chunks, meta)

            result["counts"] = counts
            # 只有 checkpoint 成功落盘后才把任务标记为成功。
            async with processed_lock:
                with open(PROCESSED_FILE, "a") as f:
                    f.write(f"{lngid}\n")
                    f.flush()   # 立即落盘，防止中断丢失
            result["success"] = True
            total = sum(counts.values())
            print(f"  [OK] [{idx:04d}] {pdf_path.name}: {total} 条 -> {counts}")
        except asyncio.TimeoutError:
            result["timeout"] = True
            result["error"] = f"MinerU 解析超过 {MINERU_TASK_TIMEOUT} 秒"
            print(f"  [TIMEOUT] [{idx:04d}] {pdf_path.name}: {result['error']}")
        except Exception as e:
            result["error"] = str(e)
            print(f"  [FAIL] [{idx:04d}] {pdf_path.name}: {e}")
        result["time"] = round(time.time() - start, 2)
        return result


async def _run_docker(*args: str) -> Tuple[int, str, str]:
    """执行 Docker 命令，避免阻塞事件循环。"""
    try:
        process = await asyncio.create_subprocess_exec(
            DOCKER_COMMAND, *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return 127, "", f"无法执行 {DOCKER_COMMAND}: {exc}"
    stdout, stderr = await process.communicate()
    return process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _find_container_for_port(port: int) -> Optional[str]:
    """根据宿主机端口查找映射该端口的唯一容器。"""
    code, stdout, stderr = await _run_docker(
        "ps", "-q", "--filter", f"publish={port}"
    )
    if code != 0:
        print(f"[WARN] 查找端口 {port} 的容器失败: {stderr.strip()}")
        return None
    container_ids = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(container_ids) != 1:
        print(f"[WARN] 端口 {port} 对应容器数量为 {len(container_ids)}，期望 1")
        return None
    return container_ids[0]


async def _wait_mineru_ready(port: int) -> bool:
    """等待 MinerU API 端口恢复；服务返回任意 HTTP 响应即视为已监听。"""
    deadline = time.monotonic() + CONTAINER_READY_TIMEOUT
    url = f"http://127.0.0.1:{port}/tasks/health"
    async with httpx.AsyncClient(trust_env=False, timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(CONTAINER_POLL_INTERVAL)
    return False


async def restart_mineru_containers() -> bool:
    """逐个重启端口对应的 MinerU 容器并等待端口恢复。"""
    for port in MINERU_PORTS:
        container_id = await _find_container_for_port(port)
        if not container_id:
            return False
        print(f"[BATCH] 重启 MinerU 容器 {container_id}（端口 {port}）...")
        code, _, stderr = await _run_docker("restart", container_id)
        if code != 0:
            print(f"[ERROR] 容器 {container_id} 重启失败: {stderr.strip()}")
            return False
        if not await _wait_mineru_ready(port):
            print(f"[ERROR] MinerU 端口 {port} 在 {CONTAINER_READY_TIMEOUT}s 内未恢复")
            return False
        print(f"[BATCH] MinerU 端口 {port} 已恢复")
    return True


# =========================================================
# 批量导入主函数
# =========================================================
async def run_batch_import(concurrent: int = 10, limit: int = BATCH_SIZE):
    if concurrent < 1 or limit < 1:
        raise ValueError("并发数和每批数量都必须大于 0")

    print("=" * 70)
    print("[AUTO BATCH IMPORT] 自动分批导入到7个索引")
    print("=" * 70)
    print(f"PDF目录: {PDF_DIR}")
    print(f"DB3: {DB3_PATH}")
    print(f"并发数: {concurrent}")
    print(f"每批上限: {limit}")
    print(f"MinerU端口: {MINERU_PORTS}")
    print(f"单篇MinerU超时: {MINERU_TASK_TIMEOUT}s")
    print(f"状态文件: {PROCESSED_FILE}")
    print("=" * 70)

    processed_ids = set()
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE, "r") as f:
            processed_ids = {line.strip() for line in f if line.strip()}
        print(f"[INFO] 已加载 {len(processed_ids)} 个已处理的 lngid")

    all_pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    discovered_ids = {
        extract_lngid_from_filename(pdf_path.name) for pdf_path in all_pdf_files
    }
    processed_ids.intersection_update(discovered_ids)
    attempted_ids = set()
    overall_start = time.time()
    batch_number = 0
    total_success = 0
    total_fail = 0
    total_timeout = 0
    total_counts = {
        "title": 0, "abstract": 0, "keyword": 0, "fulltext": 0,
        "paragraph": 0, "sentence": 0, "reference": 0,
    }
    fail_details = []
    stopped_reason = None

    print(
        f"[INFO] 扫描到 {len(all_pdf_files)} 篇，"
        f"已完成 {len(processed_ids)} 篇，"
        f"尚未完成 {len(all_pdf_files) - len(processed_ids)} 篇"
    )

    while True:
        pending_pdfs = [
            pdf_path for pdf_path in all_pdf_files
            if extract_lngid_from_filename(pdf_path.name) not in processed_ids
            and extract_lngid_from_filename(pdf_path.name) not in attempted_ids
        ]
        if not pending_pdfs:
            remaining = len(all_pdf_files) - len(processed_ids)
            if remaining > 0:
                stopped_reason = "本次运行剩余任务均已失败或超时，留待下次启动重试"
                print(f"[STOP] {stopped_reason}，剩余 {remaining} 篇")
            else:
                print("[DONE] 所有 PDF 已处理完毕。")
            break

        batch_number += 1
        batch_pdfs = pending_pdfs[:limit]
        waiting_after_batch = len(pending_pdfs) - len(batch_pdfs)
        print("\n" + "=" * 70)
        print(
            f"[BATCH {batch_number}] 本批 {len(batch_pdfs)} 篇，"
            f"本批后尚有 {waiting_after_batch} 篇未尝试"
        )
        print("=" * 70)

        semaphore = asyncio.Semaphore(concurrent)
        batch_start = time.time()
        tasks = [
            process_single_pdf(semaphore, pdf_path, idx, attempted_ids)
            for idx, pdf_path in enumerate(batch_pdfs, 1)
        ]
        results = await asyncio.gather(*tasks)
        batch_time = time.time() - batch_start

        success = [result for result in results if result["success"]]
        fail = [result for result in results if not result["success"]]
        timeout = [result for result in fail if result.get("timeout")]
        total_success += len(success)
        total_fail += len(fail)
        total_timeout += len(timeout)

        for result in success:
            processed_ids.add(result["lngid"])
            for key, value in result.get("counts", {}).items():
                total_counts[key] = total_counts.get(key, 0) + value
        for result in fail:
            fail_details.append({
                "batch": batch_number,
                "filename": result["filename"],
                "lngid": result["lngid"],
                "timeout": result.get("timeout", False),
                "error": result["error"],
            })

        remaining = len(all_pdf_files) - len(processed_ids)
        batch_success_count = len(success)
        print(
            f"[BATCH {batch_number} 完成] 成功 {batch_success_count} / "
            f"失败 {len(fail)} / 超时 {len(timeout)} / "
            f"耗时 {batch_time:.1f}s / 剩余 {remaining}"
        )

        del tasks, results, success, fail, timeout, batch_pdfs

        if remaining == 0:
            print("[DONE] 所有 PDF 已处理完毕。")
            break
        if batch_success_count == 0 or not any(
            extract_lngid_from_filename(pdf_path.name) not in attempted_ids
            and extract_lngid_from_filename(pdf_path.name) not in processed_ids
            for pdf_path in all_pdf_files
        ):
            stopped_reason = "没有新的可尝试任务"
            print(f"[STOP] {stopped_reason}，失败任务留待下次启动重试")
            break

        print(f"[BATCH] 第 {batch_number} 批结束，准备重启 MinerU 容器...")
        if not await restart_mineru_containers():
            stopped_reason = "MinerU 容器重启或健康检查失败"
            print(f"[STOP] {stopped_reason}，停止后续批次")
            break

    total_time = time.time() - overall_start
    remaining = len(all_pdf_files) - len(processed_ids)
    print("\n" + "=" * 70)
    print("[SUMMARY] 自动分批汇总")
    print("=" * 70)
    print(f"批次数: {batch_number}")
    print(f"本次成功: {total_success} 篇 / 失败: {total_fail} 篇 / 超时: {total_timeout} 篇")
    print(f"累计已完成: {len(processed_ids)}/{len(all_pdf_files)} / 剩余: {remaining}")
    print(f"总耗时: {total_time:.1f}s ({total_time/3600:.2f}h)")
    if total_success:
        print(f"本次平均每篇成功任务: {total_time/total_success:.2f}s")
    print("\n各库写入量:")
    for key, value in total_counts.items():
        print(f"  {key:<12}: {value}")

    summary = {
        "discovered_files": len(all_pdf_files),
        "batch_size": limit,
        "batches": batch_number,
        "success": total_success,
        "fail": total_fail,
        "timeout": total_timeout,
        "processed_total": len(processed_ids),
        "remaining": remaining,
        "total_time": total_time,
        "stopped_reason": stopped_reason,
        "counts": total_counts,
        "fail_details": fail_details,
    }
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n统计已保存: {STATS_FILE}")
    except Exception as e:
        print(f"\n[WARN] 无法写入统计文件: {e}")
    return summary


async def main():
    concurrent = 10
    limit = BATCH_SIZE
    if len(sys.argv) > 1:
        concurrent = int(sys.argv[1])
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    await run_batch_import(concurrent=concurrent, limit=limit)


if __name__ == "__main__":
    asyncio.run(main())
