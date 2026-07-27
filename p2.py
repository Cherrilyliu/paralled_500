#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百万 PDF 两阶段并行导入脚本。

本文件保留 parallel_test_10000.py 的 Embedding、ES bulk 和索引构建逻辑，
并借鉴 parallel_test_100.py 的 spawn 进程、任务/结果队列、哨兵、超时轮询、
worker 存活检测和断点恢复机制。

流水线：
    PDF -> MinerU 解析进程（每端口可配置多个 worker）-> 磁盘临时文件
        -> 主进程异步 Embedding/ES worker -> checkpoint

解析结果不直接放入 multiprocessing.Queue，而是先写入 spool 目录，队列中只传
文件路径，避免大批 raw_chunks 堵塞进程管道或撑爆内存。

每个 MinerU 端口的 worker 数可独立配置。例如，一个实例使用 4 个 worker：
    python parallel_test_1000000.py \
      --pdf-dir /data/pdfs --db3-path /data/meta.db3 \
      --mineru-ports 8060 --workers-per-port 4 --index-workers 8

多个实例也可分别配置 worker 数，例如 8060 使用 2 个、8061 使用 3 个：
    --mineru-ports 8060,8061 --workers-per-port 2,3

默认每端口 1 个 worker，以便先建立稳定基线；应通过吞吐、显存和失败率压测逐步调高。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
# 支持从任意工作目录直接运行本文件；同时兼容旧脚本中的 RAG 顶层导入。
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "algorithm"):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)

DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "500pdf"
DEFAULT_DB3_PATH = Path(
    os.environ.get(
        "DB3_PATH",
        "/home/vscodeuser/yanjiushequ/Encyclopedia_project/zkyrjyjs_50w_20260707.db3",
    )
)
DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "million_pdf_vec_result"
COUNT_KEYS = (
    "title", "abstract", "keyword", "fulltext",
    "paragraph", "sentence", "reference",
)


def extract_lngid(filename: str) -> str:
    """在启动重依赖前提取 ID，规则与 metadata_loader 保持一致。"""
    import re

    base = os.path.basename(str(filename))
    base = re.sub(r"\.(pdf|docx?|txt|md)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^vec_[a-f0-9]+_", "", base)
    base = re.sub(r"^temp_[a-f0-9_-]+_", "", base)
    return base.strip()


def load_processed_ids(checkpoint_file: Path) -> Set[str]:
    if not checkpoint_file.exists():
        return set()
    with checkpoint_file.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def iter_pending_pdfs(
    pdf_dir: Path,
    processed_ids: Set[str],
    limit: int,
) -> Iterable[Tuple[int, Path, str]]:
    """流式扫描，避免为一百万个 Path 再创建排序列表。"""
    submitted = 0
    for pdf_path in pdf_dir.rglob("*.pdf"):
        lngid = extract_lngid(pdf_path.name)
        if not lngid or lngid in processed_ids:
            continue
        submitted += 1
        yield submitted, pdf_path, lngid
        if limit > 0 and submitted >= limit:
            return


def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def parser_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    worker_id: int,
    mineru_port: int,
    spool_dir: str,
    task_timeout: int,
    parse_retries: int,
) -> None:
    """一个解析进程绑定一个 MinerU 端口；多个进程可共享同一端口。"""
    # 与 parallel_test_100.py 一样延迟导入，避免 spawn 前初始化重依赖。
    import asyncio as worker_asyncio
    from algorithm.knowledge_base.file_parser.service import MinerUParserWithStructure

    spool_root = Path(spool_dir)
    parser = MinerUParserWithStructure(
        mineru_api_base=f"http://127.0.0.1:{mineru_port}",
    )

    try:
        while True:
            try:
                task = task_queue.get(timeout=3.0)
            except queue.Empty:
                continue

            if task is None:
                break

            task_id, pdf_path, lngid = task
            result_queue.put({
                "kind": "started",
                "worker_id": worker_id,
                "task_id": task_id,
                "pdf_path": pdf_path,
                "lngid": lngid,
            })
            started = time.time()
            last_error: Optional[str] = None

            for attempt in range(parse_retries + 1):
                try:
                    async def parse() -> Any:
                        return await worker_asyncio.wait_for(
                            parser.parse_pdf_with_structure(
                                pdf_path, Path(pdf_path).name,
                            ),
                            timeout=task_timeout,
                        )

                    raw_chunks, _ = worker_asyncio.run(parse())
                    spool_path = spool_root / f"{task_id:09d}_{lngid}.pickle"
                    atomic_pickle_dump(raw_chunks, spool_path)
                    result_queue.put({
                        "kind": "parsed",
                        "worker_id": worker_id,
                        "task_id": task_id,
                        "pdf_path": pdf_path,
                        "lngid": lngid,
                        "spool_path": str(spool_path),
                        "parse_seconds": round(time.time() - started, 2),
                    })
                    last_error = None
                    break
                except Exception:
                    last_error = traceback.format_exc()
                    if attempt < parse_retries:
                        time.sleep(min(2 ** attempt, 10))

            if last_error is not None:
                result_queue.put({
                    "kind": "parse_failed",
                    "worker_id": worker_id,
                    "task_id": task_id,
                    "pdf_path": pdf_path,
                    "lngid": lngid,
                    "error": last_error,
                })
    except BaseException:
        result_queue.put({
            "kind": "worker_crashed",
            "worker_id": worker_id,
            "error": traceback.format_exc(),
        })
    finally:
        result_queue.put({"kind": "worker_done", "worker_id": worker_id})


def put_task_until_stopped(
    task_queue: mp.Queue,
    task: Any,
    stop_event: threading.Event,
) -> bool:
    """有界队列写入；周期醒来检查停止标志，避免 feeder 永久卡住。"""
    while not stop_event.is_set():
        try:
            task_queue.put(task, timeout=1.0)
            return True
        except queue.Full:
            continue
    return False


class PipelineState:
    def __init__(self) -> None:
        self.submitted = 0
        self.parsed = 0
        self.success = 0
        self.failed = 0
        self.counts = {key: 0 for key in COUNT_KEYS}
        self.failures: List[Dict[str, Any]] = []
        self.inflight: Dict[int, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def record_failure(self, event: Dict[str, Any], stage: str) -> None:
        async with self.lock:
            self.failed += 1
            self.failures.append({
                "task_id": event.get("task_id"),
                "filename": Path(event.get("pdf_path", "unknown")).name,
                "lngid": event.get("lngid"),
                "stage": stage,
                "error": event.get("error", "unknown error"),
            })
            task_id = event.get("task_id")
            if task_id is not None:
                self.inflight.pop(task_id, None)


async def feed_tasks(
    task_queue: mp.Queue,
    pdf_dir: Path,
    processed_ids: Set[str],
    limit: int,
    parser_workers: int,
    state: PipelineState,
    stop_event: threading.Event,
) -> None:
    try:
        for task_id, pdf_path, lngid in iter_pending_pdfs(
            pdf_dir, processed_ids, limit,
        ):
            task = (task_id, str(pdf_path), lngid)
            accepted = await asyncio.to_thread(
                put_task_until_stopped, task_queue, task, stop_event,
            )
            if not accepted:
                return
            state.submitted += 1
            if state.submitted % 1000 == 0:
                print(f"[FEED] 已提交 {state.submitted} 篇")
    finally:
        if not stop_event.is_set():
            for _ in range(parser_workers):
                await asyncio.to_thread(
                    put_task_until_stopped, task_queue, None, stop_event,
                )


def queue_get_with_timeout(result_queue: mp.Queue) -> Optional[Dict[str, Any]]:
    try:
        return result_queue.get(timeout=3.0)
    except queue.Empty:
        return None


async def dispatch_parser_results(
    result_queue: mp.Queue,
    parsed_queue: asyncio.Queue,
    processes: List[mp.Process],
    index_workers: int,
    state: PipelineState,
    stop_event: threading.Event,
) -> None:
    done_workers: Set[int] = set()
    expected_workers = len(processes)

    while len(done_workers) < expected_workers:
        event = await asyncio.to_thread(queue_get_with_timeout, result_queue)
        if event is None:
            if not any(process.is_alive() for process in processes):
                print("[ERROR] 所有解析 worker 已退出，停止等待结果。")
                stop_event.set()
                break
            continue

        kind = event.get("kind")
        if kind == "started":
            state.inflight[event["task_id"]] = event
        elif kind == "parsed":
            state.parsed += 1
            # 已经安全写入 spool，不再属于 parser worker 的在途任务；
            # 后续由 index worker 接管，失败时仍不会写入 checkpoint。
            state.inflight.pop(event["task_id"], None)
            await parsed_queue.put(event)
        elif kind == "parse_failed":
            await state.record_failure(event, "mineru")
            print(
                f"[PARSE FAIL] #{event['task_id']} "
                f"{Path(event['pdf_path']).name}"
            )
        elif kind == "worker_crashed":
            print(
                f"[WORKER CRASH] worker={event.get('worker_id')}: "
                f"{event.get('error', '').splitlines()[-1:]}"
            )
        elif kind == "worker_done":
            done_workers.add(event["worker_id"])

    # worker 异常退出时，其正在处理但没有结果的任务不写 checkpoint；下次运行会补偿。
    for event in list(state.inflight.values()):
        await state.record_failure(
            {**event, "error": "parser worker exited before returning a result"},
            "worker_exit",
        )

    for _ in range(index_workers):
        await parsed_queue.put(None)


async def index_worker(
    worker_id: int,
    parsed_queue: asyncio.Queue,
    http_client: Any,
    legacy: Any,
    db3_path: str,
    checkpoint_file: Path,
    failure_file: Path,
    checkpoint_lock: asyncio.Lock,
    index_semaphore: asyncio.Semaphore,
    state: PipelineState,
) -> None:
    while True:
        event = await parsed_queue.get()
        if event is None:
            parsed_queue.task_done()
            return

        spool_path = Path(event["spool_path"])
        started = time.time()
        try:
            raw_chunks = await asyncio.to_thread(_load_pickle, spool_path)
            meta = await asyncio.to_thread(
                legacy.load_metadata_by_lngid,
                event["lngid"],
                db3_path=db3_path,
            )
            async with index_semaphore:
                counts = await legacy.import_one_pdf_to_es_optimized(
                    http_client, event["lngid"], raw_chunks, meta,
                )

            # checkpoint 只能在索引阶段完整返回后写入，并由主进程单写者锁保护。
            async with checkpoint_lock:
                with checkpoint_file.open("a", encoding="utf-8") as handle:
                    handle.write(f"{event['lngid']}\n")
                    handle.flush()
                    os.fsync(handle.fileno())

            async with state.lock:
                state.success += 1
                state.inflight.pop(event["task_id"], None)
                for key, value in counts.items():
                    state.counts[key] = state.counts.get(key, 0) + value
                done = state.success + state.failed

            if done % 100 == 0 or state.success <= 10:
                print(
                    f"[OK] index-worker={worker_id} #{event['task_id']} "
                    f"{Path(event['pdf_path']).name} "
                    f"parse={event['parse_seconds']}s "
                    f"index={time.time() - started:.1f}s "
                    f"done={done}/{state.submitted}"
                )
        except Exception:
            failed_event = {**event, "error": traceback.format_exc()}
            await state.record_failure(failed_event, "embedding_or_es")
            async with checkpoint_lock:
                with failure_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "task_id": event["task_id"],
                        "pdf_path": event["pdf_path"],
                        "lngid": event["lngid"],
                        "stage": "embedding_or_es",
                        "error": failed_event["error"],
                    }, ensure_ascii=False) + "\n")
                    handle.flush()
            print(f"[INDEX FAIL] #{event['task_id']} {event['pdf_path']}")
        finally:
            try:
                spool_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[WARN] 无法删除临时文件 {spool_path}: {exc}")
            parsed_queue.task_done()


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


async def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    pdf_dir = Path(args.pdf_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    checkpoint_file = state_dir / "processed_ids.txt"
    failure_file = state_dir / "failures.jsonl"
    stats_file = state_dir / "stats.json"
    spool_dir = state_dir / "spool"

    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF 目录不存在: {pdf_dir}")

    state_dir.mkdir(parents=True, exist_ok=True)
    spool_dir.mkdir(parents=True, exist_ok=True)
    # 上次被强制中断留下的 spool 不代表成功，删除后由 checkpoint 机制重新解析。
    for stale_file in spool_dir.glob("*.pickle*"):
        stale_file.unlink(missing_ok=True)

    ports = [int(value.strip()) for value in args.mineru_ports.split(",") if value.strip()]
    if not ports:
        raise ValueError("至少需要一个 MinerU 端口")

    worker_counts = [
        int(value.strip())
        for value in args.workers_per_port.split(",")
        if value.strip()
    ]
    if len(worker_counts) == 1:
        worker_counts *= len(ports)
    elif len(worker_counts) != len(ports):
        raise ValueError(
            "workers-per-port 必须是一个整数，或与 mineru-ports 数量相同的列表"
        )
    if any(count < 1 for count in worker_counts):
        raise ValueError("每个端口的 worker 数必须 >= 1")

    worker_ports = [
        port
        for port, worker_count in zip(ports, worker_counts)
        for _ in range(worker_count)
    ]
    parser_workers = len(worker_ports)

    processed_ids = load_processed_ids(checkpoint_file)
    print("=" * 72)
    print("[MILLION PDF PIPELINE]")
    print(f"PDF目录: {pdf_dir}")
    print(f"已完成: {len(processed_ids)}")
    print(f"任务上限: {args.limit}")
    port_worker_config = ", ".join(
        f"{port}={count}" for port, count in zip(ports, worker_counts)
    )
    print(f"MinerU端口/worker: {port_worker_config}")
    print(f"解析进程总数: {parser_workers}")
    print(f"索引协程: {args.index_workers}")
    print(f"进程队列上限: {args.task_queue_size}")
    print(f"解析结果队列上限: {args.parsed_queue_size}")
    print(f"状态目录: {state_dir}")
    print("=" * 72)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=args.task_queue_size)
    result_queue = ctx.Queue(maxsize=max(args.parsed_queue_size * 2, 8))
    processes: List[mp.Process] = []

    for worker_id, port in enumerate(worker_ports):
        process = ctx.Process(
            target=parser_worker,
            name=f"mineru-parser-{worker_id}-port-{port}",
            args=(
                task_queue, result_queue, worker_id, port, str(spool_dir),
                args.mineru_timeout, args.parse_retries,
            ),
        )
        process.start()
        processes.append(process)

    # 子进程启动后才导入旧脚本，避免主进程提前加载 MinerU 相关依赖。
    from algorithm.ai_tools import parallel_test_10000 as legacy
    import httpx

    legacy.DB3_PATH = args.db3_path
    state = PipelineState()
    parsed_queue: asyncio.Queue = asyncio.Queue(maxsize=args.parsed_queue_size)
    checkpoint_lock = asyncio.Lock()
    index_semaphore = asyncio.Semaphore(args.index_workers)
    stop_event = threading.Event()
    started = time.time()

    timeout = httpx.Timeout(args.http_timeout)
    limits = httpx.Limits(
        max_connections=max(args.index_workers * 2, 10),
        max_keepalive_connections=max(args.index_workers, 5),
    )

    try:
        async with httpx.AsyncClient(
            trust_env=False, timeout=timeout, limits=limits,
        ) as http_client:
            feeder = asyncio.create_task(feed_tasks(
                task_queue, pdf_dir, processed_ids, args.limit,
                parser_workers, state, stop_event,
            ))
            dispatcher = asyncio.create_task(dispatch_parser_results(
                result_queue, parsed_queue, processes, args.index_workers,
                state, stop_event,
            ))
            indexers = [
                asyncio.create_task(index_worker(
                    worker_id, parsed_queue, http_client, legacy,
                    args.db3_path, checkpoint_file, failure_file,
                    checkpoint_lock, index_semaphore, state,
                ))
                for worker_id in range(args.index_workers)
            ]

            await dispatcher
            if stop_event.is_set() and not feeder.done():
                feeder.cancel()
            await asyncio.gather(feeder, return_exceptions=True)
            await parsed_queue.join()
            await asyncio.gather(*indexers)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("[STOP] 收到中断，停止提交新任务并终止解析进程。")
        stop_event.set()
    finally:
        stop_event.set()
        for process in processes:
            process.join(timeout=5)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        task_queue.close()
        result_queue.close()

    elapsed = time.time() - started
    summary = {
        "version": "million_pdf_pipeline_v1",
        "submitted": state.submitted,
        "parsed": state.parsed,
        "success": state.success,
        "failed": state.failed,
        "elapsed_seconds": round(elapsed, 2),
        "throughput_per_hour": (
            round(state.success / elapsed * 3600, 2) if elapsed > 0 else 0
        ),
        "counts": state.counts,
        "recent_failures": state.failures[-1000:],
        "unfinished_tasks": list(state.inflight.values())[:1000],
    }
    with stats_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 72)
    print(
        f"完成: success={state.success}, failed={state.failed}, "
        f"submitted={state.submitted}, elapsed={elapsed / 3600:.2f}h, "
        f"throughput={summary['throughput_per_hour']} 篇/小时"
    )
    print(f"统计文件: {stats_file}")
    print("=" * 72)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="百万 PDF：多 MinerU 进程 + 异步 Embedding/ES 流水线",
    )
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--db3-path", default=str(DEFAULT_DB3_PATH))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument(
        "--mineru-ports", default="8060",
        help="MinerU 服务端口，逗号分隔，例如 8060,8061",
    )
    parser.add_argument(
        "--workers-per-port", default="1",
        help=(
            "每个端口的解析 worker 数；可给统一值 4，或按端口给列表 2,3。"
            "总解析进程数为这些数值之和"
        ),
    )
    parser.add_argument("--index-workers", type=int, default=6)
    parser.add_argument("--task-queue-size", type=int, default=32)
    parser.add_argument("--parsed-queue-size", type=int, default=12)
    parser.add_argument("--mineru-timeout", type=int, default=600)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--parse-retries", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.index_workers < 1:
        raise ValueError("index-workers 必须 >= 1")
    if args.task_queue_size < 1 or args.parsed_queue_size < 1:
        raise ValueError("队列大小必须 >= 1")
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
