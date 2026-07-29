#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Million-PDF pipeline using local spawn-based MinerU workers (Robust Version).

Improvements:
1. Checkpoint/Resumption via pipeline_status.jsonl.
2. Non-blocking queues to prevent deadlocks.
3. Dedicated IO thread pool for blocking operations.
4. GPU memory monitoring via pynvml.
5. Comprehensive logging and error state recording.
6. Strict resource cleanup.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sys
import threading
import time
import traceback
try:
    import resource
except ImportError:  # Windows does not provide the resource module.
    resource = None
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

# 尝试导入 pynvml 用于 GPU 监控
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not found, GPU monitoring disabled.")

# Setup Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "algorithm"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "500pdf"
DEFAULT_DB3_PATH = Path(os.environ.get(
    "DB3_PATH",
    "/home/vscodeuser/yanjiushequ/Encyclopedia_project/zkyrjyjs_50w_20260707.db3",
))
DEFAULT_STATE_DIR = PROJECT_ROOT / "data" / "million_pdf_vec_result_local_v2"

COUNT_KEYS = (
    "title", "abstract", "keyword", "fulltext",
    "paragraph", "sentence", "reference",
)
SCAN_DONE = object()

# Configure Logging
def setup_logging(state_dir: Path, level=logging.INFO):
    log_file = state_dir / "pipeline.log"
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s %(name)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("MillionPDF")

logger = logging.getLogger("MillionPDF")


def _rate_per_hour(completed: int, started_at: float, now: Optional[float] = None) -> float:
    if completed <= 0 or started_at <= 0:
        return 0.0
    elapsed = (now or time.monotonic()) - started_at
    return completed / elapsed * 3600 if elapsed > 0 else 0.0


def _window_rate_per_hour(timestamps: Deque[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed * 3600 if elapsed > 0 else 0.0


# --- Data Structures ---

@dataclass(frozen=True)
class ParseTask:
    key: str
    sequence: int
    pdf_path: str
    lngid: str
    attempt: int = 1

@dataclass
class WorkerSlot:
    slot_id: int
    gpu_id: int
    generation: int = 0
    process: Optional[mp.Process] = None
    command_queue: Optional[mp.Queue] = None
    ready: bool = False
    lease: Optional[ParseTask] = None
    leased_at: float = 0.0
    disabled: bool = False
    crash_count: int = 0
    last_crash_time: float = 0.0

@dataclass
class PipelineState:
    discovered: int = 0
    submitted: int = 0
    parse_attempts: int = 0
    parsed: int = 0
    success: int = 0
    parse_failures: int = 0
    index_failures: int = 0
    timeouts: int = 0
    worker_deaths: int = 0
    worker_restarts: int = 0
    counts: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in COUNT_KEYS})
    per_gpu: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    started_at: float = 0.0
    parse_started_at: float = 0.0
    index_started_at: float = 0.0
    parse_recent: Deque[float] = field(default_factory=lambda: deque(maxlen=101))
    index_recent: Deque[float] = field(default_factory=lambda: deque(maxlen=101))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

# --- Status & Checkpointing ---

class StatusTracker:
    """Tracks processing status to enable resumption and failure recording."""
    def __init__(self, state_dir: Path):
        self.status_file = state_dir / "pipeline_status.jsonl"
        self.error_file = state_dir / "error_details.jsonl"
        self.lock = asyncio.Lock()
        self._cache: Dict[str, Dict] = {}
    
    async def load(self):
        """Load status from disk into memory cache."""
        if not self.status_file.exists():
            return
        logger.info(f"Loading status from {self.status_file}...")
        try:
            # 使用线程池读取大文件避免阻塞事件循环
            def _read():
                data = {}
                with self.status_file.open("r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            data[obj["lngid"]] = obj
                        except json.JSONDecodeError:
                            pass
                return data
            
            self._cache = await asyncio.to_thread(_read)
            logger.info(f"Loaded status for {len(self._cache)} items.")
        except Exception as e:
            logger.error(f"Failed to load status file: {e}")

    def get_status(self, lngid: str) -> Optional[Dict]:
        return self._cache.get(lngid)

    async def record_success(self, lngid: str):
        async with self.lock:
            record = {
                "lngid": lngid,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            self._cache[lngid] = record
            await self._append_log(record)

    async def record_failure(self, lngid: str, stage: str, error: str, attempt: int, pdf_path: str = ""):
        async with self.lock:
            record = {
                "lngid": lngid,
                "status": "failure",
                "stage": stage, # "parse" or "index"
                "attempt": attempt,
                "error": error,
                "pdf_path": pdf_path,
                "timestamp": datetime.now().isoformat()
            }
            self._cache[lngid] = record
            
            # Write to both generic status and detailed error log
            await self._append_log(record)
            
            error_detail = {**record, "traceback": error} # Put full error in detail file
            await self._append_log(error_detail, self.error_file)

    async def _append_log(self, record: Dict, target_file: Optional[Path] = None):
        target = target_file or self.status_file
        def _write():
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        await asyncio.to_thread(_write)

    def is_done(self, lngid: str) -> bool:
        s = self._cache.get(lngid)
        return s is not None and s.get("status") == "success"
    
    def should_skip(self, lngid: str, max_attempts: int) -> bool:
        s = self._cache.get(lngid)
        if not s: return False # New task
        if s.get("status") == "success": return True
        # If failed and attempts exceeded limit, skip
        if s.get("status") == "failure" and s.get("attempt", 0) >= max_attempts:
            return True
        return False

# --- Utilities ---

def setup_worker_env(gpu_id: int, vram_size: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
    os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
    # 设为稍大一点的值，或者根据实际显存调整
    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram_size)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def extract_lngid(filename: str) -> str:
    import re
    base = os.path.basename(str(filename))
    base = re.sub(r"\.(pdf|docx?|txt|md)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^vec_[a-f0-9]+_", "", base)
    base = re.sub(r"^temp_[a-f0-9_-]+_", "", base)
    return base.strip()

def task_key(pdf_path: Path, root: Path) -> str:
    try:
        relative = pdf_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(pdf_path.resolve())
    return hashlib.sha256(relative.encode()).hexdigest()[:20]

def get_gpu_usage(gpu_id: int) -> Optional[float]:
    """Returns GPU memory usage ratio (0.0 - 1.0)."""
    if not PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / info.total
    except Exception:
        return None

def atomic_pickle_dump(value: Any, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)

# --- MinerU Worker ---

def analyze_pdf_locally(
    pdf_path: str,
    filename: str,
    work_dir: Path,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
) -> List[Any]:
    """Run local MinerU pipeline. Isolate imports."""
    from mineru.cli.common import prepare_env, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
    from algorithm.knowledge_base.file_parser.markdown_structure_adapter import MarkdownStructureAdapter

    pdf_bytes = read_fn(pdf_path)
    file_stem = Path(filename).stem
    image_dir, markdown_dir = prepare_env(str(work_dir), file_stem, "auto")
    image_writer = FileBasedDataWriter(image_dir)
    markdown_writer = FileBasedDataWriter(markdown_dir)
    
    # Pipeline logic
    infer, image_lists, pdf_docs, langs, ocr_flags = pipeline_doc_analyze(
        [pdf_bytes], [lang], parse_method="auto",
        formula_enable=formula_enable, table_enable=table_enable,
    )
    middle_json = result_to_middle_json(
        infer[0], image_lists[0], pdf_docs[0], image_writer,
        langs[0], ocr_flags[0], formula_enable,
    )
    pdf_info = middle_json["pdf_info"]
    image_name = os.path.basename(image_dir)
    markdown = union_make(pdf_info, MakeMode.MM_MD, image_name)
    if not markdown.strip():
        raise RuntimeError("MinerU generated empty Markdown")
    
    markdown_writer.write_string(f"{file_stem}.md", markdown)
    chunks, _ = MarkdownStructureAdapter().build_chunks(markdown, filename)
    return chunks

def local_parser_worker(
    command_queue: mp.Queue,
    result_queue: mp.Queue,
    slot_id: int,
    generation: int,
    gpu_id: int,
    vram_size: int,
    spool_dir: str,
    work_root: str,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
) -> None:
    setup_worker_env(gpu_id, vram_size)
    # Signal ready
    result_queue.put({
        "kind": "worker_ready", "slot_id": slot_id,
        "generation": generation, "gpu_id": gpu_id, "pid": os.getpid(),
    })
    
    try:
        while True:
            try:
                command = command_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            if command is None:
                break
            
            task_data = command["task"]
            token = command["token"]
            task = ParseTask(**task_data)
            
            result_queue.put({
                "kind": "started", "slot_id": slot_id, "generation": generation,
                "token": token, "key": task.key, "lngid": task.lngid,
            })
            
            attempt_dir = Path(work_root) / token
            attempt_dir.mkdir(parents=True, exist_ok=True)
            spool_path = Path(spool_dir) / f"{token}.pickle"
            
            try:
                chunks = analyze_pdf_locally(
                    task.pdf_path, Path(task.pdf_path).name, attempt_dir, lang,
                    formula_enable, table_enable,
                )
                atomic_pickle_dump(chunks, spool_path)
                
                result_queue.put({
                    "kind": "parsed", "slot_id": slot_id, "generation": generation,
                    "token": token, "task": task_data, "gpu_id": gpu_id,
                    "spool_path": str(spool_path),
                })
            except BaseException:
                result_queue.put({
                    "kind": "parse_failed", "slot_id": slot_id,
                    "generation": generation, "token": token,
                    "task": task_data, "error": traceback.format_exc(),
                })
            finally:
                # Cleanup attempt dir immediately after success or failure
                # Spool file is kept for index worker to consume
                if attempt_dir.exists():
                    shutil.rmtree(attempt_dir, ignore_errors=True)
                    
    except BaseException:
        result_queue.put({
            "kind": "worker_crashed", "slot_id": slot_id,
            "generation": generation, "error": traceback.format_exc(),
        })
    finally:
        result_queue.put({
            "kind": "worker_stopped", "slot_id": slot_id,
            "generation": generation,
        })

# --- Supervisor Logic ---

class LocalSupervisor:
    def __init__(
        self,
        ctx: mp.context.BaseContext,
        gpu_slots: List[int],
        result_queue: mp.Queue,
        spool_dir: Path,
        work_dir: Path,
        args: argparse.Namespace,
        state: PipelineState,
        status_tracker: StatusTracker,
    ) -> None:
        self.ctx = ctx
        self.result_queue = result_queue
        self.spool_dir = spool_dir
        self.work_dir = work_dir
        self.args = args
        self.state = state
        self.status_tracker = status_tracker
        self.slots = [WorkerSlot(index, gpu) for index, gpu in enumerate(gpu_slots)]
        self.retry: Deque[ParseTask] = deque()
        self.active_tokens: Dict[int, str] = {}

    def spawn(self, slot: WorkerSlot, restart: bool = False) -> None:
        slot.generation += 1
        # Use a larger queue to buffer backpressure, but manage it carefully
        slot.command_queue = self.ctx.Queue(maxsize=2) 
        slot.ready = False
        slot.lease = None
        slot.leased_at = 0.0
        process = self.ctx.Process(
            target=local_parser_worker,
            name=f"local-mineru-slot-{slot.slot_id}-gpu-{slot.gpu_id}-g{slot.generation}",
            args=(
                slot.command_queue, self.result_queue, slot.slot_id,
                slot.generation, slot.gpu_id, self.args.vram,
                str(self.spool_dir), str(self.work_dir), self.args.lang,
                self.args.formula_enable, self.args.table_enable,
            ),
        )
        process.start()
        slot.process = process
        if restart:
            self.state.worker_restarts += 1
            logger.info(f"Restarted slot {slot.slot_id} (GPU {slot.gpu_id}) Gen {slot.generation}")

    async def stop_process(self, slot: WorkerSlot) -> None:
        process = slot.process
        if process and process.is_alive():
            process.terminate()
            try:
                await asyncio.to_thread(process.join, self.args.worker_kill_grace)
            except Exception:
                pass
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5.0)
        
        if slot.command_queue is not None:
            try:
                slot.command_queue.close()
            except Exception:
                pass
        slot.process = None
        slot.command_queue = None
        slot.ready = False

    def token_for(self, slot: WorkerSlot, task: ParseTask) -> str:
        return f"{task.key}.a{task.attempt}.s{slot.slot_id}.g{slot.generation}"

    async def fail_lease(self, slot: WorkerSlot, reason: str, timeout: bool = False) -> None:
        task = slot.lease
        token = self.active_tokens.pop(slot.slot_id, None)
        
        # Cleanup files
        if token:
            for path in [
                self.spool_dir / f"{token}.pickle",
                self.spool_dir / f"{token}.pickle.tmp",
                self.work_dir / token
            ]:
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)

        slot.lease = None
        slot.leased_at = 0.0
        if task is None:
            return

        # Record failure
        await self.status_tracker.record_failure(
            task.lngid, "parse", reason, task.attempt, task.pdf_path
        )

        if timeout:
            self.state.timeouts += 1
        
        # Retry logic
        if task.attempt < self.args.max_parse_attempts:
            self.retry.append(ParseTask(
                task.key, task.sequence, task.pdf_path, task.lngid, task.attempt + 1,
            ))
        else:
            self.state.parse_failures += 1
            logger.error(f"Permanent Parse Failure: {task.lngid} (Attempt {task.attempt})")

    async def restart(self, slot: WorkerSlot, reason: str, timeout: bool = False) -> None:
        await self.fail_lease(slot, reason, timeout)
        await self.stop_process(slot)
        
        # Simple backoff: check crash frequency
        now = time.monotonic()
        if now - slot.last_crash_time < 60:
            slot.crash_count += 1
            if slot.crash_count > 3:
                logger.error(f"Disabling slot {slot.slot_id} due to frequent crashes.")
                slot.disabled = True
                return
        else:
            slot.crash_count = 0
        
        slot.last_crash_time = now
        self.spawn(slot, restart=True)

    async def shutdown(self) -> None:
        logger.info("Shutting down workers...")
        for slot in self.slots:
            if slot.command_queue and slot.process and slot.process.is_alive():
                try:
                    slot.command_queue.put_nowait(None)
                except (queue.Full, OSError):
                    pass
        for slot in self.slots:
            await self.stop_process(slot)

async def run_parser_supervisor(
    supervisor: LocalSupervisor,
    pending_queue: asyncio.Queue,
    parsed_queue: asyncio.Queue,
    state: PipelineState,
) -> None:
    for slot in supervisor.slots:
        supervisor.spawn(slot)

    scan_done = False
    buffered: Deque[ParseTask] = deque()
    
    while True:
        # 1. Handle Worker Events (Non-blocking get via to_thread with timeout)
        try:
            event = await asyncio.to_thread(supervisor.result_queue.get, True, 0.1)
        except queue.Empty:
            event = None

        if event:
            slot_idx = event["slot_id"]
            if slot_idx >= len(supervisor.slots): continue # Stale event from old slot
            
            slot = supervisor.slots[slot_idx]
            
            # Generation check
            if event.get("generation") != slot.generation:
                cleanup_stale_event_artifacts(supervisor, event)
                logger.warning(
                    "Ignored stale worker event: slot=%s event_gen=%s current_gen=%s kind=%s",
                    slot.slot_id, event.get("generation"), slot.generation, event.get("kind"),
                )
                continue
            
            kind = event["kind"]
            token = event.get("token")
            current_token = supervisor.active_tokens.get(slot.slot_id)

            if kind == "worker_ready":
                slot.ready = True
            elif kind == "started" and token == current_token:
                slot.leased_at = time.monotonic()
            elif kind == "parsed" and token == current_token:
                task = slot.lease
                slot.lease = None
                slot.leased_at = 0.0
                supervisor.active_tokens.pop(slot.slot_id, None)
                slot.ready = True
                now_parsed = time.monotonic()
                state.parsed += 1
                state.per_gpu[str(slot.gpu_id)] += 1
                state.parse_recent.append(now_parsed)
                await parsed_queue.put(event)
                if state.parsed <= 10 or state.parsed % supervisor.args.progress_interval == 0:
                    logger.info(
                        "[PARSE RATE] parsed=%d/%d attempts=%d "
                        "avg=%.1f PDF/h (%.2fs/PDF) recent_%d=%.1f PDF/h "
                        "pending_index=%d gpu=%s",
                        state.parsed, state.discovered, state.parse_attempts,
                        _rate_per_hour(state.parsed, state.parse_started_at, now_parsed),
                        (now_parsed - state.parse_started_at) / state.parsed,
                        max(len(state.parse_recent) - 1, 0),
                        _window_rate_per_hour(state.parse_recent),
                        parsed_queue.qsize(), dict(state.per_gpu),
                    )
            elif kind == "parse_failed" and token == current_token:
                await supervisor.fail_lease(slot, event.get("error", "parse failure"))
                slot.ready = True
            elif kind in ("worker_crashed", "worker_stopped"):
                if slot.process and slot.process.exitcode not in (None, 0):
                    state.worker_deaths += 1
        
        # 2. Monitor Worker Health
        now = time.monotonic()
        for slot in supervisor.slots:
            process = slot.process
            if not process and not slot.disabled and (not scan_done or buffered or supervisor.retry or not pending_queue.empty()):
                supervisor.spawn(slot, restart=True)
            elif process and not process.is_alive() and not slot.disabled:
                if slot.lease is not None:
                    state.worker_deaths += 1
                    await supervisor.restart(slot, f"worker exited code={process.exitcode}")
                else:
                    await supervisor.stop_process(slot)
                    supervisor.spawn(slot, restart=True)
            elif slot.lease and slot.leased_at and now - slot.leased_at > supervisor.args.parse_timeout:
                await supervisor.restart(slot, "parse timeout", timeout=True)

        # 3. Dispatch Tasks
        # Refill buffer
        while not scan_done and len(buffered) < supervisor.args.task_queue_size:
            try:
                item = pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is SCAN_DONE:
                scan_done = True
            else:
                buffered.append(item)
            pending_queue.task_done()

        # Assign tasks
        for slot in supervisor.slots:
            if not slot.ready or slot.lease is not None or slot.disabled:
                continue
            
            # GPU Throttling Check (Optional optimization)
            if PYNVML_AVAILABLE:
                usage = get_gpu_usage(slot.gpu_id)
                if usage and usage > 0.95:
                    # Skip this slot for a moment to let GC happen
                    continue

            task = supervisor.retry.popleft() if supervisor.retry else (buffered.popleft() if buffered else None)
            if task is None:
                continue
                
            token = supervisor.token_for(slot, task)
            # Register the lease before handing work to the child so an immediate
            # started/parsed event can never arrive without a matching token.
            slot.lease = task
            slot.leased_at = now
            slot.ready = False
            supervisor.active_tokens[slot.slot_id] = token
            try:
                slot.command_queue.put_nowait({"task": task.__dict__, "token": token})
                state.submitted += 1
                state.parse_attempts += 1
            except queue.Full:
                supervisor.active_tokens.pop(slot.slot_id, None)
                slot.lease = None
                slot.leased_at = 0.0
                slot.ready = True
                supervisor.retry.appendleft(task)

        # 4. Check Completion
        if (
            scan_done and not buffered and not supervisor.retry
            and all(slot.lease is None for slot in supervisor.slots)
        ):
            break
        
        if not any(s.process and s.process.is_alive() for s in supervisor.slots) and not scan_done:
            raise RuntimeError("All workers died and scan not done")
            
        await asyncio.sleep(0.05)

    await supervisor.shutdown()

# --- Indexing Worker ---

async def index_worker(
    worker_id: int,
    parsed_queue: asyncio.Queue,
    http_client: Any,
    legacy: Any,
    db3_path: str,
    io_executor: ThreadPoolExecutor,
    status_tracker: StatusTracker,
    state: PipelineState,
    args: argparse.Namespace,
) -> None:
    while True:
        event = await parsed_queue.get()
        if event is None:
            parsed_queue.task_done()
            return
        
        spool_path = Path(event["spool_path"])
        task = ParseTask(**event["task"])
        
        # Double check if already done (race condition protection)
        if status_tracker.is_done(task.lngid):
            spool_path.unlink(missing_ok=True)
            parsed_queue.task_done()
            continue

        try:
            # Offload blocking IO to thread pool
            chunks = await asyncio.to_thread(load_pickle, spool_path)
            
            # Load metadata (Optimized SQLite connection)
            meta = await asyncio.to_thread(
                legacy.load_metadata_by_lngid, task.lngid, db3_path=str(db3_path),
            )
            
            # Vectorize title/abstract/keyword/full text/chunks and write seven ES indexes.
            counts = await legacy.import_one_pdf_to_es_optimized(
                http_client, task.lngid, chunks, meta,
            )

            # Persist success only after the full indexing coroutine returns.
            await status_tracker.record_success(task.lngid)
            now_indexed = time.monotonic()
            async with state.lock:
                state.success += 1
                state.index_recent.append(now_indexed)
                for key, value in counts.items():
                    state.counts[key] = state.counts.get(key, 0) + value
                completed = state.success
                count_snapshot = dict(state.counts)

            if completed <= 10 or completed % args.progress_interval == 0:
                logger.info(
                    "[INDEX RATE] worker=%d lngid=%s indexed=%d/%d "
                    "avg=%.1f PDF/h (%.2fs/PDF) recent_%d=%.1f PDF/h queue=%d counts=%s",
                    worker_id, task.lngid, completed, state.discovered,
                    _rate_per_hour(completed, state.index_started_at, now_indexed),
                    (now_indexed - state.index_started_at) / completed,
                    max(len(state.index_recent) - 1, 0),
                    _window_rate_per_hour(state.index_recent),
                    parsed_queue.qsize(), count_snapshot,
                )
                
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"[INDEX FAIL] worker={worker_id} lngid={task.lngid} error={str(e)}")
            await status_tracker.record_failure(
                task.lngid, "index", str(e), task.attempt, task.pdf_path
            )
            async with state.lock:
                state.index_failures += 1
        finally:
            # Cleanup spool file
            spool_path.unlink(missing_ok=True)
            parsed_queue.task_done()

def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def cleanup_stale_event_artifacts(
    supervisor: LocalSupervisor,
    event: Dict[str, Any],
) -> None:
    """Remove files produced by an obsolete worker generation."""
    token = event.get("token")
    if not token:
        return
    candidates = [
        supervisor.spool_dir / f"{token}.pickle",
        supervisor.spool_dir / f"{token}.pickle.tmp",
    ]
    event_spool = event.get("spool_path")
    if event_spool:
        candidates.append(Path(event_spool))
    for path in candidates:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Failed to remove stale spool %s: %s", path, exc)
    shutil.rmtree(supervisor.work_dir / token, ignore_errors=True)
    logger.info("Cleaned stale artifacts: token=%s", token)


def cleanup_previous_run_artifacts(spool_dir: Path, work_dir: Path) -> Tuple[int, int]:
    """Remove uncheckpointed handoff files left by an interrupted prior run."""
    spool_count = 0
    work_count = 0
    for pattern in ("*.pickle", "*.pickle.tmp"):
        for path in spool_dir.glob(pattern):
            try:
                path.unlink()
                spool_count += 1
            except OSError as exc:
                logger.warning("Failed to remove old spool %s: %s", path, exc)
    for path in work_dir.iterdir():
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            work_count += 1
        except OSError as exc:
            logger.warning("Failed to remove old work artifact %s: %s", path, exc)
    return spool_count, work_count


# --- Main Pipeline ---

async def feed_tasks(
    pending_queue: asyncio.Queue,
    pdf_dir: Path,
    limit: int,
    state: PipelineState,
    status_tracker: StatusTracker,
    max_attempts: int,
) -> None:
    sequence = 0
    # Counters for logging
    skipped_count = 0
    retry_count = 0
    
    for pdf_path in pdf_dir.rglob("*.pdf"):
        lngid = extract_lngid(pdf_path.name)
        if not lngid:
            continue
        
        # Check status
        if status_tracker.is_done(lngid):
            skipped_count += 1
            continue
            
        if status_tracker.should_skip(lngid, max_attempts):
            skipped_count += 1
            continue
            
        # Prepare task
        # Check if we have a previous failure to increment attempt count
        status = status_tracker.get_status(lngid)
        attempt = 1
        if status and status.get("status") == "failure":
            attempt = status.get("attempt", 1) + 1
            retry_count += 1
            
        sequence += 1
        state.discovered += 1
        await pending_queue.put(ParseTask(
            task_key(pdf_path, pdf_dir), sequence, str(pdf_path), lngid, attempt
        ))
        
        if limit > 0 and sequence >= limit:
            break
            
    await pending_queue.put(SCAN_DONE)
    logger.info(f"Scan done. Discovered: {state.discovered}, Skipped: {skipped_count}, Retries: {retry_count}")

async def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    global logger
    # Setup Environment (resource is unavailable on Windows).
    if resource is not None:
        try:
            _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
        except Exception:
            logger.warning("Could not increase file descriptor limit")

    pdf_dir = Path(args.pdf_dir).resolve()
    state_dir = Path(args.state_dir).resolve()
    if not pdf_dir.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    state_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging
    logger = setup_logging(state_dir)
    logger.info("=" * 72)
    logger.info("Million PDF Local MinerU Pipeline V2 Started")
    logger.info(f"PDF Dir: {pdf_dir}")
    logger.info(f"State Dir: {state_dir}")
    logger.info(
        "Config: limit=%d GPUs=%s workers_per_gpu=%s vram=%dGB "
        "index_workers=%d progress_interval=%d",
        args.limit, args.gpus, args.workers_per_gpu, args.vram,
        args.index_workers, args.progress_interval,
    )
    logger.info("=" * 72)

    spool_dir = state_dir / "spool" / "ready"
    work_dir = state_dir / "spool" / "work"
    spool_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    old_spools, old_work = cleanup_previous_run_artifacts(spool_dir, work_dir)
    if old_spools or old_work:
        logger.warning(
            "Cleaned artifacts from interrupted prior run: spool=%d work=%d",
            old_spools, old_work,
        )

    # Initialize Components
    status_tracker = StatusTracker(state_dir)
    await status_tracker.load()
    
    gpu_slots = parse_worker_config(args.gpus, args.workers_per_gpu)
    
    # Shared IO Executor for blocking operations
    io_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="io_")
    
    # Queues
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=args.result_queue_size)
    pending_queue: asyncio.Queue = asyncio.Queue(maxsize=args.task_queue_size)
    parsed_queue: asyncio.Queue = asyncio.Queue(maxsize=args.parsed_queue_size)
    
    state = PipelineState()
    state.started_at = time.monotonic()
    state.parse_started_at = state.started_at
    state.index_started_at = state.started_at

    # Supervisor
    supervisor = LocalSupervisor(
        ctx, gpu_slots, result_queue, spool_dir, work_dir, args, state, status_tracker
    )

    # Feeder
    feeder = asyncio.create_task(feed_tasks(
        pending_queue, pdf_dir, args.limit, state, status_tracker, args.max_parse_attempts
    ))
    
    # Parser
    parser_task = asyncio.create_task(run_parser_supervisor(
        supervisor, pending_queue, parsed_queue, state
    ))

    # Give the supervisor a chance to spawn workers before importing the legacy
    # indexing module, which indirectly imports heavier application modules.
    await asyncio.sleep(0)
    from algorithm.ai_tools import parallel_test_10000 as legacy
    import httpx

    # Indexing
    timeout = httpx.Timeout(args.http_timeout)
    limits = httpx.Limits(
        max_connections=max(args.index_workers * 4, 20),
        max_keepalive_connections=max(args.index_workers, 10),
    )
    
    started = time.time()
    try:
        async with httpx.AsyncClient(
            trust_env=False, timeout=timeout, limits=limits,
        ) as client:
            indexers = [
                asyncio.create_task(index_worker(
                    i, parsed_queue, client, legacy, args.db3_path,
                    io_executor, status_tracker, state, args
                )) for i in range(args.index_workers)
            ]
            
            # Wait for feeder and parser to finish
            await asyncio.gather(feeder, parser_task)
            
            # Signal indexers to stop
            for _ in indexers:
                await parsed_queue.put(None)
                
            # Wait for indexers
            await parsed_queue.join()
            await asyncio.gather(*indexers)
            
    except BaseException:
        logger.critical("Pipeline crashed, shutting down...")
        await supervisor.shutdown()
        raise
    finally:
        result_queue.close()
        io_executor.shutdown(wait=True)

    elapsed = time.time() - started
    final_parse_rate = _rate_per_hour(state.parsed, state.parse_started_at)
    final_index_rate = _rate_per_hour(state.success, state.index_started_at)
    logger.info("=" * 72)
    logger.info("[SUMMARY] 本地 MinerU 百万 PDF 流水线")
    logger.info(
        "解析完成: %d | 入库成功: %d | 解析失败: %d | 入库失败: %d | 超时: %d",
        state.parsed, state.success, state.parse_failures,
        state.index_failures, state.timeouts,
    )
    logger.info(
        "总耗时: %.1fs (%.2fh) | 解析速率: %.1f PDF/h | "
        "端到端入库速率: %.1f PDF/h | 平均成功任务: %.2fs/PDF",
        elapsed, elapsed / 3600, final_parse_rate, final_index_rate,
        elapsed / state.success if state.success else 0.0,
    )
    logger.info("各索引写入量: %s", dict(state.counts))
    logger.info("每 GPU 解析量: %s", dict(state.per_gpu))
    logger.info("=" * 72)
    summary = {
        "version": "million_pdf_local_pipeline_v2",
        "discovered": state.discovered,
        "submitted": state.submitted,
        "parse_attempts": state.parse_attempts,
        "parsed": state.parsed,
        "success": state.success,
        "parse_failures": state.parse_failures,
        "index_failures": state.index_failures,
        "timeouts": state.timeouts,
        "worker_deaths": state.worker_deaths,
        "worker_restarts": state.worker_restarts,
        "elapsed_seconds": round(elapsed, 2),
        "parse_throughput_per_hour": round(final_parse_rate, 2),
        "throughput_per_hour": round(final_index_rate, 2),
        "average_seconds_per_success": round(elapsed / state.success, 2) if state.success else 0,
        "counts": state.counts,
        "per_gpu": dict(state.per_gpu),
    }
    
    stats_file = state_dir / "stats.json"
    with stats_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info("Pipeline Finished.")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

# --- CLI & Helpers ---

def parse_worker_config(gpus_text: str, workers_text: str) -> List[int]:
    gpus = [int(v.strip()) for v in gpus_text.split(",") if v.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPUs must be unique and non-empty")
    counts = [int(v.strip()) for v in workers_text.split(",") if v.strip()]
    if len(counts) == 1:
        counts *= len(gpus)
    elif len(counts) != len(gpus):
        raise ValueError("Workers-per-gpu count mismatch")
    if any(count < 1 for count in counts):
        raise ValueError("Workers-per-gpu values must be >= 1")
    return [g for g, c in zip(gpus, counts) for _ in range(c)]

def _add_boolean_argument(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    """Add --feature/--no-feature flags on Python versions before 3.9 too."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Million PDF Local MinerU Pipeline V2")
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--db3-path", default=str(DEFAULT_DB3_PATH))
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers-per-gpu", default="1")
    parser.add_argument("--vram", type=int, default=8)
    parser.add_argument("--lang", default="zh")
    _add_boolean_argument(parser, "formula-enable", True, "启用公式识别")
    _add_boolean_argument(parser, "table-enable", True, "启用表格识别")
    parser.add_argument("--parse-timeout", type=float, default=600.0)
    parser.add_argument("--max-parse-attempts", type=int, default=2)
    parser.add_argument("--worker-kill-grace", type=float, default=10.0)
    parser.add_argument("--task-queue-size", type=int, default=64) # Increased buffer
    parser.add_argument("--result-queue-size", type=int, default=64)
    parser.add_argument("--parsed-queue-size", type=int, default=32)
    parser.add_argument("--index-workers", type=int, default=6)
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument(
        "--progress-interval", type=int, default=100,
        help="每完成多少篇打印一次解析和入库速率",
    )
    return parser

def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "vram": args.vram,
        "parse-timeout": args.parse_timeout,
        "max-parse-attempts": args.max_parse_attempts,
        "task-queue-size": args.task_queue_size,
        "result-queue-size": args.result_queue_size,
        "parsed-queue-size": args.parsed_queue_size,
        "index-workers": args.index_workers,
        "progress-interval": args.progress_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"这些参数必须大于 0: {', '.join(invalid)}")
    parse_worker_config(args.gpus, args.workers_per_gpu)


def main():
    args = build_parser().parse_args()
    validate_args(args)
    asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
