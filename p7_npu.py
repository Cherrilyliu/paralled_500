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
import re
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

# ── GPU/NPU 监控（可选）────────────────────────────────────────
# pynvml 用于在分配任务前检查 NVIDIA GPU 显存占用，超过 95% 则跳过该 slot
# 华为昇腾 NPU 暂不支持显存实时监控（pynvml 不可用），需手动保证显存充足
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    logging.warning("pynvml not found, GPU monitoring disabled.")

# ── 路径初始化 ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "algorithm"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

import httpx
from RAG.datamerge.data_types import Chunk
from algorithm.ai_tools.metadata_loader import load_metadata_by_lngid
from algorithm.ai_tools.sentence_chunker import paragraph_to_sentence_chunks
from values.global_config_variable import (
    ES_URL,
    ES_TITLE_INDEX, ES_ABSTRACT_INDEX, ES_KEYWORD_INDEX, ES_FULLTEXT_INDEX,
    ES_PARAGRAPH_INDEX, ES_SENTENCE_INDEX, ES_REFERENCE_INDEX,
    embedding_base_url, Embedding_model_name, embedding_api_key,
)

# ── 默认目录与路径常量 ────────────────────────────────────────
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
SCAN_DONE = object()  # 哨兵对象，标记 PDF 目录扫描完毕

# ── Embedding 安全参数 ────────────────────────────────────────
# 8000 tokens 是 embedding 模型的上限；中英文混合按 ~2 字符/token 估算
# 0.875 安全系数给 tokenizer 的波动留余量，即实际上限 = 8000 × 2 × 0.875 = 14000 字符
MAX_FULL_TOKENS = 8000
CHARS_PER_TOKEN_ESTIMATE = 2
TOKEN_SAFETY_RATIO = 0.875
MAX_EMBEDDING_CHARS = int(
    MAX_FULL_TOKENS * CHARS_PER_TOKEN_ESTIMATE * TOKEN_SAFETY_RATIO
)

EMBEDDING_API_URL = f"{embedding_base_url.rstrip('/')}/embeddings"
EMBEDDING_MODEL = Embedding_model_name
EMBEDDING_HEADERS = {
    "Authorization": f"Bearer {embedding_api_key}",
    "Content-Type": "application/json",
}

# ── Embedding 自适应 batch_size ─────────────────────────────────
# _embedding_safe_batch 是全局学到的安全值，遇到 413 时自动下调
# 初始 64，每遇一次 413 减半，收敛到服务器稳定接受的值
EMBEDDING_BATCH_SIZE = 64
_embedding_safe_batch = EMBEDDING_BATCH_SIZE
_batch_lock = asyncio.Lock()

# ── ES _bulk 写入参数 ────────────────────────────────────────
ES_BULK_MAX_COUNT = 100        # 单次 _bulk 最多 100 条
ES_BULK_MAX_MB = 2             # 单次 _bulk 最多 2MB（NDJSON 序列化后）
ES_BULK_MAX_BYTES = ES_BULK_MAX_MB * 1024 * 1024

# ── ES 索引滚动参数（大规模数据场景，防止单索引过胖）─────────
ROLLOVER_MAX_DOCS = 5_000_000
ROLLOVER_MAX_PRIMARY_BYTES = 30 * 1024 ** 3
ROLLOVER_STATS_EVERY_BULKS = 25
ROLLOVER_STATS_EVERY_DOCS = 2_500
ROLLOVER_NEAR_THRESHOLD_RATIO = 0.90


class RolloverError(RuntimeError):
    """ES 索引滚动异常。由 rollover_coordinator 在索引超限时抛出，
    提示调用方需要切换新索引后重试。当前 rollover_coordinator 未实现，
    此异常仅作预留，实际不会触发。
    """

# ── 日志配置 ──────────────────────────────────────────────────
def setup_logging(state_dir: Path, level=logging.INFO):
    """初始化日志：同时输出到文件（pipeline.log）和 stdout。"""
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
    """从启动到现在的平均吞吐量（PDF/h），用于日志输出。"""
    if completed <= 0 or started_at <= 0:
        return 0.0
    elapsed = (now or time.monotonic()) - started_at
    return completed / elapsed * 3600 if elapsed > 0 else 0.0


def _window_rate_per_hour(timestamps: Deque[float]) -> float:
    """最近 N 个完成任务的滑动窗口速率（PDF/h），反映当前实时速度而非全程平均。"""
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / elapsed * 3600 if elapsed > 0 else 0.0




# --- Embedding and Elasticsearch import logic ---


# ═══════════════════════════════════════════════════════════════════════════
# 超长文本安全切分模块
# ─────────────────────────────────────────────────────────────────────────
# 背景：Embedding 模型有 8000 token 的上限。虽然 MarkdownStructureAdapter
# 产出 chunk 时已将段落控制在 ≤ 1500 字符，但为防御极端情况（如无标点的
# 超长文本），在 embedding 前再做一次安全切分作为兜底。
# 绝大多数情况下 fragment_count == 1（透传），极少数超长文本才触发降级拆分。
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TextFragment:
    """段落/参考文献的安全 fragment。由 build_text_fragments() 产出。

    对于正常长度的 chunk（≤ 14000 字符），一个 chunk → 一个 fragment（透传）。
    对于超长 chunk，按边界逐级切分为多个 fragment，fragment_count > 1。
    """
    text: str
    source_index: int           # 原始 chunk 在列表中的位置索引
    fragment_index: int         # 当前 fragment 在该 chunk 所有 fragment 中的序号（0 起）
    fragment_count: int         # 该 chunk 被切成了几个 fragment（1 = 未拆分）
    source_chunk_id: str        # 原始 chunk 的 id
    source_chunk_seq: int       # 原始 chunk 的 chunk_seq
    section_path: str           # 章节路径，如 "引言 > 方法"
    page_num: int               # 所在页码
    element_type: str           # 元素类型（"正文"/"表格"/"参考文献条目" 等）


@dataclass(frozen=True)
class SentenceFragment:
    """句子的安全 fragment。由 build_sentence_fragments() 产出。

    流程：chunk 文本 → paragraph_to_sentence_chunks() 拆句 → 每个句子过 safety split。
    句子通常远小于 14000 字符，fragment_count 几乎恒为 1，此处的 split 仅作兜底。
    """
    text: str
    paragraph_index: int        # 原始 chunk 在段落列表中的位置
    sentence_index: int         # 句子在该段落中的序号
    fragment_index: int         # 该句子的 fragment 序号（几乎恒为 0）
    fragment_count: int         # 该句子的 fragment 数量（几乎恒为 1）
    source_chunk_id: str
    source_chunk_seq: int
    section_path: str
    page_num: int
    element_type: str


def estimate_tokens(text: str) -> int:
    """保守估算 token 数：中文 ~2 字符/token，上取整。"""
    return (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE


def _split_preserving_delimiters(text: str, pattern: str) -> List[str]:
    """按正则切分文本，但保留分隔符在前一段末尾（零宽断言 lookbehind）。
    例如按句号切 "A。B。" → ["A。", "B。"]，标点不丢失。
    """
    return [part.strip() for part in re.split(pattern, text) if part.strip()]


def _merge_safe_parts(parts: List[str], max_chars: int) -> List[str]:
    """尝试合并相邻的小片段，使每个片段尽可能接近 max_chars 但不超出。
    避免过度切分导致大量碎片化的小文本，影响 embedding 质量。
    """
    merged: List[str] = []
    current = ""
    for part in parts:
        candidate = part if not current else current + part
        if current and len(candidate) > max_chars:
            merged.append(current.strip())
            current = part
        else:
            current = candidate
    if current.strip():
        merged.append(current.strip())
    return merged


def split_text_to_safe_fragments(
    text: str,
    max_chars: int = MAX_EMBEDDING_CHARS,
) -> List[str]:
    """将文本切分为满足 embedding token 上限的安全片段。

    切分策略（逐级降级，前一级切不动再试下一级）：
      1. 双换行（段落边界）   → r"(?<=\n\n)"
      2. 句末标点（。！？.!?） → 按句子边界
      3. 分号/逗号             → 更弱的子句边界
      4. 空格                  → 单词边界（英文场景）
      5. 硬截断                → 直接按 max_chars 切，纯兜底

    最后 _merge_safe_parts 合并过小的片段以获得更好的 embedding 质量。
    对于 ≤ max_chars 的文本直接原样返回（透传，大多数情况）。
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]  # ← 绝大多数情况走这里，透传

    # 逐级降级：从强边界往弱边界尝试
    levels = [
        r"(?<=\n\n)",              # 1. 段落边界
        r"(?<=[。！？.!?])",        # 2. 句子边界
        r"(?<=[；;：:，,、])",      # 3. 子句边界
        r"(?<=\s)",                # 4. 单词边界
    ]
    parts = [text]
    for pattern in levels:
        next_parts: List[str] = []
        for part in parts:
            if len(part) <= max_chars:
                next_parts.append(part)          # 已满足要求，不动
            else:
                split = _split_preserving_delimiters(part, pattern)
                next_parts.extend(split if len(split) > 1 else [part])
        parts = next_parts

    # 所有边界策略都无效 → 硬截断（最后手段）
    final: List[str] = []
    for part in parts:
        if len(part) <= max_chars:
            final.append(part)
        else:
            final.extend(
                part[offset:offset + max_chars].strip()
                for offset in range(0, len(part), max_chars)
                if part[offset:offset + max_chars].strip()
            )
    result = _merge_safe_parts(final, max_chars)
    if any(len(part) > max_chars for part in result):
        raise RuntimeError("Token safety splitter produced an oversized fragment")
    return result


def _chunk_value(chunk: Any, name: str, default: Any) -> Any:
    """安全获取 chunk 属性，因为部分 chunk 可能由不同 adapter 产出，字段不完全一致。"""
    return getattr(chunk, name, default)


def build_text_fragments(chunks: List[Any]) -> List[TextFragment]:
    """将段落/参考文献 chunk 列表转为带安全切分的 TextFragment 列表。
    每个 chunk → split_text_to_safe_fragments() → 1 个或多个 fragment。
    绝大多数 chunk（≤ 14000 字符）透传，fragment_count == 1。
    """
    records: List[TextFragment] = []
    for source_index, chunk in enumerate(chunks):
        text = _chunk_value(chunk, "text", "").strip()
        if not text:
            continue
        fragments = split_text_to_safe_fragments(text)  # 安全切分（通常透传）
        for fragment_index, fragment in enumerate(fragments):
            records.append(TextFragment(
                text=fragment,
                source_index=source_index,
                fragment_index=fragment_index,
                fragment_count=len(fragments),
                source_chunk_id=str(_chunk_value(chunk, "id", "")),
                source_chunk_seq=int(_chunk_value(chunk, "chunk_seq", source_index + 1)),
                section_path=str(_chunk_value(chunk, "section_path", "")),
                page_num=int(_chunk_value(chunk, "page_num", 1)),
                element_type=str(_chunk_value(chunk, "element_type", "")),
            ))
    return records


def build_sentence_fragments(chunks: List[Any]) -> List[SentenceFragment]:
    """将段落 chunk 列表先拆句、再做安全切分，转为 SentenceFragment 列表。
    流程：chunk.text → paragraph_to_sentence_chunks() 拆句 → 跳过过短句子（<2字符）
          → split_text_to_safe_fragments() 每句安全切分（句子极短，几乎恒透传）。
    """
    records: List[SentenceFragment] = []
    for paragraph_index, chunk in enumerate(chunks):
        text = _chunk_value(chunk, "text", "").strip()
        if not text:
            continue
        for sentence_index, sentence in enumerate(paragraph_to_sentence_chunks(text)):
            if not sentence or len(sentence.strip()) < 2:  # 过滤过短片段
                continue
            fragments = split_text_to_safe_fragments(sentence)  # 兜底，句子通常不触发
            for fragment_index, fragment in enumerate(fragments):
                records.append(SentenceFragment(
                    text=fragment,
                    paragraph_index=paragraph_index,
                    sentence_index=sentence_index,
                    fragment_index=fragment_index,
                    fragment_count=len(fragments),
                    source_chunk_id=str(_chunk_value(chunk, "id", "")),
                    source_chunk_seq=int(_chunk_value(chunk, "chunk_seq", paragraph_index + 1)),
                    section_path=str(_chunk_value(chunk, "section_path", "")),
                    page_num=int(_chunk_value(chunk, "page_num", 1)),
                    element_type=str(_chunk_value(chunk, "element_type", "")),
                ))
    return records


async def async_embed_texts(
    http_client: httpx.AsyncClient,
    texts: List[str],
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """
    自适应 Embedding 调用：
    - 初始用全局学习到的安全 batch_size，避免反复撞墙
    - 遇到 413 (Payload Too Large) 自动减半重试，并更新全局安全值
    - 遇到 5xx / 超时 自动减半重试
    - 最终减到 1 还失败才抛异常
    """
    global _embedding_safe_batch

    if not texts:
        return []

    if batch_size is None:
        batch_size = _embedding_safe_batch

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1

        payload = {"input": batch_texts, "model": EMBEDDING_MODEL}
        resp = await http_client.post(
            EMBEDDING_API_URL, json=payload, headers=EMBEDDING_HEADERS,
        )

        # ── 自适应降级 ──────────────────────────────────
        if resp.status_code in (413, 502, 503, 504) or resp.status_code >= 500:
            if batch_size <= 1:
                raise RuntimeError(
                    f"Embedding API 错误 batch_size=1 仍失败 "
                    f"(HTTP {resp.status_code}): {resp.text[:500]}"
                )
            smaller = max(batch_size // 2, 1)
            print(
                f"  [Embedding] 批次 {batch_num} HTTP {resp.status_code}，"
                f"batch_size {batch_size} → {smaller}"
            )
            # 动态更新全局安全值（取最小值，越学越保守）
            async with _batch_lock:
                _embedding_safe_batch = min(_embedding_safe_batch, smaller)
            # 递归：用减半后的 size 重试这批文本
            sub_embeddings = await async_embed_texts(
                http_client, batch_texts, smaller,
            )
            all_embeddings.extend(sub_embeddings)
            continue

        if resp.status_code != 200:
            # 非 413/5xx 错误（如 4xx 参数错误），尝试减半
            if batch_size > 1:
                smaller = max(batch_size // 2, 1)
                print(
                    f"  [Embedding] 批次 {batch_num} HTTP {resp.status_code}，"
                    f"尝试降级 batch_size={smaller}"
                )
                sub_embeddings = await async_embed_texts(
                    http_client, batch_texts, smaller,
                )
                all_embeddings.extend(sub_embeddings)
                continue
            raise RuntimeError(
                f"Embedding API 错误 (HTTP {resp.status_code}): {resp.text[:500]}"
            )

        batch_result = resp.json()
        batch_embeddings = [item["embedding"] for item in batch_result["data"]]
        all_embeddings.extend(batch_embeddings)

        if total_batches > 1:
            print(
                f"  [Embedding] 批次 {batch_num}/{total_batches} 完成 "
                f"({len(batch_texts)} 条)"
            )

    return all_embeddings


async def es_bulk_index(
    http_client: httpx.AsyncClient,
    actions: List[Dict[str, Any]],
    max_count: int = ES_BULK_MAX_COUNT,
    max_mb: int = ES_BULK_MAX_MB,
) -> Dict[str, Any]:
    """
    自适应 ES _bulk 写入：自动按数量和载荷大小拆分成多个子批次。

    - actions 超过 max_count 条 → 拆
    - 估算 NDJSON 大小超过 max_mb → 拆
    - 单个子批次遇到 413 / timeout → 再拆更小重试
    """
    if not actions:
        return {"took": 0, "errors": False, "items": []}

    # ── 第一阶段：按静态阈值预拆分 ──────────────────────
    chunks = _split_actions_into_chunks(actions, max_count, max_mb)

    merged_result: Dict[str, Any] = {"took": 0, "errors": False, "items": []}

    for chunk_idx, chunk_actions in enumerate(chunks):
        if len(chunks) > 1:
            print(
                f"  [ES Bulk] 子批次 {chunk_idx + 1}/{len(chunks)} "
                f"({len(chunk_actions)} 条)"
            )

        chunk_result = await _es_bulk_send(
            http_client, chunk_actions, max_mb,
        )
        merged_result["took"] += chunk_result.get("took", 0)
        if chunk_result.get("errors"):
            merged_result["errors"] = True
        merged_result["items"].extend(chunk_result.get("items", []))

    return merged_result


def _serialize_bulk_action(action: Dict[str, Any]) -> bytes:
    """将单条 ES action 序列化为 NDJSON 格式。
    ES _bulk API 要求每行一条 JSON：
      {"index": {"_index": "...", "_id": "..."}}  ← 元数据行
      {"field1": "value1", ...}                   ← 文档体行
    """
    meta = json.dumps(
        {"index": {"_index": action["_index"], "_id": action["_id"]}},
        ensure_ascii=False,
        separators=(",", ":"),  # 紧凑格式，减少字节数
    )
    source = json.dumps(
        action["_source"], ensure_ascii=False, separators=(",", ":")
    )
    return (meta + "\n" + source + "\n").encode("utf-8")


def _estimate_bulk_mb(actions: List[Dict[str, Any]]) -> float:
    """估算 actions 的 NDJSON 字节大小（MB）。用于判断是否需要拆分。"""
    return sum(len(_serialize_bulk_action(action)) for action in actions) / (1024 * 1024)


def _split_actions_into_chunks(
    actions: List[Dict[str, Any]],
    max_count: int,
    max_mb: float,
) -> List[List[Dict[str, Any]]]:
    """按数量上限和载荷大小将 ES actions 列表拆分成多个子批次。
    每个子批次同时满足：条数 ≤ max_count，NDJSON 字节数 ≤ max_mb。
    如果单条 action 就超过 max_mb，直接抛异常（异常数据，无法写入）。
    """
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_bytes = 0
    max_bytes = int(max_mb * 1024 * 1024)

    for a in actions:
        single_bytes = len(_serialize_bulk_action(a))
        if single_bytes > max_bytes:
            raise ValueError(
                f"单条 ES action 超过 {max_mb}MB: "
                f"index={a['_index']} id={a['_id']} bytes={single_bytes}"
            )
        # 当前 chunk 加上这条会超限 → 先封存当前 chunk，再开新的
        if current and (
            len(current) >= max_count
            or current_bytes + single_bytes > max_bytes
        ):
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(a)
        current_bytes += single_bytes

    if current:
        chunks.append(current)

    return chunks


async def _es_bulk_send(
    http_client: httpx.AsyncClient,
    actions: List[Dict[str, Any]],
    max_mb: float,
    _depth: int = 0,
) -> Dict[str, Any]:
    """发送单次 _bulk 请求。遇 413（载荷过大）或网络超时，自动拆成两半递归重试。
    _depth 限制最多递归 3 层，防止无限拆分。
    """
    if not actions:
        return {"took": 0, "errors": False, "items": []}

    body = b"".join(_serialize_bulk_action(action) for action in actions)

    url = f"{ES_URL.rstrip('/')}/_bulk"
    try:
        resp = await http_client.post(
            url,
            content=body,
            headers={"Content-Type": "application/x-ndjson"},
        )
    except Exception as e:
        # 网络错误（超时等）→ 拆成两半递归重试，最多 3 层
        if len(actions) > 1 and _depth < 3:
            smaller_mb = max(max_mb / 2, 0.5)
            print(
                f"  [ES Bulk] 请求失败 ({e})，"
                f"拆成两半重试 (mb={smaller_mb:.1f})"
            )
            chunks = _split_actions_into_chunks(
                actions, max(len(actions) // 2, 1), smaller_mb,
            )
            merged = {"took": 0, "errors": False, "items": []}
            for chunk in chunks:
                sub = await _es_bulk_send(
                    http_client, chunk, smaller_mb, _depth + 1,
                )
                merged["took"] += sub.get("took", 0)
                if sub.get("errors"):
                    merged["errors"] = True
                merged["items"].extend(sub.get("items", []))
            return merged
        raise  # 单条或已达最大深度，不再重试

    # HTTP 413 载荷过大 → 拆成两半递归重试
    if resp.status_code == 413 and len(actions) > 1 and _depth < 3:
        smaller_mb = max(max_mb / 2, 0.5)
        print(
            f"  [ES Bulk] HTTP 413 载荷过大，"
            f"拆成两半重试 (mb={smaller_mb:.1f})"
        )
        chunks = _split_actions_into_chunks(
            actions, max(len(actions) // 2, 1), smaller_mb,
        )
        merged = {"took": 0, "errors": False, "items": []}
        for chunk in chunks:
            sub = await _es_bulk_send(
                http_client, chunk, smaller_mb, _depth + 1,
            )
            merged["took"] += sub.get("took", 0)
            if sub.get("errors"):
                merged["errors"] = True
            merged["items"].extend(sub.get("items", []))
        return merged

    # 非 200 非 413 → 记录错误返回，不重试
    if resp.status_code != 200:
        print(
            f"  [ES Bulk Error] HTTP {resp.status_code}: {resp.text[:500]}"
        )
        return {"took": 0, "errors": True, "items": []}

    result = resp.json()
    # 200 但部分文档写入失败 → 打印前 5 条错误详情
    if result.get("errors"):
        failed_items = [
            item for item in result.get("items", [])
            if "error" in item.get("index", {})
        ]
        for fi in failed_items[:5]:
            idx_info = fi.get("index", {})
            print(
                f"  [ES Bulk Error] _id={idx_info.get('_id')}: "
                f"{idx_info.get('error')}"
            )
        if len(failed_items) > 5:
            print(
                f"  [ES Bulk Error] ... 还有 "
                f"{len(failed_items) - 5} 条失败"
            )

    return result


async def import_one_pdf_to_es_optimized(
    http_client: httpx.AsyncClient,
    lngid: str,
    raw_chunks: List[Any],
    meta: Dict[str, Any],
    rollover_coordinator: Optional[Any] = None,
) -> Dict[str, int]:
    """处理单篇 PDF 的完整入库流程：向量化 + 写入 7 个 ES 索引。

    流程：
      1. 拆分 chunk：按 element_type / section_path 分为【段落】和【参考文献】两组
      2. 组装全文：拼接所有 chunk 文本 → 检查是否超过 MAX_EMBEDDING_CHARS → 超限则丢弃
      3. 安全切分：段落 → build_text_fragments(), 句子 → build_sentence_fragments()
      4. 元数据检查：title/abstract/keyword 超限则跳过该索引
      5. 批量 embedding：meta（标题+摘要+关键词+全文）/ paragraph / sentence / reference 四组
      6. 构造 ES actions：7 个索引分别组装 NDJSON
      7. _bulk 写入：自适应拆分 + 重试

    返回每个索引实际写入的文档数。
    """
    counts = {key: 0 for key in COUNT_KEYS}

    # ── 提取元数据字段 ─────────────────────────────────────────
    title_c = meta.get("title_c", "")
    title_e = meta.get("title_e", "")
    keyword_c = meta.get("keyword_c", "")
    keyword_e = meta.get("keyword_e", "")
    remark_c = meta.get("remark_c", "")
    remark_e = meta.get("remark_e", "")

    # 所有 ES 文档共用的基础字段
    common = {
        "file_id": lngid,
        "doi": meta.get("doi", ""),
        "firstwriter": meta.get("firstwriter", ""),
        "showwriter": meta.get("showwriter", ""),
        "delete_flag": 0,
    }

    # ── 第 1 步：分离段落 chunk 和参考文献 chunk ──────────────
    paragraph_chunks: List[Any] = []
    reference_chunks: List[Any] = []
    for chunk in raw_chunks:
        element_type = _chunk_value(chunk, "element_type", "")
        section_path = _chunk_value(chunk, "section_path", "")
        if element_type in ("参考文献", "参考文献条目") or "参考文献" in section_path:
            reference_chunks.append(chunk)
        else:
            paragraph_chunks.append(chunk)

    # ── 第 2 步：组装全文文本 ──────────────────────────────────
    full_text = "\n".join(
        _chunk_value(chunk, "text", "") for chunk in raw_chunks
        if _chunk_value(chunk, "text", "")
    )
    full_text = re.sub(r"\s+", " ", full_text).strip()
    estimated_tokens = estimate_tokens(full_text)
    # 全文超出 embedding 安全限制 → 舍弃全文索引（不像段落那样拆分，因为全文应该是一个整体向量）
    if len(full_text) > MAX_EMBEDDING_CHARS:
        logger.warning(
            "[%s] 全文超过 Embedding 安全限制，舍弃全文索引: chars=%d estimated_tokens=%d",
            lngid, len(full_text), estimated_tokens,
        )
        full_text = ""

    # ── 第 3 步：安全切分（兜底，绝大多数情况透传）────────────
    paragraph_records = build_text_fragments(paragraph_chunks)
    sentence_records = build_sentence_fragments(paragraph_chunks)
    reference_records = build_text_fragments(reference_chunks)

    # ── 第 4 步：检查元数据是否超限 ──────────────────────────
    optional_meta = {
        "title": (title_c + " " + title_e).strip(),
        "abstract": (remark_c + " " + remark_e).strip(),
        "keyword": (keyword_c + " " + keyword_e).strip(),
    }
    for label, text in list(optional_meta.items()):
        if len(text) > MAX_EMBEDDING_CHARS:
            logger.warning("[%s] %s 超过 Embedding 安全限制，跳过", lngid, label)
            optional_meta[label] = ""
    title_text = optional_meta["title"]
    abstract_text = optional_meta["abstract"]
    keyword_text = optional_meta["keyword"]

    # ── 第 5 步：分批向量化 ──────────────────────────────────
    # embed_group 是内部 helper：失败时不抛异常，返回空向量占位
    async def embed_group(label: str, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return await async_embed_texts(http_client, texts)
        except Exception as exc:
            print(f"  [WARN] {lngid} Embedding({label}) 失败: {exc}")
            return [[] for _ in texts]  # 返回空向量占位，不影响其他索引

    # 元数据组：title + abstract + keyword + fulltext 合并一次请求
    meta_labels: List[str] = []
    meta_texts: List[str] = []
    for label, text in (
        ("title", title_text), ("abstract", abstract_text),
        ("keyword", keyword_text), ("fulltext", full_text),
    ):
        if text:
            meta_labels.append(label)
            meta_texts.append(text)

    # 四组向量化请求（可考虑并发，但当前逐个调用避免 API 侧排队）
    meta_vecs = await embed_group("meta", meta_texts)
    para_vecs = await embed_group("paragraph", [item.text for item in paragraph_records])
    sent_vecs = await embed_group("sentence", [item.text for item in sentence_records])
    ref_vecs = await embed_group("reference", [item.text for item in reference_records])

    # 将 meta 向量映射回各自的 label
    meta_vectors = {
        label: meta_vecs[index] if index < len(meta_vecs) else []
        for index, label in enumerate(meta_labels)
    }

    # ── 第 6 步：构造 7 个索引的 ES actions ──────────────────
    actions: List[Dict[str, Any]] = []

    # 索引 1：标题库
    if title_text and meta_vectors.get("title"):
        actions.append({"_index": ES_TITLE_INDEX, "_id": f"{lngid}_title", "_source": {
            **common, "title_c": title_c, "title_e": title_e,
            "title_text": title_text, "title_embedding": meta_vectors["title"],
        }})
        counts["title"] += 1

    # 索引 2：摘要库（用 remark 字段）
    if abstract_text and meta_vectors.get("abstract"):
        actions.append({"_index": ES_ABSTRACT_INDEX, "_id": f"{lngid}_abstract", "_source": {
            **common, "remark_c": remark_c, "remark_e": remark_e,
            "abstract_embedding": meta_vectors["abstract"],
        }})
        counts["abstract"] += 1

    # 索引 3：关键词库
    if keyword_text and meta_vectors.get("keyword"):
        actions.append({"_index": ES_KEYWORD_INDEX, "_id": f"{lngid}_keyword", "_source": {
            **common, "keyword_c": keyword_c, "keyword_e": keyword_e,
            "keyword_text": keyword_text, "keyword_embedding": meta_vectors["keyword"],
        }})
        counts["keyword"] += 1

    # 索引 4：全文库（可能因超限被舍弃）
    if full_text and meta_vectors.get("fulltext"):
        actions.append({"_index": ES_FULLTEXT_INDEX, "_id": f"{lngid}_fulltext", "_source": {
            **common, "full_text": full_text,
            "full_embedding": meta_vectors["fulltext"], "token_count": estimated_tokens,
        }})
        counts["fulltext"] += 1

    # 索引 5：段落库（每个 fragment 一条文档）
    for item, vector in zip(paragraph_records, para_vecs):
        if not vector:
            continue
        actions.append({"_index": ES_PARAGRAPH_INDEX,
                        "_id": f"{lngid}_p{item.source_index}_f{item.fragment_index}",
                        "_source": {
            **common, "para_index": item.source_index, "para_text": item.text,
            "para_embedding": vector, "section_path": item.section_path,
            "element_type": item.element_type, "page_num": item.page_num,
            "source_chunk_id": item.source_chunk_id,
            "source_chunk_seq": item.source_chunk_seq,
            "fragment_seq": item.fragment_index,
            "fragment_count": item.fragment_count,
        }})
        counts["paragraph"] += 1

    # 索引 6：句子库（每个 sentence fragment 一条文档）
    for item, vector in zip(sentence_records, sent_vecs):
        if not vector:
            continue
        actions.append({"_index": ES_SENTENCE_INDEX,
                        "_id": f"{lngid}_p{item.paragraph_index}s{item.sentence_index}_f{item.fragment_index}",
                        "_source": {
            **common, "para_index": item.paragraph_index,
            "sent_seq": item.sentence_index, "sent_text": item.text,
            "sent_embedding": vector, "section_path": item.section_path,
            "element_type": item.element_type, "page_num": item.page_num,
            "source_chunk_id": item.source_chunk_id,
            "source_chunk_seq": item.source_chunk_seq,
            "fragment_seq": item.fragment_index,
            "fragment_count": item.fragment_count,
        }})
        counts["sentence"] += 1

    # 索引 7：参考文献库
    for item, vector in zip(reference_records, ref_vecs):
        if not vector:
            continue
        actions.append({"_index": ES_REFERENCE_INDEX,
                        "_id": f"{lngid}_r{item.source_index}_f{item.fragment_index}",
                        "_source": {
            **common, "ref_seq": item.source_index, "ref_text": item.text,
            "ref_embedding": vector, "section_path": item.section_path,
            "element_type": item.element_type, "page_num": item.page_num,
            "source_chunk_id": item.source_chunk_id,
            "source_chunk_seq": item.source_chunk_seq,
            "fragment_seq": item.fragment_index,
            "fragment_count": item.fragment_count,
        }})
        counts["reference"] += 1

    # ── 第 7 步：批量写入 ES ──────────────────────────────────
    if actions:
        try:
            if rollover_coordinator is None:
                await es_bulk_index(http_client, actions)
            else:
                await rollover_coordinator.bulk_index(http_client, actions)
        except RolloverError:
            raise
        except Exception as exc:
            print(f"  [WARN] {lngid} ES _bulk 写入失败: {exc}")

    return counts


# ═══════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParseTask:
    """一个待解析的 PDF 任务，由 feed_tasks 产出，经 pending_queue 到达 Supervisor。

    字段说明：
      key:      PDF 相对路径的 SHA256 前 20 位，用作任务的唯一标识
      sequence: 全局递增编号，用于排序和日志
      pdf_path: PDF 文件绝对路径
      lngid:    从文件名提取的文档 ID，用于查 SQLite 元数据
      attempt:  当前是第几次重试（从 1 开始），超过 max_parse_attempts 则放弃
    """
    key: str
    sequence: int
    pdf_path: str
    lngid: str
    attempt: int = 1

@dataclass
class WorkerSlot:
    """一个 GPU Worker 槽位，代表一个 MinerU 子进程。

    字段说明：
      slot_id:       槽位编号（全局唯一），从 0 开始
      gpu_id:        绑定的 GPU 编号
      generation:    代数，每次重启 +1，用于过滤旧进程发来的 stale 事件
      process:       子进程对象
      command_queue: 父→子 命令队列（任务通过此队列发给子进程）
      ready:         子进程是否就绪（已完成 MinerU 模型加载）
      lease:         当前正在执行的任务（None 表示空闲）
      leased_at:     租约开始时间，用于超时检测
      disabled:      是否已被熔断（频繁崩溃则禁用此槽位）
      crash_count:   60 秒内的崩溃次数，≥3 触发熔断
      last_crash_time: 最近一次崩溃的时间戳
    """
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
    """全局流水线状态，跨协程共享，用于统计和日志输出。

    字段说明：
      discovered:      文件扫描阶段发现的有效 PDF 总数
      submitted:       已提交给子进程的任务数
      parse_attempts:  解析尝试总次数（含重试）
      parsed:          MinerU 解析成功的 PDF 数
      success:         最终入库成功的 PDF 数
      parse_failures:  解析永久失败（重试耗尽）的 PDF 数
      index_failures:  入库失败的 PDF 数
      timeouts:        解析超时次数
      worker_deaths:   Worker 异常退出次数
      worker_restarts: Worker 重启次数
      counts:          各 ES 索引累计写入文档数
      per_gpu:         每张 GPU 完成的解析数
      started_at:      流水线启动时间戳
      parse_started_at: 第一份解析请求发出时间
      index_started_at: 第一条入库完成时间
      parse_recent:    最近 100 个解析完成的时间戳（用于滑动窗口速率）
      index_recent:    最近 100 个入库完成的时间戳
      lock:            保护 state 字段的异步锁
    """
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

# ═══════════════════════════════════════════════════════════════════════════
# 断点续传模块
# ─────────────────────────────────────────────────────────────────────────
# StatusTracker 负责记录每个 lngid 的处理状态到 pipeline_status.jsonl，
# 重启后自动跳过已成功的文件，失败的根据 attempt 次数决定是否重试。
# 使用 JSONL 格式（每行一条 JSON），append 写入 + fsync 保证崩溃时数据不丢。
# ═══════════════════════════════════════════════════════════════════════════

class StatusTracker:
    """处理状态追踪器，支持断点续传和失败记录。

    pipeline_status.jsonl: 记录成功/失败（精简，每行一条）
    error_details.jsonl:   记录失败时的完整 traceback（供排查）
    _cache:                内存中按 lngid 索引的字典，加速查询
    """
    def __init__(self, state_dir: Path):
        self.status_file = state_dir / "pipeline_status.jsonl"
        self.error_file = state_dir / "error_details.jsonl"
        self.lock = asyncio.Lock()
        self._cache: Dict[str, Dict] = {}

    async def load(self):
        """从磁盘加载状态文件到内存。使用线程池避免阻塞事件循环。
        如果状态文件不存在（首次运行），直接跳过。
        """
        if not self.status_file.exists():
            return
        logger.info(f"Loading status from {self.status_file}...")
        try:
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
        """查询某个 lngid 的处理状态。"""
        return self._cache.get(lngid)

    async def record_success(self, lngid: str):
        """记录入库成功（只在 ES _bulk 全部完成后调用）。"""
        async with self.lock:
            record = {
                "lngid": lngid,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            self._cache[lngid] = record
            await self._append_log(record)

    async def record_failure(self, lngid: str, stage: str, error: str, attempt: int, pdf_path: str = ""):
        """记录失败。stage = "parse" 或 "index"，代表失败发生的阶段。
        同时写入主状态文件和详细错误文件。
        """
        async with self.lock:
            record = {
                "lngid": lngid,
                "status": "failure",
                "stage": stage,
                "attempt": attempt,
                "error": error,
                "pdf_path": pdf_path,
                "timestamp": datetime.now().isoformat()
            }
            self._cache[lngid] = record
            await self._append_log(record)
            # 详细错误单独存一个文件，方便排查
            error_detail = {**record, "traceback": error}
            await self._append_log(error_detail, self.error_file)

    async def _append_log(self, record: Dict, target_file: Optional[Path] = None):
        """追加一条记录到 JSONL 文件，fsync 保证落盘。"""
        target = target_file or self.status_file
        def _write():
            with target.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        await asyncio.to_thread(_write)

    def is_done(self, lngid: str) -> bool:
        """判断是否已成功处理。"""
        s = self._cache.get(lngid)
        return s is not None and s.get("status") == "success"

    def should_skip(self, lngid: str, max_attempts: int) -> bool:
        """判断是否应该跳过该文件：
        - 已成功 → 跳过
        - 失败且 attempt 已达上限 → 跳过（永久放弃）
        - 失败但 attempt 未达上限 → 重试
        - 新文件 → 处理
        """
        s = self._cache.get(lngid)
        if not s: return False  # 新任务，不跳过
        if s.get("status") == "success": return True  # 已成功，跳过
        if s.get("status") == "failure" and s.get("attempt", 0) >= max_attempts:
            return True  # 重试次数耗尽，跳过
        return False

# ═══════════════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════════════

def setup_worker_env(gpu_id: int, vram_size: int, device_type: str = "cuda") -> None:
    """配置 Worker 子进程的设备环境和 MinerU 模型路径。

    根据 device_type 自动选择对应的设备隔离方式：
      cuda: CUDA_VISIBLE_DEVICES + MINERU_DEVICE_MODE=cuda:0
      npu:  ASCEND_RT_VISIBLE_DEVICES + MINERU_DEVICE_MODE=npu:0
      cpu:  不做设备隔离

    MinerU 3.4 注意：
      首次使用前需先在命令行执行 mineru-models-download -s modelscope -m pipeline
      下载完成后 ~/.mineru.json 会自动记录模型路径，子进程读取该文件定位模型。
    """
    # ── 设备隔离 ──────────────────────────────────────────
    if device_type == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MINERU_DEVICE_MODE"] = "cuda:0"
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    elif device_type == "npu":
        # 华为昇腾 NPU：使用 ASCEND_RT_VISIBLE_DEVICES 隔离，MinerU 内部通过 get_device() 识别 npu
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["MINERU_DEVICE_MODE"] = "npu:0"
    elif device_type == "cpu":
        os.environ["MINERU_DEVICE_MODE"] = "cpu"
    else:
        raise ValueError(f"Unsupported device_type: {device_type}")

    os.environ["MINERU_VIRTUAL_VRAM_SIZE"] = str(vram_size)

    # MinerU 3.4 模型配置：优先本地已下载的模型，避免子进程网络不通导致下载失败
    os.environ.setdefault("MINERU_MODEL_SOURCE", "modelscope")

def extract_lngid(filename: str) -> str:
    """从 PDF 文件名提取 lngid（文档唯一标识）：
    - 去掉扩展名 (.pdf/.docx/.txt/.md)
    - 去掉临时前缀（vec_xxx_ / temp_xxx_）
    例如：temp_a1b2c3_1000004517957.pdf → 1000004517957
    """
    import re
    base = os.path.basename(str(filename))
    base = re.sub(r"\.(pdf|docx?|txt|md)$", "", base, flags=re.IGNORECASE)
    base = re.sub(r"^vec_[a-f0-9]+_", "", base)
    base = re.sub(r"^temp_[a-f0-9_-]+_", "", base)
    return base.strip()

def task_key(pdf_path: Path, root: Path) -> str:
    """生成任务的唯一 key：PDF 相对路径的 SHA256 前 20 位。
    即使同一文件多次出现也能通过 key 去重或追踪。
    """
    try:
        relative = pdf_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(pdf_path.resolve())
    return hashlib.sha256(relative.encode()).hexdigest()[:20]

def get_gpu_usage(gpu_id: int) -> Optional[float]:
    """获取 GPU 显存使用率（0.0 ~ 1.0）。
    用于任务调度：显存超过 95% 则跳过该 slot，等待 GC 释放。
    """
    if not PYNVML_AVAILABLE:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / info.total
    except Exception:
        return None

def atomic_pickle_dump(value: Any, destination: Path) -> None:
    """原子化 pickle 写入：先写 .tmp 临时文件，fsync 后 rename。
    如果写入过程中崩溃，.tmp 文件残留但不会影响下次启动。
    """
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)

# ═══════════════════════════════════════════════════════════════════════════
# MinerU 解析 Worker（子进程中运行）
# ─────────────────────────────────────────────────────────────────────────
# 设计要点：
#   1. 每个 Worker 绑定一张 GPU，通过 CUDA_VISIBLE_DEVICES 隔离
#   2. MinerU import 在函数内部完成，避免主进程加载 CUDA 库
#   3. 解析结果通过 pickle 写入 spool 目录，解耦解析和入库
#   4. 子进程通过 command_queue 接收任务，result_queue 发回结果
# ═══════════════════════════════════════════════════════════════════════════

def analyze_pdf_locally(
    pdf_path: str,
    filename: str,
    work_dir: Path,
    lang: str,
    formula_enable: bool,
    table_enable: bool,
) -> List[Any]:
    """本地 MinerU 解析管线（在子进程中调用）。

    MinerU 3.4.4 版流程（与 2.7 版不同）：
      1. read_fn():            读取 PDF 字节流
      2. doc_analyze_streaming(): 版面分析 + OCR + 公式/表格识别 + 中间JSON生成
         （3.4.4 内部已包含 result_to_middle_json 的逻辑，通过 on_doc_ready 回调返回结果）
      3. union_make():         中间 JSON → Markdown 文本（在回调中调用）
      4. MarkdownStructureAdapter.build_chunks(): Markdown → Chunk 列表

    MinerU import 全部在函数内部完成（延迟导入），避免主进程触发 CUDA 初始化。

    2.7 → 3.4.4 主要变更：
      - doc_analyze() → doc_analyze_streaming()（函数名变化）
      - 返回值(5元组) → 通过 on_doc_ready 回调异步传结果
      - result_to_middle_json() 不再需要手动调用（streaming 内部处理）
      - image_writer_list 作为新参数传入（每篇 PDF 一个 writer）
    """
    from mineru.cli.common import prepare_env, read_fn
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import MakeMode
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make
    from algorithm.knowledge_base.file_parser.markdown_structure_adapter import MarkdownStructureAdapter

    # 3.4.4 不再需要 result_to_middle_json — streaming 内部已处理

    pdf_bytes = read_fn(pdf_path)
    file_stem = Path(filename).stem
    image_dir, markdown_dir = prepare_env(str(work_dir), file_stem, "auto")
    image_writer = FileBasedDataWriter(image_dir)
    markdown_writer = FileBasedDataWriter(markdown_dir)

    # ── 用闭包捕获 on_doc_ready 回调产出的 markdown ─────────
    # 3.4.4 的 doc_analyze_streaming 无返回值，通过回调异步返回每篇文档的处理结果
    # 回调签名（来自源码 _finalize_processing_window_context）：
    #   on_doc_ready(doc_index, model_list, middle_json, ocr_enable)
    result_markdown = [None]  # 列表包装，在闭包内可修改

    def on_doc_ready(doc_index: int, model_list: list, middle_json: dict, ocr_enable: bool) -> None:
        """当单篇文档所有页面处理完成后被 doc_analyze_streaming 回调。
        参数由 _finalize_processing_window_context 传入：
          doc_index:   文档在 pdf_bytes_list 中的索引
          model_list:  MinerU 推理过程中使用的模型列表
          middle_json: 已完成的中间 JSON（包含 pdf_info）
          ocr_enable:  是否启用了 OCR
        """
        pdf_info = middle_json["pdf_info"]
        image_name = os.path.basename(image_dir)
        markdown_text = union_make(pdf_info, MakeMode.MM_MD, image_name)
        result_markdown[0] = markdown_text

    # ── MinerU 3.4.4 核心管线：版面分析 → 中间 JSON → 回调生成 Markdown ──
    # 旧版(2.7): doc_analyze() → result_to_middle_json() → union_make()
    # 新版(3.4): doc_analyze_streaming() → on_doc_ready(doc_index, model_list, middle_json, ocr_enable) → union_make()
    doc_analyze_streaming(
        pdf_bytes_list=[pdf_bytes],
        image_writer_list=[image_writer],
        lang_list=[lang],
        on_doc_ready=on_doc_ready,
        parse_method="auto",
        formula_enable=formula_enable,
        table_enable=table_enable,
    )

    markdown = result_markdown[0]
    if not markdown or not markdown.strip():
        raise RuntimeError("MinerU generated empty Markdown")

    markdown_writer.write_string(f"{file_stem}.md", markdown)
    # Markdown → Chunk：与服务端版本使用相同的 MarkdownStructureAdapter
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
    device_type: str = "cuda",
) -> None:
    """MinerU 解析子进程主循环。

    生命周期：
      1. 设置 GPU 环境变量 → 发送 worker_ready
      2. 循环从 command_queue 取任务 → 调 analyze_pdf_locally() → 结果 pickle 写入 spool
      3. 收到 None 命令或异常退出 → 发送 worker_stopped / worker_crashed

    通信协议（result_queue 中传递的事件类型）：
      worker_ready:   模型加载完毕，可以接任务
      started:        开始解析某个 PDF
      parsed:         解析成功，spool_path 指向 pickle 文件
      parse_failed:   解析失败，附带 traceback
      worker_crashed: 子进程崩溃
      worker_stopped: 正常退出
    """
    setup_worker_env(gpu_id, vram_size, device_type)
    result_queue.put({
        "kind": "worker_ready", "slot_id": slot_id,
        "generation": generation, "gpu_id": gpu_id, "pid": os.getpid(),
    })

    try:
        while True:
            try:
                command = command_queue.get(timeout=1.0)  # 1 秒超时，允许检查退出信号
            except queue.Empty:
                continue

            if command is None:  # 主进程发送的优雅退出信号
                break

            task_data = command["task"]
            token = command["token"]  # token 用于关联 command 和 result
            task = ParseTask(**task_data)

            result_queue.put({
                "kind": "started", "slot_id": slot_id, "generation": generation,
                "token": token, "key": task.key, "lngid": task.lngid,
            })

            # 为每个任务创建独立的工作目录
            attempt_dir = Path(work_root) / token
            attempt_dir.mkdir(parents=True, exist_ok=True)
            spool_path = Path(spool_dir) / f"{token}.pickle"

            try:
                chunks = analyze_pdf_locally(
                    task.pdf_path, Path(task.pdf_path).name, attempt_dir, lang,
                    formula_enable, table_enable,
                )
                atomic_pickle_dump(chunks, spool_path)  # 原子写入 spool

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
                # 无论成功或失败，清理临时工作目录（spool 文件保留给 index_worker 消费）
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

# ═══════════════════════════════════════════════════════════════════════════
# Supervisor — Worker 生命周期管理与任务调度
# ─────────────────────────────────────────────────────────────────────────
# LocalSupervisor 负责：
#   1. 启动/停止 Worker 子进程
#   2. 任务分发：从 pending_queue 获取任务 → 分配给空闲 slot
#   3. 健康监控：超时检测、崩溃计数、自动重启/熔断
#   4. 失败处理：记录失败原因、决定是否重试
# ═══════════════════════════════════════════════════════════════════════════

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
        # 每个 GPU 卡槽对应一个 WorkerSlot
        self.slots = [WorkerSlot(index, gpu) for index, gpu in enumerate(gpu_slots)]
        self.retry: Deque[ParseTask] = deque()           # 重试队列（优先于 buffered）
        self.active_tokens: Dict[int, str] = {}           # slot_id → token 映射，用于校验事件有效性

    def spawn(self, slot: WorkerSlot, restart: bool = False) -> None:
        """启动一个 Worker 子进程。restart=True 时记录重启计数。
        command_queue 容量设为 2，控制反压，避免子进程积压过多任务。
        """
        slot.generation += 1
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
                self.args.device,
            ),
        )
        process.start()
        slot.process = process
        if restart:
            self.state.worker_restarts += 1
            logger.info(f"Restarted slot {slot.slot_id} (GPU {slot.gpu_id}) Gen {slot.generation}")

    async def stop_process(self, slot: WorkerSlot) -> None:
        """优雅停止子进程：先 terminate，等待 grace 秒，不响应则 kill。
        清理 command_queue 和 process 引用。
        """
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
        """生成唯一的任务 token，格式：{task_key}.a{attempt}.s{slot_id}.g{generation}
        用于关联 command 和 result 消息，防止 stale 事件被误处理。
        """
        return f"{task.key}.a{task.attempt}.s{slot.slot_id}.g{slot.generation}"

    async def fail_lease(self, slot: WorkerSlot, reason: str, timeout: bool = False) -> None:
        """处理失败的 lease：清理 spool 文件和临时目录，记录失败，决定是否重试。
        - attempt < max_parse_attempts → 放入 retry 队列，attempt + 1
        - attempt >= max_parse_attempts → 永久放弃
        """
        task = slot.lease
        token = self.active_tokens.pop(slot.slot_id, None)

        # 清理该 token 关联的所有文件
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

        await self.status_tracker.record_failure(
            task.lngid, "parse", reason, task.attempt, task.pdf_path
        )

        if timeout:
            self.state.timeouts += 1

        # 重试逻辑
        if task.attempt < self.args.max_parse_attempts:
            self.retry.append(ParseTask(
                task.key, task.sequence, task.pdf_path, task.lngid, task.attempt + 1,
            ))
        else:
            self.state.parse_failures += 1
            logger.error(f"Permanent Parse Failure: {task.lngid} (Attempt {task.attempt})")

    async def restart(self, slot: WorkerSlot, reason: str, timeout: bool = False) -> None:
        """重启 Worker：先处理当前 lease 的失败，停止进程，检查崩溃频率决定是否熔断。
        熔断条件：60 秒内 crash ≥ 3 次 → slot.disabled = True（不再重启）。
        """
        await self.fail_lease(slot, reason, timeout)
        await self.stop_process(slot)

        now = time.monotonic()
        if now - slot.last_crash_time < 60:
            slot.crash_count += 1
            if slot.crash_count > 3:
                logger.error(f"Disabling slot {slot.slot_id} due to frequent crashes.")
                slot.disabled = True
                return
        else:
            slot.crash_count = 0  # 距上次崩溃超过 60 秒，重置计数

        slot.last_crash_time = now
        self.spawn(slot, restart=True)

    async def shutdown(self) -> None:
        """优雅关闭所有 Worker：先发 None 命令通知退出，再逐个停止进程。"""
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
    """解析 Supervisor 主循环（阶段二的核心调度器）。

    职责：
      1. 事件循环：从 result_queue 获取 Worker 事件，更新 slot 状态
      2. 健康监控：检测子进程退出、任务超时，触发重启/熔断
      3. 任务分发：从 pending_queue 取任务 → buffered 缓冲 → 分配给空闲 slot
      4. 完成判断：扫描完毕 + 所有任务完成 → 退出

    调度优先级：retry（重试）> buffered（新任务）
    slot 状态机：ready → leased（任务分发）→ ready（parsed/failed）→ ...
    """
    # 启动所有 Worker 子进程
    for slot in supervisor.slots:
        supervisor.spawn(slot)

    scan_done = False
    buffered: Deque[ParseTask] = deque()  # 本地任务缓冲池

    while True:
        # ── 1. 处理 Worker 事件 ──────────────────────────────
        # 使用 asyncio.to_thread 避免阻塞事件循环
        try:
            event = await asyncio.to_thread(supervisor.result_queue.get, True, 0.1)
        except queue.Empty:
            event = None

        if event:
            slot_idx = event["slot_id"]
            if slot_idx >= len(supervisor.slots):
                continue  # 旧 slot 的 stale 事件

            slot = supervisor.slots[slot_idx]

            # generation 校验：只处理当前代的进程发出的事件
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
                # 子进程模型加载完成，可以接收任务
                slot.ready = True
            elif kind == "started" and token == current_token:
                # 子进程已开始处理
                slot.leased_at = time.monotonic()
            elif kind == "parsed" and token == current_token:
                # 解析成功 → 释放 slot，结果推入 parsed_queue 等待入库
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
                # 解析失败 → 处理 lease 失败逻辑（可能重试）
                await supervisor.fail_lease(slot, event.get("error", "parse failure"))
                slot.ready = True
            elif kind in ("worker_crashed", "worker_stopped"):
                if slot.process and slot.process.exitcode not in (None, 0):
                    state.worker_deaths += 1

        # ── 2. Worker 健康监控 ─────────────────────────────────
        now = time.monotonic()
        for slot in supervisor.slots:
            process = slot.process
            # 进程不存在且未禁用 → 需重启（还有剩余任务）
            if not process and not slot.disabled and (not scan_done or buffered or supervisor.retry or not pending_queue.empty()):
                supervisor.spawn(slot, restart=True)
            # 进程存在但已死亡 → 重启
            elif process and not process.is_alive() and not slot.disabled:
                if slot.lease is not None:
                    state.worker_deaths += 1
                    await supervisor.restart(slot, f"worker exited code={process.exitcode}")
                else:
                    await supervisor.stop_process(slot)
                    supervisor.spawn(slot, restart=True)
            # 任务超时检查（lease 超过 parse_timeout 秒未完成）
            elif slot.lease and slot.leased_at and now - slot.leased_at > supervisor.args.parse_timeout:
                await supervisor.restart(slot, "parse timeout", timeout=True)

        # ── 3. 任务分发 ─────────────────────────────────────
        # 3a. 从 pending_queue 补充 buffered 缓冲池
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

        # 3b. 将任务分配给空闲 slot（retry 优先）
        for slot in supervisor.slots:
            if not slot.ready or slot.lease is not None or slot.disabled:
                continue

            # GPU 显存检查：超过 95% 则跳过，等下一轮
            if PYNVML_AVAILABLE:
                usage = get_gpu_usage(slot.gpu_id)
                if usage and usage > 0.95:
                    continue

            # retry 队列优先于 buffered
            task = supervisor.retry.popleft() if supervisor.retry else (buffered.popleft() if buffered else None)
            if task is None:
                continue

            token = supervisor.token_for(slot, task)
            # 先登记 lease 再发送命令，防止子进程瞬间完成导致 started/parsed 事件收不到
            slot.lease = task
            slot.leased_at = now
            slot.ready = False
            supervisor.active_tokens[slot.slot_id] = token
            try:
                slot.command_queue.put_nowait({"task": task.__dict__, "token": token})
                state.submitted += 1
                state.parse_attempts += 1
            except queue.Full:
                # 队列满 → 回退 lease，任务放回 retry 队列头
                supervisor.active_tokens.pop(slot.slot_id, None)
                slot.lease = None
                slot.leased_at = 0.0
                slot.ready = True
                supervisor.retry.appendleft(task)

        # ── 4. 完成判断 ─────────────────────────────────────
        if (
            scan_done and not buffered and not supervisor.retry
            and all(slot.lease is None for slot in supervisor.slots)
        ):
            break  # 所有任务完成

        # 没有任何活着的 Worker 且扫描未完成 → 死局，抛异常
        if not any(s.process and s.process.is_alive() for s in supervisor.slots) and not scan_done:
            raise RuntimeError("All workers died and scan not done")

        await asyncio.sleep(0.05)  # 避免忙等，每轮 50ms

    await supervisor.shutdown()

# ═══════════════════════════════════════════════════════════════════════════
# 入库 Worker（阶段三）
# ─────────────────────────────────────────────────────────────────────────
# 多个 asyncio 协程并发消费 parsed_queue，每个协程：
#   1. 从 spool 反序列化 chunk 列表
#   2. 从 SQLite 加载元数据
#   3. 调用 import_one_pdf_to_es_optimized() → 向量化 + 写入 7 个 ES 索引
#   4. 记录成功/失败到 checkpoint，删除 spool 文件
# ═══════════════════════════════════════════════════════════════════════════

async def index_worker(
    worker_id: int,
    parsed_queue: asyncio.Queue,
    http_client: Any,
    db3_path: str,
    io_executor: ThreadPoolExecutor,
    status_tracker: StatusTracker,
    state: PipelineState,
    args: argparse.Namespace,
) -> None:
    """入库协程（多个实例并发运行）。

    从 parsed_queue 消费解析完成的事件，执行向量化和 ES 写入。
    parsed_queue 收到 None 时退出（由主协程发出的停止信号）。
    """
    while True:
        event = await parsed_queue.get()
        if event is None:  # 停止信号
            parsed_queue.task_done()
            return

        spool_path = Path(event["spool_path"])
        task = ParseTask(**event["task"])

        # 二次确认：防止竞态条件（另一个 worker 可能已处理过）
        if status_tracker.is_done(task.lngid):
            spool_path.unlink(missing_ok=True)
            parsed_queue.task_done()
            continue

        try:
            # 从 spool 反序列化 chunk 列表（IO 密集 → 线程池）
            chunks = await asyncio.to_thread(load_pickle, spool_path)

            # 从 SQLite 加载元数据（IO 密集 → 线程池）
            meta = await asyncio.to_thread(
                load_metadata_by_lngid, task.lngid, db3_path=str(db3_path),
            )

            # 核心：向量化 + 写入 7 个 ES 索引
            counts = await import_one_pdf_to_es_optimized(
                http_client, task.lngid, chunks, meta,
            )

            # 只在完全成功后记录（确保断点续传不会漏掉未入库的数据）
            await status_tracker.record_success(task.lngid)
            now_indexed = time.monotonic()
            async with state.lock:
                state.success += 1
                state.index_recent.append(now_indexed)
                for key, value in counts.items():
                    state.counts[key] = state.counts.get(key, 0) + value
                completed = state.success
                count_snapshot = dict(state.counts)

            # ── 终端打印 per-file 结果 ──────
            total = sum(counts.values())
            counts_str = ", ".join(f"{k}={v}" for k, v in counts.items() if v > 0)
            print(f"  [OK] [{completed:04d}] {task.lngid}.pdf: {total} 条 -> {{{counts_str}}}")

            # 进度日志
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
            print(f"  [FAIL] {task.lngid}.pdf: {e}")
            await status_tracker.record_failure(
                task.lngid, "index", str(e), task.attempt, task.pdf_path
            )
            async with state.lock:
                state.index_failures += 1
        finally:
            # 入库完成后立即删除 spool 文件，释放磁盘空间
            spool_path.unlink(missing_ok=True)
            parsed_queue.task_done()

def load_pickle(path: Path) -> Any:
    """从磁盘反序列化 pickle 文件。"""
    with path.open("rb") as handle:
        return pickle.load(handle)


def cleanup_stale_event_artifacts(
    supervisor: LocalSupervisor,
    event: Dict[str, Any],
) -> None:
    """清理旧一代 Worker 产出的残留文件。
    当收到 generation 不匹配的事件时调用，防止旧进程的文件污染。
    """
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
    """清理上次运行时残留的未完成文件（因崩溃中断等原因留下）。
    在流水线启动时调用，避免旧残留文件干扰当前运行。
    """
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


# ═══════════════════════════════════════════════════════════════════════════
# 任务喂料器（阶段一）
# ─────────────────────────────────────────────────────────────────────────
# 扫描 PDF 目录，过滤已完成/永久失败的文件，将待处理任务放入 pending_queue
# ═══════════════════════════════════════════════════════════════════════════

async def feed_tasks(
    pending_queue: asyncio.Queue,
    pdf_dir: Path,
    limit: int,
    state: PipelineState,
    status_tracker: StatusTracker,
    max_attempts: int,
) -> None:
    """扫描 PDF 目录，生成 ParseTask 并放入 pending_queue。

    流程：
      1. 递归遍历 pdf_dir 下所有 .pdf 文件
      2. 从文件名提取 lngid
      3. 查 checkpoint：已成功 → 跳过；失败且 attempt 耗尽 → 跳过
      4. 之前失败但可重试的 → attempt 递增后重新放入
      5. 新文件 → attempt=1
      6. 达到 limit 上限或扫描完毕 → 放入 SCAN_DONE 哨兵
    """
    sequence = 0
    skipped_count = 0
    retry_count = 0

    for pdf_path in pdf_dir.rglob("*.pdf"):
        lngid = extract_lngid(pdf_path.name)
        if not lngid:
            continue

        # 断点续传：已成功处理 → 跳过
        if status_tracker.is_done(lngid):
            skipped_count += 1
            print(f"[SKIP] {pdf_path.name} 已处理，跳过")
            continue

        if status_tracker.should_skip(lngid, max_attempts):
            skipped_count += 1
            print(f"[SKIP] {pdf_path.name} 已达最大重试次数，跳过")
            continue

        # 检查是否有之前的失败记录 → attempt 递增
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
    print(f"\n待处理: {sequence} 篇（已跳过 {skipped_count} 篇，重试 {retry_count} 篇）\n")

async def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """流水线主入口（asyncio 主协程）。

    三阶段流水线：
      阶段一 feed_tasks:              扫描 PDF → pending_queue
      阶段二 run_parser_supervisor:   多 GPU Worker 解析 → result_queue → parsed_queue
      阶段三 index_worker (×N):       向量化 + ES 入库（多个并发协程）

    所有阶段并发运行：
      - feed_tasks 和 parser_supervisor 通过 pending_queue 连接
      - parser_supervisor 和 index_workers 通过 parsed_queue 连接
      - feed_tasks + parser_supervisor 先完成 → 然后发 None 关闭 indexers
    """
    global logger

    # ── 提高文件描述符上限（Linux），Windows 跳过 ─────────────
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

    # ── 日志初始化 ─────────────────────────────────────────
    logger = setup_logging(state_dir)

    print("=" * 70)
    print("[BATCH IMPORT] 本地 MinerU 批量导入到 7 个索引 (file_id = lngid)")
    print("=" * 70)
    print(f"PDF目录: {pdf_dir}")
    print(f"DB3路径: {args.db3_path}")
    print(f"状态目录: {state_dir}")
    print(f"设备类型: {args.device}  |  卡号: {args.gpus}  |  每卡Worker数: {args.workers_per_gpu}  |  显存: {args.vram}GB")
    print(f"入库协程数: {args.index_workers}")
    print(f"文件上限: {args.limit}")
    print(f"断点文件: {state_dir / 'pipeline_status.jsonl'}")
    print("=" * 70)

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

    # ── spool 目录（解析结果中转站）──────────────────────────
    spool_dir = state_dir / "spool" / "ready"   # 解析完成的 pickle 文件
    work_dir = state_dir / "spool" / "work"      # 解析中的临时文件
    spool_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    old_spools, old_work = cleanup_previous_run_artifacts(spool_dir, work_dir)
    if old_spools or old_work:
        logger.warning(
            "Cleaned artifacts from interrupted prior run: spool=%d work=%d",
            old_spools, old_work,
        )

    # ── 初始化组件 ────────────────────────────────────────
    status_tracker = StatusTracker(state_dir)
    await status_tracker.load()

    # GPU 配置解析：--gpus "0,1" --workers-per-gpu "2,2" → [0,0,1,1]
    gpu_slots = parse_worker_config(args.gpus, args.workers_per_gpu)

    # IO 线程池（用于 pickle 读写、SQLite 查询等阻塞操作）
    io_executor = ThreadPoolExecutor(max_workers=32, thread_name_prefix="io_")

    # ── 三阶段之间的消息队列 ──────────────────────────────
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=args.result_queue_size)     # Worker → Supervisor（跨进程）
    pending_queue: asyncio.Queue = asyncio.Queue(maxsize=args.task_queue_size)  # Feeder → Supervisor
    parsed_queue: asyncio.Queue = asyncio.Queue(maxsize=args.parsed_queue_size) # Supervisor → Indexer

    state = PipelineState()
    state.started_at = time.monotonic()
    state.parse_started_at = state.started_at
    state.index_started_at = state.started_at

    # ── 启动各阶段的协程/任务 ─────────────────────────────
    supervisor = LocalSupervisor(
        ctx, gpu_slots, result_queue, spool_dir, work_dir, args, state, status_tracker
    )

    feeder = asyncio.create_task(feed_tasks(            # 阶段一
        pending_queue, pdf_dir, args.limit, state, status_tracker, args.max_parse_attempts
    ))

    parser_task = asyncio.create_task(run_parser_supervisor(  # 阶段二
        supervisor, pending_queue, parsed_queue, state
    ))

    await asyncio.sleep(0)  # 让 Supervisor 有机会先启动 Worker

    # ── 阶段三：HTTP 客户端 + 入库协程 ───────────────────────
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
                    i, parsed_queue, client, args.db3_path,
                    io_executor, status_tracker, state, args
                )) for i in range(args.index_workers)
            ]

            # 等 feed_tasks 和 parser_supervisor 完成
            await asyncio.gather(feeder, parser_task)

            # 发 None 信号通知所有 indexer 退出
            for _ in indexers:
                await parsed_queue.put(None)

            # 等待 indexer 处理完队列中剩余的任务
            await parsed_queue.join()
            await asyncio.gather(*indexers)

    except BaseException:
        logger.critical("Pipeline crashed, shutting down...")
        await supervisor.shutdown()
        raise
    finally:
        result_queue.close()
        io_executor.shutdown(wait=True)

    # ── 汇总统计 ─────────────────────────────────────────
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

    # ── 终端汇总打印 ────────────────
    print("\n" + "=" * 70)
    print("[SUMMARY] 汇总")
    print("=" * 70)
    print(f"发现: {state.discovered} 篇  |  提交: {state.submitted} 篇")
    print(f"解析成功: {state.parsed} 篇  |  入库成功: {state.success} 篇")
    print(f"解析失败: {state.parse_failures} 篇  |  入库失败: {state.index_failures} 篇")
    print(f"超时: {state.timeouts} 篇  |  Worker 崩溃: {state.worker_deaths} 次")
    print(f"总耗时: {elapsed:.1f}s ({elapsed/3600:.2f}h)")
    print(f"解析速率: {final_parse_rate:.1f} PDF/h  |  入库速率: {final_index_rate:.1f} PDF/h")
    print(f"平均每篇: {elapsed / state.success:.2f}s" if state.success else "平均每篇: N/A")
    print(f"\n各库写入量:")
    for k, v in state.counts.items():
        print(f"  {k:<12}: {v}")
    print(f"\n每 GPU 解析量: {dict(state.per_gpu)}")
    print(f"\n统计已保存: {stats_file}")
    print("=" * 70)

    logger.info("Pipeline Finished.")
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

# ═══════════════════════════════════════════════════════════════════════════
# CLI 命令行接口
# ═══════════════════════════════════════════════════════════════════════════

def parse_worker_config(gpus_text: str, workers_text: str) -> List[int]:
    """解析 GPU 配置字符串，返回 GPU 槽位列表。

    示例：
      --gpus "0,1" --workers-per-gpu "2"    → [0, 0, 1, 1]  (每卡 2 个 Worker)
      --gpus "0,1" --workers-per-gpu "2,1"  → [0, 0, 1]     (GPU0 两个，GPU1 一个)
    """
    gpus = [int(v.strip()) for v in gpus_text.split(",") if v.strip()]
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPUs must be unique and non-empty")
    counts = [int(v.strip()) for v in workers_text.split(",") if v.strip()]
    if len(counts) == 1:
        counts *= len(gpus)  # 所有 GPU 使用相同 Worker 数
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
    """为 argparse 添加 --feature/--no-feature 开/关标志对。
    兼容 Python < 3.9。
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false")
    parser.set_defaults(**{name.replace("-", "_"): default})


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Million PDF Local MinerU Pipeline V2")
    # ── 路径相关 ──────────────────────────────────────
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR),
                        help="PDF 文件目录")
    parser.add_argument("--db3-path", default=str(DEFAULT_DB3_PATH),
                        help="SQLite 元数据库路径")
    parser.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR),
                        help="状态/日志/结果输出目录")
    # ── 任务规模 ──────────────────────────────────────
    parser.add_argument("--limit", type=int, default=1_000_000,
                        help="最多处理 PDF 数量，0 表示不限制")
    # ── GPU 配置 ──────────────────────────────────────
    parser.add_argument("--gpus", default="0",
                        help="使用的 GPU 编号，逗号分隔，如 0,1,2")
    parser.add_argument("--workers-per-gpu", default="1",
                        help="每张 GPU 的 Worker 数量，逗号分隔或单一值")
    parser.add_argument("--vram", type=int, default=8,
                        help="虚拟显存大小（GB），用于 MinerU 显存管理")
    # ── MinerU 参数 ────────────────────────────────────
    parser.add_argument("--lang", default="ch",
                        help="文档语言（ch/en）")
    parser.add_argument("--device", default="cuda", choices=["cuda", "npu", "cpu"],
                        help="设备类型: cuda(NVIDIA GPU) / npu(华为昇腾) / cpu(纯CPU)")
    _add_boolean_argument(parser, "formula-enable", True, "启用公式识别")
    _add_boolean_argument(parser, "table-enable", True, "启用表格识别")
    # ── 超时与重试 ────────────────────────────────────
    parser.add_argument("--parse-timeout", type=float, default=600.0,
                        help="单个 PDF 解析超时（秒）")
    parser.add_argument("--max-parse-attempts", type=int, default=2,
                        help="单篇 PDF 最大解析尝试次数")
    parser.add_argument("--worker-kill-grace", type=float, default=10.0,
                        help="Worker 终止优雅等待时间（秒）")
    # ── 队列配置 ──────────────────────────────────────
    parser.add_argument("--task-queue-size", type=int, default=64,
                        help="任务队列容量")
    parser.add_argument("--result-queue-size", type=int, default=64,
                        help="结果队列容量")
    parser.add_argument("--parsed-queue-size", type=int, default=32,
                        help="已解析队列容量")
    # ── 入库配置 ──────────────────────────────────────
    parser.add_argument("--index-workers", type=int, default=6,
                        help="入库协程数量")
    parser.add_argument("--http-timeout", type=float, default=300.0,
                        help="HTTP 请求超时（秒）")
    # ── 日志配置 ──────────────────────────────────────
    parser.add_argument(
        "--progress-interval", type=int, default=100,
        help="每完成多少篇打印一次解析和入库速率",
    )
    return parser

def validate_args(args: argparse.Namespace) -> None:
    """校验命令行参数合法性。"""
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
    parse_worker_config(args.gpus, args.workers_per_gpu)  # 提前解析以便尽早报错


def main():
    """CLI 入口。"""
    args = build_parser().parse_args()
    validate_args(args)
    asyncio.run(run_pipeline(args))

if __name__ == "__main__":
    mp.freeze_support()
    # 使用 spawn 方式启动子进程，避免 CUDA/fork 不兼容问题
    mp.set_start_method("spawn", force=True)
    main()
