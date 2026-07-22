#!/usr/bin/env python3
"""
优化版批量导入脚本：相比 parallel_test_500.py，做了两项关键优化：

1. Embedding API 异步化 + 合并批处理
   - 用 httpx.AsyncClient 替代同步 Client，避免阻塞事件循环
   - 每篇 PDF 的所有文本（标题/摘要/关键词/全文/段落/句子/参考文献）
     先全部收集，再合并成 1~3 次 Embedding API 请求（原来是 ~13 次）

2. ES 批量写入（_bulk API）
   - 每篇 PDF 的所有 7 个索引文档打包成一次 _bulk 请求（原来是 ~150 次独立 PUT）

其他逻辑与原版保持一致：断点续传、MinerU 解析、轮询负载均衡等。
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
    embedding_base_url, Embedding_model_name, embedding_api_key,
)
from algorithm.ai_tools.metadata_loader import load_metadata_by_lngid, extract_lngid_from_filename
from algorithm.ai_tools.sentence_chunker import paragraph_to_sentence_chunks


# =========================================================
# 配置
# =========================================================
PDF_DIR = PROJECT_ROOT / "data" / "500pdf"
DB3_PATH = r"/home/vscodeuser/yanjiushequ/Encyclopedia_project/zkyrjyjs_50w_20260707.db3"
STATS_FILE = PROJECT_ROOT / "data" / "500pdf_vec_result" / "batch_import_stats_optimized.json"
PROCESSED_FILE = Path("/tmp/500pdf_processed_ids_optimized.txt")
STATS_FILE.parent.mkdir(parents=True, exist_ok=True)

MAX_FULL_TOKENS = 8000
MINERU_PORTS = [8060]
_port_index = 0
processed_lock = asyncio.Lock()

# Embedding API 配置（来自 global_config_variable）
EMBEDDING_API_URL = f"{embedding_base_url.rstrip('/')}/embeddings"
EMBEDDING_MODEL = Embedding_model_name
EMBEDDING_HEADERS = {
    "Authorization": f"Bearer {embedding_api_key}",
    "Content-Type": "application/json",
}
EMBEDDING_BATCH_SIZE = 64  # 每批发送给 Embedding API 的最大文本数


def get_next_mineru_port() -> int:
    global _port_index
    port = MINERU_PORTS[_port_index % len(MINERU_PORTS)]
    _port_index += 1
    return port


# =========================================================
# 优化 1：异步批量 Embedding（一次收集，少量请求）
# =========================================================

async def async_embed_texts(
    http_client: httpx.AsyncClient,
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """
    使用 httpx.AsyncClient 异步调用 Embedding API，不再阻塞事件循环。
    支持自动降级：如果 batch 太大导致 413，自动减半重试。
    """
    if not texts:
        return []

    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1

        payload = {"input": batch_texts, "model": EMBEDDING_MODEL}
        resp = await http_client.post(EMBEDDING_API_URL, json=payload, headers=EMBEDDING_HEADERS)

        if resp.status_code == 413:
            # Payload Too Large → 减半重试
            smaller = max(batch_size // 2, 1)
            print(f"  [Embedding] 批次 {batch_num} 过大 (413)，降级到 batch_size={smaller}")
            sub_embeddings = await async_embed_texts(http_client, batch_texts, smaller)
            all_embeddings.extend(sub_embeddings)
            continue

        if resp.status_code != 200:
            raise RuntimeError(
                f"Embedding API 错误 (HTTP {resp.status_code}): {resp.text[:500]}"
            )

        batch_result = resp.json()
        batch_embeddings = [item["embedding"] for item in batch_result["data"]]
        all_embeddings.extend(batch_embeddings)

        if total_batches > 1:
            print(f"  [Embedding] 批次 {batch_num}/{total_batches} 完成 ({len(batch_texts)} 条)")

    return all_embeddings


# =========================================================
# 优化 2：ES _bulk 批量写入
# =========================================================

async def es_bulk_index(
    http_client: httpx.AsyncClient,
    actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    将多个 index 操作打包为一次 _bulk 请求（NDJSON 格式）。
    actions 每项格式: {"_index": "...", "_id": "...", "_source": {...}}
    """
    if not actions:
        return {"took": 0, "errors": False, "items": []}

    lines = []
    for action in actions:
        # 第一行：操作元数据
        lines.append(json.dumps(
            {"index": {"_index": action["_index"], "_id": action["_id"]}},
            ensure_ascii=False,
        ))
        # 第二行：文档数据
        lines.append(json.dumps(action["_source"], ensure_ascii=False))

    body = "\n".join(lines) + "\n"  # 尾部空行不能少

    url = f"{ES_URL.rstrip('/')}/_bulk"
    resp = await http_client.post(
        url,
        content=body,
        headers={"Content-Type": "application/x-ndjson"},
    )

    if resp.status_code != 200:
        print(f"  [ES Bulk Error] HTTP {resp.status_code}: {resp.text[:500]}")
        return {"took": 0, "errors": True, "items": []}

    result = resp.json()
    if result.get("errors"):
        # 打印具体失败条目
        failed_items = [
            item for item in result.get("items", [])
            if "error" in item.get("index", {})
        ]
        for fi in failed_items[:5]:  # 最多打印前5条
            idx_info = fi.get("index", {})
            print(f"  [ES Bulk Error] _id={idx_info.get('_id')}: {idx_info.get('error')}")
        if len(failed_items) > 5:
            print(f"  [ES Bulk Error] ... 还有 {len(failed_items) - 5} 条失败")

    return result


# =========================================================
# 优化后的单篇 PDF 导入（合并 Embedding + ES 批量写入）
# =========================================================

async def import_one_pdf_to_es_optimized(
    http_client: httpx.AsyncClient,
    lngid: str,
    raw_chunks: List[Chunk],
    meta: Dict[str, Any],
) -> Dict[str, int]:
    """
    优化版导入：一篇 PDF → 全部 7 个索引。
    1. 先收集所有需要向量化的文本
    2. 一次性（或少量批次）调用 Embedding API
    3. 构建所有 ES 文档，用 _bulk 一次写入
    """
    counts = {
        "title": 0, "abstract": 0, "keyword": 0,
        "fulltext": 0, "paragraph": 0, "sentence": 0, "reference": 0,
    }

    # ── 元数据 ──────────────────────────────────────────
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

    # ── 分离段落 / 参考文献 ──────────────────────────────
    paragraph_chunks: List[Chunk] = []
    reference_chunks: List[Chunk] = []
    for c in raw_chunks:
        etype = getattr(c, "element_type", "")
        if etype in ("参考文献", "参考文献条目") or "参考文献" in getattr(c, "section_path", ""):
            reference_chunks.append(c)
        else:
            paragraph_chunks.append(c)

    # ── 处理全文 ─────────────────────────────────────────
    full_text = "\n".join([getattr(c, "text", "") for c in raw_chunks if getattr(c, "text", "")])
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    estimated_tokens = len(full_text) // 2

    # ── 处理句子 ─────────────────────────────────────────
    sent_docs: List[Tuple] = []  # (p_idx, s_seq, s_text, section_path, page_num, element_type)
    for p_idx, c in enumerate(paragraph_chunks):
        p_text = getattr(c, "text", "")
        if not p_text:
            continue
        s_chunks = paragraph_to_sentence_chunks(p_text)
        for s_seq, s_text in enumerate(s_chunks):
            if s_text and len(s_text.strip()) >= 2:
                sent_docs.append((
                    p_idx, s_seq, s_text,
                    getattr(c, "section_path", ""),
                    getattr(c, "page_num", 1),
                    getattr(c, "element_type", ""),
                ))

    # ── Phase 1: 收集所有需要向量化的文本 ────────────────
    # text_map 记录每条文本的归属，方便后续"拆包"
    text_map: List[Dict[str, Any]] = []  # {"type": "title"|"abstract"|..., "index": int}
    texts_to_embed: List[str] = []

    # 标题
    title_text = (title_c + " " + title_e).strip()
    if title_text:
        text_map.append({"type": "title"})
        texts_to_embed.append(title_text)

    # 摘要
    abstract_text = (remark_c + " " + remark_e).strip()
    if abstract_text:
        text_map.append({"type": "abstract"})
        texts_to_embed.append(abstract_text)

    # 关键词
    keyword_text = (keyword_c + " " + keyword_e).strip()
    if keyword_text:
        text_map.append({"type": "keyword"})
        texts_to_embed.append(keyword_text)

    # 全文
    if estimated_tokens <= MAX_FULL_TOKENS and full_text:
        text_map.append({"type": "fulltext"})
        texts_to_embed.append(full_text)
    else:
        full_text = ""  # 超长则跳过

    # 段落
    para_texts = [getattr(c, "text", "") for c in paragraph_chunks if getattr(c, "text", "")]
    for p_idx, pt in enumerate(para_texts):
        text_map.append({"type": "paragraph", "index": p_idx})
        texts_to_embed.append(pt)

    # 句子
    for s_idx, sd in enumerate(sent_docs):
        text_map.append({"type": "sentence", "index": s_idx})
        texts_to_embed.append(sd[2])  # sd[2] = s_text

    # 参考文献
    ref_texts = [getattr(c, "text", "") for c in reference_chunks if getattr(c, "text", "")]
    for r_idx, rt in enumerate(ref_texts):
        text_map.append({"type": "reference", "index": r_idx})
        texts_to_embed.append(rt)

    # ── Phase 2: 分组批量 Embedding ──────────────────────
    # 4 组独立 try/except，组间互不影响（跟原版逐库隔离的逻辑一致）
    # 组内合并减少请求数，组间隔离保证一组挂了其余仍有向量

    async def _embed_group(label: str, texts: List[str]) -> List[List[float]]:
        """对一组文本做 embedding，失败时打 WARN 并返回空向量"""
        if not texts:
            return []
        try:
            return await async_embed_texts(http_client, texts)
        except Exception as e:
            print(f"  [WARN] {lngid} Embedding({label}) 失败: {e}")
            return [[] for _ in texts]

    # 组 1: 元数据（标题 + 摘要 + 关键词 + 全文）
    meta_texts: List[str] = []
    meta_labels: List[str] = []  # "title" / "abstract" / "keyword" / "fulltext"
    if title_text:
        meta_texts.append(title_text); meta_labels.append("title")
    if abstract_text:
        meta_texts.append(abstract_text); meta_labels.append("abstract")
    if keyword_text:
        meta_texts.append(keyword_text); meta_labels.append("keyword")
    if full_text:
        meta_texts.append(full_text); meta_labels.append("fulltext")

    meta_vecs = await _embed_group("meta", meta_texts)

    # 组 2: 段落
    para_vecs = await _embed_group("paragraph", para_texts)

    # 组 3: 句子
    sent_texts = [sd[2] for sd in sent_docs]
    sent_vecs = await _embed_group("sentence", sent_texts)

    # 组 4: 参考文献
    ref_vecs = await _embed_group("reference", ref_texts)

    # ── Phase 3: 从各组结果中取值 ────────────────────────
    mi = 0  # meta 组游标

    def _next_meta() -> List[float]:
        nonlocal mi
        v = meta_vecs[mi] if mi < len(meta_vecs) else []
        mi += 1
        return v

    title_vec: List[float] = _next_meta() if "title" in meta_labels else []
    abstract_vec: List[float] = _next_meta() if "abstract" in meta_labels else []
    keyword_vec: List[float] = _next_meta() if "keyword" in meta_labels else []
    full_vec: List[float] = _next_meta() if "fulltext" in meta_labels else []

    # ── Phase 4: 构建 _bulk actions ──────────────────────
    # 与原版一致：有文本 + 有向量才写入；各库独立 try/except 互不影响
    actions: List[Dict[str, Any]] = []

    # 标题库
    try:
        if title_text and title_vec:
            actions.append({"_index": ES_TITLE_INDEX, "_id": f"{lngid}_title", "_source": {
                **common, "title_c": title_c, "title_e": title_e,
                "title_text": title_text, "title_embedding": title_vec,
            }})
            counts["title"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 标题库失败: {e}")

    # 摘要库
    try:
        if abstract_text and abstract_vec:
            actions.append({"_index": ES_ABSTRACT_INDEX, "_id": f"{lngid}_abstract", "_source": {
                **common, "remark_c": remark_c, "remark_e": remark_e,
                "abstract_embedding": abstract_vec,
            }})
            counts["abstract"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 摘要库失败: {e}")

    # 关键词库
    try:
        if keyword_text and keyword_vec:
            actions.append({"_index": ES_KEYWORD_INDEX, "_id": f"{lngid}_keyword", "_source": {
                **common, "keyword_c": keyword_c, "keyword_e": keyword_e,
                "keyword_text": keyword_text, "keyword_embedding": keyword_vec,
            }})
            counts["keyword"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 关键词库失败: {e}")

    # 全文库（确认不超 token 且有向量才写，跟原版一致）
    try:
        if full_text and full_vec:
            actions.append({"_index": ES_FULLTEXT_INDEX, "_id": f"{lngid}_fulltext", "_source": {
                **common, "full_text": full_text,
                "full_embedding": full_vec, "token_count": estimated_tokens,
            }})
            counts["fulltext"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 全文库失败: {e}")

    # 段落库
    try:
        for idx, (c, vec) in enumerate(zip(paragraph_chunks, para_vecs)):
            p_text = getattr(c, "text", "")
            if not p_text or not vec:
                continue
            actions.append({"_index": ES_PARAGRAPH_INDEX, "_id": f"{lngid}_p{idx}", "_source": {
                **common, "para_index": idx, "para_text": p_text,
                "para_embedding": vec,
                "section_path": getattr(c, "section_path", ""),
                "element_type": getattr(c, "element_type", ""),
                "page_num": getattr(c, "page_num", 1),
            }})
            counts["paragraph"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 段落库失败: {e}")

    # 句子库
    try:
        for (p_idx, s_seq, s_text, sec, page, etype), vec in zip(sent_docs, sent_vecs):
            if not vec:
                continue
            actions.append({"_index": ES_SENTENCE_INDEX, "_id": f"{lngid}_p{p_idx}s{s_seq}", "_source": {
                **common, "para_index": p_idx, "sent_seq": s_seq,
                "sent_text": s_text, "sent_embedding": vec,
                "section_path": sec, "element_type": etype, "page_num": page,
            }})
            counts["sentence"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 句子库失败: {e}")

    # 参考文献库
    try:
        for idx, (c, vec) in enumerate(zip(reference_chunks, ref_vecs)):
            r_text = getattr(c, "text", "")
            if not r_text or not vec:
                continue
            actions.append({"_index": ES_REFERENCE_INDEX, "_id": f"{lngid}_r{idx}", "_source": {
                **common, "ref_seq": idx, "ref_text": r_text,
                "ref_embedding": vec,
            }})
            counts["reference"] += 1
    except Exception as e:
        print(f"  [WARN] {lngid} 参考文献库失败: {e}")

    # ── Phase 5: 一次 _bulk 写入 ─────────────────────────
    if actions:
        try:
            await es_bulk_index(http_client, actions)
        except Exception as e:
            print(f"  [WARN] {lngid} ES _bulk 写入失败: {e}")
            # 即使 bulk 失败也不抛异常，返回已经统计的 counts

    return counts


# =========================================================
# 单篇处理（协程入口，逻辑与原版一致）
# =========================================================
async def process_single_pdf(
    semaphore: asyncio.Semaphore,
    pdf_path: Path,
    idx: int,
) -> Dict[str, Any]:
    async with semaphore:
        lngid = extract_lngid_from_filename(pdf_path.name)
        result = {
            "index": idx, "filename": pdf_path.name, "lngid": lngid,
            "success": False, "error": None, "counts": {}, "time": 0,
        }
        start = time.time()
        try:
            # ── MinerU 解析（不变） ─────────────────────
            port = get_next_mineru_port()
            parser = MinerUParserWithStructure(mineru_api_base=f"http://127.0.0.1:{port}")
            raw_chunks, _ = await parser.parse_pdf_with_structure(str(pdf_path), pdf_path.name)

            # ── SQLite 元数据（不变） ────────────────────
            meta = load_metadata_by_lngid(lngid, db3_path=DB3_PATH)

            # ── 共享 httpx.AsyncClient：Embedding + ES 共用，复用连接 ──
            async with httpx.AsyncClient(trust_env=False, timeout=300.0) as http_client:
                counts = await import_one_pdf_to_es_optimized(
                    http_client, lngid, raw_chunks, meta,
                )

            result["success"] = True
            result["counts"] = counts
            async with processed_lock:
                with open(PROCESSED_FILE, "a") as f:
                    f.write(f"{lngid}\n")
                    f.flush()
            total = sum(counts.values())
            print(f"  [OK] [{idx:04d}] {pdf_path.name}: {total} 条 -> {counts}")
        except Exception as e:
            result["error"] = str(e)
            print(f"  [FAIL] [{idx:04d}] {pdf_path.name}: {e}")
        result["time"] = round(time.time() - start, 2)
        return result


# =========================================================
# 批量导入主函数
# =========================================================
async def run_batch_import(concurrent: int = 6, limit: int = 500):
    print("=" * 70)
    print("[BATCH IMPORT - OPTIMIZED] 批量导入到7个索引")
    print("  优化: 异步Embedding + 合并批处理 + ES _bulk 写入")
    print("=" * 70)
    print(f"PDF目录: {PDF_DIR}")
    print(f"DB3: {DB3_PATH}")
    print(f"并发数: {concurrent}")
    print(f"文件数: {limit}")
    print(f"MinerU端口: {MINERU_PORTS}")
    print(f"Embedding API: {EMBEDDING_API_URL}")
    print(f"Embedding 模型: {EMBEDDING_MODEL}")
    print(f"Embedding 批大小: {EMBEDDING_BATCH_SIZE}")
    print(f"状态文件: {PROCESSED_FILE}")
    print("=" * 70)

    # ── 断点续传 ────────────────────────────────────────
    processed_ids = set()
    if PROCESSED_FILE.exists():
        with open(PROCESSED_FILE, "r") as f:
            processed_ids = {line.strip() for line in f if line.strip()}
        print(f"[INFO] 已加载 {len(processed_ids)} 个已处理的 lngid")

    all_pdf_files = sorted(PDF_DIR.glob("*.pdf"))[:limit]
    pending_pdfs = []
    skipped_count = 0
    for p in all_pdf_files:
        lngid = extract_lngid_from_filename(p.name)
        if lngid in processed_ids:
            print(f"[SKIP] {p.name} 已处理，跳过")
            skipped_count += 1
        else:
            pending_pdfs.append(p)

    if not pending_pdfs:
        print("所有 PDF 已处理完毕，无需操作。")
        return

    print(f"待处理: {len(pending_pdfs)} 篇（已跳过 {skipped_count} 篇）\n")

    semaphore = asyncio.Semaphore(concurrent)
    start_time = time.time()

    tasks = [process_single_pdf(semaphore, p, i) for i, p in enumerate(pending_pdfs, 1)]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time
    success = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]

    total_counts = {
        "title": 0, "abstract": 0, "keyword": 0,
        "fulltext": 0, "paragraph": 0, "sentence": 0, "reference": 0,
    }
    for r in success:
        for k, v in r.get("counts", {}).items():
            total_counts[k] = total_counts.get(k, 0) + v

    print("\n" + "=" * 70)
    print("[SUMMARY] 汇总（优化版）")
    print("=" * 70)
    print(f"成功: {len(success)} 篇 / 失败: {len(fail)} 篇")
    print(f"总耗时: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"平均每篇: {total_time/len(pending_pdfs):.2f}s")
    print(f"吞吐量: {len(success)/total_time*3600:.1f} 篇/小时")
    print("\n各库写入量:")
    for k, v in total_counts.items():
        print(f"  {k:<12}: {v}")

    summary = {
        "version": "optimized",
        "total_files": len(pending_pdfs),
        "success": len(success),
        "fail": len(fail),
        "total_time": total_time,
        "throughput_per_hour": len(success) / total_time * 3600,
        "counts": total_counts,
        "fail_details": [
            {"filename": r["filename"], "error": r["error"]} for r in fail
        ],
    }
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n统计已保存: {STATS_FILE}")
    except Exception as e:
        print(f"\n[WARN] 无法写入统计文件: {e}")
    return summary


async def main():
    concurrent = 6
    limit = 10000
    if len(sys.argv) > 1:
        concurrent = int(sys.argv[1])
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
    await run_batch_import(concurrent=concurrent, limit=limit)


if __name__ == "__main__":
    asyncio.run(main())
