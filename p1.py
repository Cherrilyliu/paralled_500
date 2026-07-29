"""Pure Markdown-to-Chunk conversion used by local MinerU workers.

This module intentionally has no HTTP clients, service singletons, MinerU imports,
or CUDA initialization side effects. It mirrors the structural parsing behavior in
``file_parser/service.py`` closely enough for the local and API pipelines to emit
the same project ``Chunk`` type.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

try:
    # Deployment scripts add <project>/algorithm to sys.path.
    from RAG.datamerge.data_types import Chunk, ChunkStrategy
except ModuleNotFoundError:
    # Package-style imports from the repository root use algorithm.RAG.
    from algorithm.RAG.datamerge.data_types import Chunk, ChunkStrategy


@dataclass
class Section:
    level: int
    title: str
    content: str = ""
    page_num: int = 1
    children: List["Section"] = field(default_factory=list)


_TITLE_KEYWORDS = (
    "摘要", "引言", "前言", "绪论", "方法", "方法学", "结果", "讨论",
    "结论", "结语", "结 语", "参考文献", "参 考 文 献", "致谢", "致 谢",
    "附录", "附 录", "实验", "实 验", "相关工作", "相 关 工 作", "背景",
    "背 景", "理论", "理论基础", "分析", "展望", "评估", "设计", "实现",
    "验证", "测试", "对比", "总结",
)

_SECTION_TYPES = (
    ("摘要", ("摘要", "abstract")),
    ("关键词", ("关键词", "key words", "keywords")),
    ("引言", ("引言", "前言", "绪论", "introduction", "intro")),
    ("背景", ("背景", "background", "相关背景")),
    ("相关工作", ("相关工作", "related work", "related works")),
    ("方法", ("方法", "方法学", "methods", "methodology", "模型", "model")),
    ("实验", ("实验", "试验", "experiment", "experiments")),
    ("结果", ("结果", "results", "findings")),
    ("讨论", ("讨论", "discussion")),
    ("分析", ("分析", "analysis")),
    ("结论", ("结论", "结语", "总结", "conclusion", "未来工作", "future work")),
    ("致谢", ("致谢", "acknowledgements", "acknowledgment")),
    ("参考文献", ("参考文献", "references", "reference")),
    ("附录", ("附录", "appendix")),
    ("设计", ("设计", "design")),
    ("实现", ("实现", "implementation")),
    ("验证", ("验证", "verification", "validation")),
    ("测试", ("测试", "test", "testing")),
    ("对比", ("对比", "comparison")),
    ("评估", ("评估", "evaluation")),
    ("展望", ("展望", "未来工作", "future work")),
)


def _clean_text(text: str, is_table: bool = False) -> str:
    text = text.replace("\x00", "").strip()
    if is_table:
        return text
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_sections(markdown: str) -> List[Section]:
    roots: List[Section] = []
    stack: List[Section] = []
    preamble: List[str] = []

    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if stack and stack[-1].content and not stack[-1].content.endswith("\n\n"):
                stack[-1].content += "\n"
            continue

        match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            page_match = re.search(r"\(p\.(\d+)\)$", title)
            page_num = int(page_match.group(1)) if page_match else 1
            title = re.sub(r"\s*\(p\.\d+\)$", "", title)
            new_section = Section(level, _clean_text(title), page_num=page_num)
        else:
            compact = re.sub(r"\s+", "", stripped)
            title_like = any(
                re.sub(r"\s+", "", keyword).lower() in compact.lower()
                for keyword in _TITLE_KEYWORDS
            ) and len(compact) < 50
            if not title_like:
                if stack:
                    stack[-1].content += line + "\n"
                else:
                    preamble.append(line)
                continue
            level = 2
            new_section = Section(level, _clean_text(stripped))

        while stack and stack[-1].level >= new_section.level:
            stack.pop()
        if stack:
            stack[-1].children.append(new_section)
        else:
            roots.append(new_section)
        stack.append(new_section)

    if preamble:
        preamble_text = "\n".join(preamble).strip()
        if roots:
            roots[0].content = preamble_text + "\n\n" + roots[0].content
        else:
            roots.append(Section(1, "正文", preamble_text))
    return roots


def _mark_special_sections(sections: List[Section]) -> None:
    groups = {
        "references": ("参考文献", "reference", "references", "参 考 文 献"),
        "appendix": ("附录", "appendix", "附 录"),
        "acknowledgement": ("致谢", "acknowledgement", "acknowledgments", "致 谢"),
    }

    def visit(section: Section) -> None:
        lowered = section.title.lower()
        for label, keywords in groups.items():
            if any(keyword in lowered for keyword in keywords):
                section.title = f"[{label}] {section.title}"
                break
        for child in section.children:
            visit(child)

    for root in sections:
        visit(root)


def _parse_references(content: str) -> List[str]:
    content = re.sub(r"[（(][上下](?:接|转)第\s*\d+\s*页[）)]", "", content)
    matches = re.findall(r"\[(\d+)\]\s*([^\[]+?)(?=\[\d+\]|$)", content, re.DOTALL)
    references: List[str] = []
    for number, text in matches:
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[（(]责任编辑[：:].*", "", text)
        if text:
            references.append(f"[{number}] {text}")
    return references


def _section_type(path: str) -> str:
    lowered = path.lower()
    for label, keywords in _SECTION_TYPES:
        if any(keyword in lowered for keyword in keywords):
            return label
    return "正文"


def _build_chunks(sections: List[Section], filename: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    sequence = 0

    def append_chunk(text: str, node: Section, path: str, element_type: str, metadata: Dict) -> None:
        nonlocal sequence
        chunk_id = hashlib.md5(f"{filename}_{path}_{sequence}".encode()).hexdigest()[:16]
        chunks.append(Chunk(
            id=chunk_id,
            text=text,
            doc_name=filename,
            page_num=node.page_num,
            chunk_seq=sequence + 1,
            element_type=element_type,
            section_path=path,
            strategy=ChunkStrategy.STRUCTURE,
            metadata=metadata,
        ))
        sequence += 1

    def visit(node: Section, parents: List[str]) -> None:
        path = " > ".join(parents + [node.title]) if parents else node.title
        content = node.content.strip()
        if "参考文献" in node.title or "references" in node.title.lower():
            references = _parse_references(content)
            if references:
                for reference in references:
                    append_chunk(reference, node, path, "参考文献条目", {
                        "bbox": None, "text_level": 0, "is_reference": True,
                    })
                return

        for paragraph in content.split("\n\n") if content else []:
            paragraph = paragraph.strip()
            if len(paragraph) < 10:
                continue
            is_table = "|" in paragraph and paragraph.count("|") > 2
            paragraph = _clean_text(paragraph, is_table)
            pieces = [paragraph]
            if len(paragraph) > 1500:
                pieces = [piece.strip() for piece in paragraph.replace("。", "。\n").splitlines() if piece.strip()]
            for piece in pieces:
                if len(paragraph) > 1500 and not piece.endswith("。"):
                    piece += "。"
                append_chunk(piece, node, path, _section_type(path), {
                    "bbox": None, "text_level": node.level, "is_table": is_table,
                })

        for child in node.children:
            visit(child, parents + [node.title])

    for section in sections:
        visit(section, [])
    return chunks


class MarkdownStructureAdapter:
    """Convert MinerU Markdown into project Chunk objects without service setup."""

    def build_chunks(self, markdown: str, filename: str) -> Tuple[List[Chunk], Dict[str, str]]:
        if not markdown or not markdown.strip():
            raise ValueError("MinerU Markdown 内容为空")
        sections = _parse_sections(markdown)
        if not sections:
            raise ValueError("未能从 MinerU Markdown 构建章节")
        _mark_special_sections(sections)
        chunks = _build_chunks(sections, filename)
        if not chunks:
            raise ValueError("未能从 MinerU Markdown 构建 Chunk")
        metadata = {
            "title": sections[0].title,
            "abstract": self._find_abstract(sections),
        }
        return chunks, metadata

    def _find_abstract(self, sections: List[Section]) -> str:
        def visit(section: Section) -> str:
            if "摘要" in section.title or "abstract" in section.title.lower():
                return section.content.strip()[:2000]
            for child in section.children:
                found = visit(child)
                if found:
                    return found
            return ""

        for section in sections:
            found = visit(section)
            if found:
                return found
        return ""
