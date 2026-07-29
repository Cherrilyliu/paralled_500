"""MinerU Markdown -> Chunk adapter extracted from ``file_parser/service.py``.

The conversion rules in this module intentionally preserve the existing service
behavior. This module has no HTTP client, MinerU import, CUDA initialization, or
service singleton side effects, so local spawn workers can import it safely.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

try:
    from RAG.datamerge.data_types import Chunk, ChunkStrategy
except ModuleNotFoundError:
    from algorithm.RAG.datamerge.data_types import Chunk, ChunkStrategy


class UniversalTextCleaner:
    """The text cleaning rules used by ``service.py`` chunk conversion."""

    @classmethod
    def clean_text(cls, raw_text: str, is_table: bool = False) -> str:
        """增强版：彻底清洗文本，表格内容保留空格"""
        if not raw_text:
            return ""

        text = raw_text

        # 如果是表格内容，只做基本的空白清理，不处理空格
        if is_table:
            return text.strip()

        # 1. 移除所有HTML/XML标签
        text = re.sub(r'<[^>]+>', '', text)

        # 2. 处理单引号分隔符（学术论文常见）
        text = text.replace("'", ", ")

        # 3. 处理中文括号和英文括号混用
        text = text.replace("（", "(").replace("）", ")")

        # 4. 处理斜杠前后的空格： "a / b" → "a/b"
        text = re.sub(r'\s*/\s*', '/', text)

        # 5. 移除括号内的多余空格： "( text )" → "(text)"
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)

        # 6. 移除多余逗号空格
        text = re.sub(r',\s+,', ',', text)

        # 7. 处理"－"连字符（中文破折号）转英文连字符
        text = text.replace('－', '-')

        # 8. 规范空白字符
        text = re.sub(r'\s+', ' ', text)

        # 9. 去除首尾空白
        text = text.strip()

        return text


@dataclass
class Section:
    """章节节点，用于构建文档的树形结构"""

    level: int
    title: str
    content: str = ""
    page_num: int = 1
    children: List["Section"] = field(default_factory=list)

    def add_child(self, child: "Section"):
        self.children.append(child)


class MarkdownStructureAdapter:
    """The existing service.py Markdown-to-Chunk conversion without API setup."""

    def build_chunks(self, markdown: str, filename: str) -> Tuple[List[Chunk], Dict[str, Any]]:
        if not markdown:
            raise ValueError("MinerU返回的Markdown内容为空")
        root_sections = self._parse_markdown_to_sections(markdown)
        self._mark_special_sections(root_sections)
        chunks = self._build_chunks_from_sections(root_sections, filename)
        if not chunks:
            raise ValueError("未能从MinerU Markdown构建Chunk")
        # 正式论文元数据由调用方根据 lngid 从 SQLite 读取。
        return chunks, {}

    def _parse_markdown_to_sections(self, md_content: str) -> List[Section]:
        """解析 Markdown，将 #, ##, ### 转换为树形章节结构，支持带空格的标题"""
        lines = md_content.split('\n')
        root_sections = []
        stack = []

        title_keywords = [
            '摘要', '引言', '前言', '绪论', '方法', '方法学', '结果', '讨论',
            '结论', '结语', '结 语',
            '参考文献', '参 考 文 献', '致谢', '致 谢', '附录', '附 录',
            '实验', '实 验',
            '相关工作', '相 关 工 作', '背景', '背 景',
            '理论', '理论基础',
            '分析', '展望', '评估', '设计', '实现', '验证', '测试', '对比', '总结'
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            match = re.match(r'^(#{1,4})\s+(.+)$', stripped)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                page_match = re.search(r'\(p\.(\d+)\)$', title)
                page_num = int(page_match.group(1)) if page_match else 1
                if page_match:
                    title = re.sub(r'\s*\(p\.\d+\)$', '', title)

                title = UniversalTextCleaner.clean_text(title)
                new_section = Section(level=level, title=title, page_num=page_num)

                while stack and stack[-1].level >= level:
                    stack.pop()
                if stack:
                    stack[-1].children.append(new_section)
                else:
                    root_sections.append(new_section)
                stack.append(new_section)
            else:
                no_space_text = re.sub(r'\s+', '', stripped)
                found_title = False
                for kw in title_keywords:
                    clean_kw = re.sub(r'\s+', '', kw)
                    if clean_kw in no_space_text and len(no_space_text) < 50:
                        title = stripped
                        new_section = Section(level=2, title=title, page_num=1)
                        while stack and stack[-1].level >= 2:
                            stack.pop()
                        if stack:
                            stack[-1].children.append(new_section)
                        else:
                            root_sections.append(new_section)
                        stack.append(new_section)
                        found_title = True
                        break

                if not found_title:
                    if stack:
                        stack[-1].content += line + "\n"
                    elif root_sections:
                        root_sections[-1].content += line + "\n"

        return root_sections

    def _mark_special_sections(self, sections: List[Section]):
        """递归标记参考文献、致谢等特殊章节，方便后续识别"""
        special_keywords = {
            'references': ['参考文献', 'reference', 'references', '参 考 文 献'],
            'appendix': ['附录', 'appendix', '附 录'],
            'acknowledgement': ['致谢', 'acknowledgement', 'acknowledgments', '致 谢'],
        }

        def traverse(node: Section):
            title_lower = node.title.lower()
            for key, keywords in special_keywords.items():
                if any(kw in title_lower for kw in keywords):
                    node.title = f"[{key}] {node.title}"
                    break
            for child in node.children:
                traverse(child)

        for sec in sections:
            traverse(sec)

    def _parse_references(self, content: str) -> List[str]:
        """从参考文献章节的内容中提取每一条参考文献"""
        refs = []
        cleaned_content = re.sub(r'[（(]上接第\s*\d+\s*页[）)]', '', content)
        cleaned_content = re.sub(r'[（(]下转第\s*\d+\s*页[）)]', '', cleaned_content)

        pattern = r'\[(\d+)\]\s*([^\[]+?)(?=\[\d+\]|$)'
        matches = re.findall(pattern, cleaned_content, re.DOTALL)

        if matches:
            for num, ref_text in matches:
                ref_text = ref_text.strip()
                ref_text = re.sub(r'\s+', ' ', ref_text)
                ref_text = re.sub(r'[（(]责任编辑[：:].*', '', ref_text)
                ref_text = re.sub(r'\)\)$', ')', ref_text)
                if ref_text:
                    refs.append(f"[{num}] {ref_text}")

        return refs

    def _map_section_type(self, section_path: str) -> str:
        """根据章节路径的关键词，返回中文章节类型"""
        if not section_path:
            return "正文"

        mapping = [
            ("摘要", ["摘要", "abstract"]),
            ("关键词", ["关键词", "key words", "keywords"]),
            ("引言", ["引言", "前言", "绪论", "introduction", "intro"]),
            ("背景", ["背景", "background", "相关背景"]),
            ("相关工作", ["相关工作", "related work", "related works"]),
            ("方法", ["方法", "方法学", "methods", "methodology", "模型", "model"]),
            ("实验", ["实验", "试验", "experiment", "experiments"]),
            ("结果", ["结果", "results", "findings"]),
            ("讨论", ["讨论", "discussion"]),
            ("分析", ["分析", "analysis"]),
            ("结论", ["结论", "结语", "总结", "conclusion", "总结与展望", "future work"]),
            ("致谢", ["致谢", "acknowledgements", "acknowledgment"]),
            ("参考文献", ["参考文献", "references", "reference"]),
            ("附录", ["附录", "appendix"]),
            ("设计", ["设计", "design"]),
            ("实现", ["实现", "implementation"]),
            ("验证", ["验证", "verification", "validation"]),
            ("测试", ["测试", "test", "testing"]),
            ("对比", ["对比", "comparison"]),
            ("评估", ["评估", "evaluation"]),
            ("展望", ["展望", "未来工作", "future work"]),
        ]

        path_lower = section_path.lower()
        for zh_type, keywords in mapping:
            for kw in keywords:
                if kw in path_lower:
                    return zh_type
        return "正文"

    def _build_chunks_from_sections(self, sections: List[Section], filename: str) -> List[Chunk]:
        """递归遍历章节树，为每个文本块生成包含完整章节路径的 Chunk 对象"""
        final_chunks = []
        seq_num = 0

        def traverse(node: Section, path_titles: List[str]):
            nonlocal seq_num
            full_section_path = " > ".join(path_titles + [node.title]) if path_titles else node.title
            content = node.content.strip()

            if '参考文献' in node.title or 'references' in node.title.lower():
                refs = self._parse_references(content)
                if refs:
                    for ref in refs:
                        chunk_id = hashlib.md5(
                            f"{filename}_{full_section_path}_ref_{seq_num}".encode()
                        ).hexdigest()[:16]
                        final_chunks.append(Chunk(
                            id=chunk_id,
                            text=ref,
                            doc_name=filename,
                            page_num=node.page_num,
                            chunk_seq=seq_num + 1,
                            element_type="参考文献条目",
                            section_path=f"{full_section_path}",
                            strategy=ChunkStrategy.STRUCTURE,
                            metadata={'bbox': None, 'text_level': 0, 'is_reference': True}
                        ))
                        seq_num += 1
                    return

            if content:
                paragraphs = content.split('\n\n')
                for para in paragraphs:
                    para = para.strip()
                    if not para or len(para) < 10:
                        continue

                    is_table = '|' in para and para.count('|') > 2
                    para = UniversalTextCleaner.clean_text(para, is_table=is_table)

                    if len(para) > 1500:
                        sentences = para.replace('。', '。\n').split('\n')
                        for sent in sentences:
                            sent = sent.strip()
                            if sent:
                                sent += '。' if not sent.endswith('。') else ''
                                chunk_id = hashlib.md5(
                                    f"{filename}_{full_section_path}_{seq_num}".encode()
                                ).hexdigest()[:16]
                                final_chunks.append(Chunk(
                                    id=chunk_id,
                                    text=sent,
                                    doc_name=filename,
                                    page_num=node.page_num,
                                    chunk_seq=seq_num + 1,
                                    element_type=self._map_section_type(full_section_path),
                                    section_path=full_section_path,
                                    strategy=ChunkStrategy.STRUCTURE,
                                    metadata={'bbox': None, 'text_level': node.level, 'is_table': is_table}
                                ))
                                seq_num += 1
                    else:
                        chunk_id = hashlib.md5(
                            f"{filename}_{full_section_path}_{seq_num}".encode()
                        ).hexdigest()[:16]
                        final_chunks.append(Chunk(
                            id=chunk_id,
                            text=para,
                            doc_name=filename,
                            page_num=node.page_num,
                            chunk_seq=seq_num + 1,
                            element_type=self._map_section_type(full_section_path),
                            section_path=full_section_path,
                            strategy=ChunkStrategy.STRUCTURE,
                            metadata={'bbox': None, 'text_level': node.level, 'is_table': is_table}
                        ))
                        seq_num += 1

            for child in node.children:
                traverse(child, path_titles + [node.title])

        for sec in sections:
            traverse(sec, [])

        return final_chunks
