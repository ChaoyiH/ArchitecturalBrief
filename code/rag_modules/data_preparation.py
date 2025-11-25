"""数据准备模块"""

import json
import logging
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
import uuid

from config import ExhibitionConfig, PublicServiceConfig, BusinessResearchConfig

logger = logging.getLogger(__name__)

class DataPreparationModule:
    """数据准备模块 - 负责数据加载、清洗和预处理"""
    
    def __init__(self, data_paths: list):
        """
        初始化数据准备模块
        
        Args:
            data_paths: 数据文件夹路径列表
        """
        self.data_paths = data_paths if isinstance(data_paths, list) else [data_paths]
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []     # 子文档（按标题分割的小块）
        self.parent_child_map: Dict[str, str] = {}  # 子块ID -> 父文档ID的映射
    
    def load_documents(self) -> List[Document]:
        """
        加载文档数据
        
        Returns:
            加载的文档列表
        """
        logger.info(f"正在从 {self.data_paths} 加载文档...")
        
        # 直接读取Markdown文件以保持原始格式
        documents = []
        
        # 遍历所有数据路径
        for data_path in self.data_paths:
            data_path_obj = Path(data_path)
            if not data_path_obj.exists():
                logger.warning(f"数据路径不存在: {data_path}")
                continue
                
            logger.info(f"正在读取: {data_path}")
            for md_file in data_path_obj.rglob("*.md"):
                try:
                    # 直接读取文件内容，保持Markdown格式
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 为每个父文档分配确定性的唯一ID（基于相对路径）
                    try:
                        data_root = Path(data_path).resolve()
                        relative_path = Path(md_file).resolve().relative_to(data_root).as_posix()
                    except Exception:
                        relative_path = Path(md_file).as_posix()
                    parent_id = hashlib.md5(relative_path.encode("utf-8")).hexdigest()

                    # 创建Document对象
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(md_file),
                            "parent_id": parent_id,
                            "doc_type": "parent",  # 标记为父文档
                            "doc_name": md_file.stem
                        }
                    )
                    documents.append(doc)

                except Exception as e:
                    logger.warning(f"读取文件 {md_file} 失败: {e}")
        
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个文档")
        return documents
    
    def chunk_documents(self) -> List[Document]:
        """
        Markdown结构感知分块

        Returns:
            分块后的文档列表
        """
        logger.info("正在进行Markdown结构感知分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        # 使用Markdown标题分割器
        chunks = self._markdown_header_split()

        # 为每个chunk添加基础元数据
        for i, chunk in enumerate(chunks):
            if 'chunk_id' not in chunk.metadata:
                # 如果没有chunk_id（比如分割失败的情况），则生成一个
                chunk.metadata['chunk_id'] = str(uuid.uuid4())
            chunk.metadata['batch_index'] = i  # 在当前批次中的索引
            chunk.metadata['chunk_size'] = len(chunk.page_content)

        self.chunks = chunks
        logger.info(f"Markdown分块完成，共生成 {len(chunks)} 个chunk")
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        使用Markdown标题分割器进行结构化分割

        Returns:
            按标题结构分割的文档列表
        """
        # 定义要分割的标题层级
        headers_to_split_on = [
            ("#", "主标题"),      # 标准/规范等名称
            ("##", "二级标题"),   # 二级类目
            ("###", "三级标题"),  # 三级类目
            ("####", "四级标题")  # 四级类目
        ]

        # 创建Markdown分割器
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # 保留标题，便于理解上下文
        )

        all_chunks = []

        for doc in self.documents:
            try:
                # 检查文档内容是否包含Markdown标题
                content_preview = doc.page_content[:200]
                has_headers = any(line.strip().startswith('#') for line in content_preview.split('\n'))

                if not has_headers:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 内容中没有发现Markdown标题")
                    logger.debug(f"内容预览: {content_preview}")

                # 对每个文档进行Markdown分割
                md_chunks = markdown_splitter.split_text(doc.page_content)

                logger.debug(f"文档 {doc.metadata.get('dish_name', '未知')} 分割成 {len(md_chunks)} 个chunk")

                # 如果没有分割成功，说明文档可能没有标题结构
                if len(md_chunks) <= 1:
                    logger.warning(f"文档 {doc.metadata.get('dish_name', '未知')} 未能按标题分割，可能缺少标题结构")

                # 为每个子块建立与父文档的关系
                parent_id = doc.metadata["parent_id"]
                doc_name = doc.metadata.get("doc_name", "未知文档")

                for i, chunk in enumerate(md_chunks):
                    # 为子块分配唯一ID
                    child_id = str(uuid.uuid4())

                    # 合并原文档元数据和新的标题元数据
                    chunk.metadata.update(doc.metadata)
                    chunk.metadata.update({
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "doc_type": "child",  # 标记为子文档
                        "chunk_index": i,      # 在父文档中的位置
                        "doc_name": doc_name
                    })

                    # 建立父子映射关系
                    self.parent_child_map[child_id] = parent_id

                all_chunks.extend(md_chunks)

            except Exception as e:
                logger.warning(f"文档 {doc.metadata.get('source', '未知')} Markdown分割失败: {e}")
                # 如果Markdown分割失败，将整个文档作为一个chunk
                all_chunks.append(doc)

        logger.info(f"Markdown结构分割完成，生成 {len(all_chunks)} 个结构化块")
        return all_chunks
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            统计信息字典
        """
        if not self.documents:
            return {}

        doc_names = {}
        for doc in self.documents:
            doc_name = doc.metadata.get('doc_name', '未知')
            doc_names[doc_name] = doc_names.get(doc_name, 0) + 1

        return {
            'total_documents': len(self.documents),
            'total_chunks': len(self.chunks),
            'doc_count': len(doc_names),
            'avg_chunk_size': sum(chunk.metadata.get('chunk_size', 0) for chunk in self.chunks) / len(self.chunks) if self.chunks else 0
        }
    
    def export_metadata(self, output_path: str):
        """
        导出元数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        metadata_list = []
        for doc in self.documents:
            metadata_list.append({
                'source': doc.metadata.get('source'),
                'doc_name': doc.metadata.get('doc_name'),
                'content_length': len(doc.page_content)
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块获取对应的父文档（智能去重）

        Args:
            child_chunks: 检索到的子块列表

        Returns:
            对应的父文档列表（去重，按相关性排序）
        """
        # 统计每个父文档被匹配的次数（相关性指标）
        parent_relevance = {}
        parent_docs_map = {}

        # 收集所有相关的父文档ID和相关性分数
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                # 增加相关性计数
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

                # 缓存父文档（避免重复查找）
                if parent_id not in parent_docs_map:
                    for doc in self.documents:
                        if doc.metadata.get("parent_id") == parent_id:
                            parent_docs_map[parent_id] = doc
                            break

        # 按相关性排序（匹配次数多的排在前面）
        sorted_parent_ids = sorted(parent_relevance.keys(),
                                 key=lambda x: parent_relevance[x],
                                 reverse=True)

        # 构建去重后的父文档列表
        parent_docs = []
        for parent_id in sorted_parent_ids:
            if parent_id in parent_docs_map:
                parent_docs.append(parent_docs_map[parent_id])

        # 收集父文档名称和相关性信息用于日志
        parent_info = []
        for doc in parent_docs:
            doc_name = doc.metadata.get('doc_name', '未知文档')
            parent_id = doc.metadata.get('parent_id')
            relevance_count = parent_relevance.get(parent_id, 0)
            parent_info.append(f"{doc_name}({relevance_count}块)")

        logger.info(f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}")
        return parent_docs


class ExhibitionDataExtractor:
    """基于语义字段提取展览空间相关片段的处理器 (KnowledgeBaseProcessor)。"""

    GB_KEYWORDS = ("展厅", "陈列", "净高", "柱网", "荷载")
    ZLJ_SECTION_PATTERNS = ("陈列展览区", "陈列", "布局")

    def __init__(self, config: ExhibitionConfig):
        self.config = config

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []
        documents.extend(self._load_structured_json(self.config.china_data_path, source_type="china"))
        documents.extend(self._load_structured_json(self.config.world_data_path, source_type="world"))
        documents.extend(self._load_archdaily(self.config.archdaily_data_path))
        documents.extend(self._load_gb_markdown(self.config.gb_data_path))
        documents.extend(self._load_zlj_markdown(self.config.zlj_data_path))
        logger.info("展览空间数据抽取完成，共 %d 条", len(documents))
        return documents

    def _load_structured_json(self, folder: str, source_type: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            logger.warning("展览空间数据路径不存在: %s", folder)
            return []

        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            payload = self._collect_json_fields(data)
            if not payload:
                continue

            project_name = data.get("name") or data.get("Project Title") or json_file.stem
            total_area = data.get("total_construction_area") or data.get("total_construction_area_sqm")
            central_hall = data.get("central_hall")

            content_lines = [f"项目: {project_name}"]
            if total_area:
                content_lines.append(f"总建筑面积: {total_area}")
            if central_hall:
                content_lines.append(f"中央大厅: {central_hall}")
            content_lines.extend(payload)

            docs.append(Document(
                page_content="\n".join(content_lines),
                metadata={
                    "source": str(json_file),
                    "source_type": source_type,
                    "project_name": project_name,
                    "total_area": total_area,
                    "section": "permanent_exhibitions",
                }
            ))

        return docs

    def _collect_json_fields(self, data: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        field_map = {
            "permanent_exhibitions": "常设展览",
            "central_hall": "中央大厅",
            "total_construction_area": "总建筑面积",
            "陈列展览区": "陈列展览区",
        }
        for key, label in field_map.items():
            value = data.get(key)
            if isinstance(value, list):
                value = "\n".join(value)
            if value:
                values.append(f"{label}: {value}".strip())
        return values

    def _load_archdaily(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            content = data.get("陈列展览区") or data.get("Description")
            if isinstance(content, list):
                content = "\n".join(content[:3])
            if not content:
                continue
            project_name = data.get("Project Title", json_file.stem)
            docs.append(Document(
                page_content=f"项目: {project_name}\n{content}",
                metadata={
                    "source": str(json_file),
                    "source_type": "archdaily",
                    "project_name": project_name,
                }
            ))
        return docs

    def _load_gb_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue
            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                if any(keyword in para for keyword in self.GB_KEYWORDS):
                    docs.append(Document(
                        page_content=para.strip(),
                        metadata={
                            "source": str(md_file),
                            "source_type": "gb_standard",
                            "doc_name": md_file.stem,
                            "section": "GB 展陈条文",
                        }
                    ))
        return docs

    def _load_zlj_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue

            for section_header, section_body in self._iter_sections(text):
                docs.append(Document(
                    page_content=f"{section_header}\n{section_body.strip()}",
                    metadata={
                        "source": str(md_file),
                        "source_type": "zlj",  # 资料集
                        "section": section_header.strip('# ').strip(),
                        "doc_name": md_file.stem,
                    }
                ))
        return docs

    def _iter_sections(self, text: str):
        pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        for idx, match in enumerate(matches):
            header = match.group(0)
            title = match.group(1)
            if not any(token in title for token in self.ZLJ_SECTION_PATTERNS):
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            yield header, text[start:end]


class PublicServiceDataExtractor:
    """公共服务区数据抽取器。"""

    JSON_DESCRIPTION_KEYWORDS = ("lobby", "foyer", "shop", "cafe", "restaurant", "entrance")
    MD_KEYWORDS = ("门厅", "休息", "厕所", "卫生间", "餐饮", "商店", "无障碍", "寄存", "母婴")
    SPECIAL_GB_KEYWORDS = ("卫生设施", "卫生间", "厕位", "第三卫生间")

    def __init__(self, config: PublicServiceConfig):
        self.config = config

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []
        documents.extend(self._load_structured_json(self.config.china_data_path, source_type="china"))
        documents.extend(self._load_structured_json(self.config.world_data_path, source_type="world"))
        documents.extend(self._load_archdaily(self.config.archdaily_data_path))
        documents.extend(self._load_gb_markdown(self.config.gb_data_path))
        documents.extend(self._load_zlj_markdown(self.config.zlj_data_path))
        logger.info("公共服务区数据抽取完成，共 %d 条", len(documents))
        return documents

    def _load_structured_json(self, folder: str, source_type: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            logger.warning("公共服务区数据路径不存在: %s", folder)
            return []

        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            payload = self._collect_public_sections(data)
            desc_snippets = self._extract_description_snippets(data.get("Description"))
            if not payload and not desc_snippets:
                continue

            project_name = data.get("name") or data.get("Project Title") or json_file.stem
            total_area = data.get("total_construction_area") or data.get("total_construction_area_sqm")

            content_lines = [f"项目: {project_name}"]
            if total_area:
                content_lines.append(f"总建筑面积: {total_area}")
            content_lines.extend(payload)
            if desc_snippets:
                content_lines.append("公共服务描述片段:")
                content_lines.extend(desc_snippets)

            docs.append(Document(
                page_content="\n".join(content_lines),
                metadata={
                    "source": str(json_file),
                    "source_type": source_type,
                    "project_name": project_name,
                    "section": "public_service",
                }
            ))

        return docs

    def _collect_public_sections(self, data: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        field_map = {
            "公共服务区": "公共服务区",
            "central_hall": "中央大厅",
            "lobby": "门厅",
            "public_service": "公共服务",
        }
        for key, label in field_map.items():
            value = data.get(key)
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            if value:
                values.append(f"{label}: {value}".strip())
        return values

    def _extract_description_snippets(self, description_field: Any) -> List[str]:
        if not description_field:
            return []
        if isinstance(description_field, list):
            paragraphs = description_field
        else:
            paragraphs = re.split(r"\n\s*\n", str(description_field))

        snippets: List[str] = []
        for para in paragraphs:
            lower_para = para.lower()
            if any(keyword in lower_para for keyword in self.JSON_DESCRIPTION_KEYWORDS):
                snippets.append(para.strip())
        return snippets

    def _load_archdaily(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            content_parts: List[str] = []
            public_service = data.get("公共服务区")
            if isinstance(public_service, list):
                public_service = "\n".join(public_service)
            if public_service:
                content_parts.append(str(public_service))

            desc_snippets = self._extract_description_snippets(data.get("Description"))
            content_parts.extend(desc_snippets)
            if not content_parts:
                continue

            project_name = data.get("Project Title", json_file.stem)
            docs.append(Document(
                page_content=f"项目: {project_name}\n" + "\n".join(content_parts),
                metadata={
                    "source": str(json_file),
                    "source_type": "archdaily",
                    "project_name": project_name,
                }
            ))
        return docs

    def _load_gb_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue

            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                cleaned = para.strip()
                if not cleaned:
                    continue
                if any(keyword in cleaned for keyword in self.MD_KEYWORDS):
                    docs.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source": str(md_file),
                            "source_type": "gb_standard",
                            "doc_name": md_file.stem,
                            "section": "GB 公共服务",
                        }
                    ))
                elif "博物馆建筑设计规范" in md_file.stem and any(keyword in cleaned for keyword in self.SPECIAL_GB_KEYWORDS):
                    docs.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source": str(md_file),
                            "source_type": "gb_standard",
                            "doc_name": md_file.stem,
                            "section": "GB 卫生设施",
                        }
                    ))
        return docs

    def _load_zlj_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue

            for header, body in self._iter_public_sections(text):
                docs.append(Document(
                    page_content=f"{header}\n{body.strip()}",
                    metadata={
                        "source": str(md_file),
                        "source_type": "zlj",
                        "section": header.strip('# ').strip(),
                        "doc_name": md_file.stem,
                    }
                ))
        return docs

    def _iter_public_sections(self, text: str):
        pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        for idx, match in enumerate(matches):
            header = match.group(0)
            title = match.group(1)
            if not any(keyword in title for keyword in self.MD_KEYWORDS):
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            yield header, text[start:end]


class BusinessResearchDataExtractor:
    """业务科研区数据抽取器。"""

    JSON_DESC_KEYWORDS = (
        "office",
        "administration",
        "admin",
        "research",
        "laboratory",
        "laboratories",
        "conservation",
        "curator",
        "restoration",
    )
    STRUCTURE_TEXT_KEYWORDS = (
        "业务", "科研", "研究", "行政", "办公", "藏品", "库前", "修复", "实验", "laboratory", "office", "admin"
    )
    ZLJ_KEYWORDS = ("业务", "行政", "技术", "库前", "修复", "科研")

    def __init__(self, config: BusinessResearchConfig):
        self.config = config

    def load_documents(self) -> List[Document]:
        documents: List[Document] = []
        documents.extend(self._load_structured_json(self.config.china_data_path, source_type="china"))
        documents.extend(self._load_structured_json(self.config.world_data_path, source_type="world"))
        documents.extend(self._load_archdaily(self.config.archdaily_data_path))
        documents.extend(self._load_gb_markdown(self.config.gb_data_path))
        documents.extend(self._load_zlj_markdown(self.config.zlj_data_path))
        logger.info("业务科研区数据抽取完成，共 %d 条", len(documents))
        return documents

    def _load_structured_json(self, folder: str, source_type: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            logger.warning("业务科研区数据路径不存在: %s", folder)
            return []

        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            payload = self._collect_structured_sections(data)
            desc_snippets = self._extract_description_snippets(data.get("Description"))
            if not payload and not desc_snippets:
                continue

            project_name = data.get("name") or data.get("Project Title") or json_file.stem
            total_area = data.get("total_construction_area") or data.get("total_construction_area_sqm")
            content_lines = [f"项目: {project_name}"]
            if total_area:
                content_lines.append(f"总建筑面积: {total_area}")
            content_lines.extend(payload)
            if desc_snippets:
                content_lines.append("业务科研相关描述:")
                content_lines.extend(desc_snippets)

            docs.append(Document(
                page_content="\n".join(content_lines),
                metadata={
                    "source": str(json_file),
                    "source_type": source_type,
                    "project_name": project_name,
                    "section": "business_research",
                }
            ))
        return docs

    def _collect_structured_sections(self, data: Dict[str, Any]) -> List[str]:
        values: List[str] = []
        field_map = {
            "业务科研用房": "业务科研用房",
            "business_research": "业务科研",
            "administration_zone": "行政管理区",
            "collection_management": "藏品管理区",
            "库前区": "库前区",
            "research_facilities": "科研设施",
            "office_area": "办公区域",
        }
        for key, label in field_map.items():
            value = data.get(key)
            if isinstance(value, list):
                value = "\n".join(str(v) for v in value)
            if value:
                values.append(f"{label}: {value}".strip())

        for key, value in data.items():
            if not isinstance(value, str):
                continue
            if any(token in value for token in self.STRUCTURE_TEXT_KEYWORDS):
                values.append(f"{key}: {value}".strip())
        return list(dict.fromkeys(values))

    def _extract_description_snippets(self, description_field: Any) -> List[str]:
        if not description_field:
            return []
        if isinstance(description_field, list):
            paragraphs = description_field
        else:
            paragraphs = re.split(r"\n\s*\n", str(description_field))

        snippets: List[str] = []
        for para in paragraphs:
            lower_para = para.lower()
            if any(keyword in lower_para for keyword in self.JSON_DESC_KEYWORDS):
                snippets.append(para.strip())
        return snippets

    def _load_archdaily(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            content_parts: List[str] = []
            business_field = data.get("业务科研用房")
            if isinstance(business_field, list):
                business_field = "\n".join(business_field)
            if business_field:
                content_parts.append(str(business_field))

            desc_snippets = self._extract_description_snippets(data.get("Description"))
            content_parts.extend(desc_snippets)
            if not content_parts:
                continue

            project_name = data.get("Project Title", json_file.stem)
            docs.append(Document(
                page_content=f"项目: {project_name}\n" + "\n".join(content_parts),
                metadata={
                    "source": str(json_file),
                    "source_type": "archdaily",
                    "project_name": project_name,
                }
            ))
        return docs

    def _load_gb_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.rglob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue

            if "博物馆建筑设计规范" in md_file.stem:
                for section in ("4.4", "4.5", "4.6"):
                    extracted = self._extract_section(text, section)
                    if extracted:
                        docs.append(Document(
                            page_content=extracted.strip(),
                            metadata={
                                "source": str(md_file),
                                "source_type": "gb_standard",
                                "doc_name": md_file.stem,
                                "section": f"章节 {section}",
                            }
                        ))
                continue

            paragraphs = re.split(r"\n\s*\n", text)
            for para in paragraphs:
                cleaned = para.strip()
                if not cleaned:
                    continue
                if any(keyword in cleaned for keyword in self.STRUCTURE_TEXT_KEYWORDS):
                    docs.append(Document(
                        page_content=cleaned,
                        metadata={
                            "source": str(md_file),
                            "source_type": "gb_standard",
                            "doc_name": md_file.stem,
                            "section": "业务科研条文",
                        }
                    ))
        return docs

    def _extract_section(self, text: str, section_prefix: str) -> Optional[str]:
        pattern = re.compile(rf"(^###\s+{re.escape(section_prefix)}[\s\S]+?)(?=^###\s+|\Z)", re.MULTILINE)
        match = pattern.search(text)
        if match:
            return match.group(1)
        return None

    def _load_zlj_markdown(self, folder: str) -> List[Document]:
        path = Path(folder)
        if not path.exists():
            return []
        docs: List[Document] = []
        for md_file in path.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning("读取%s失败: %s", md_file, exc)
                continue

            if md_file.stem.lower() == "bwg":
                for header, body in self._iter_keyword_sections(text):
                    docs.append(Document(
                        page_content=f"{header}\n{body.strip()}",
                        metadata={
                            "source": str(md_file),
                            "source_type": "zlj",
                            "section": header.strip('# ').strip(),
                            "doc_name": md_file.stem,
                        }
                    ))
            else:
                paragraphs = re.split(r"\n\s*\n", text)
                for para in paragraphs:
                    cleaned = para.strip()
                    if cleaned and any(keyword in cleaned for keyword in self.ZLJ_KEYWORDS):
                        docs.append(Document(
                            page_content=cleaned,
                            metadata={
                                "source": str(md_file),
                                "source_type": "zlj",
                                "doc_name": md_file.stem,
                                "section": "业务科研描述",
                            }
                        ))
        return docs

    def _iter_keyword_sections(self, text: str):
        pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        for idx, match in enumerate(matches):
            header = match.group(0)
            title = match.group(1)
            if not any(keyword in title for keyword in self.ZLJ_KEYWORDS):
                continue
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            yield header, text[start:end]
