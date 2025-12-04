"""Design Concept RAG pipeline utilities."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from config import DesignConceptConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


@dataclass
class DesignConceptDocument:
    """Lightweight container describing extracted concept text and metadata."""

    content: str
    metadata: Dict[str, object]

    def to_document(self) -> Document:
        return Document(page_content=self.content, metadata=self.metadata)


class DesignConceptETL:
    """Extracts semantic fields for design concepts from heterogeneous data sources."""

    TARGET_ZLJ_SECTIONS = {"### 展陈设计要求", "### 环境与景观"}

    def __init__(self, config: DesignConceptConfig):
        self.config = config
        self.documents: List[Document] = []

    def load_documents(self) -> List[Document]:
        docs: List[DesignConceptDocument] = []
        docs.extend(self._load_structured_json(self.config.china_data_path, source_type="china"))
        docs.extend(self._load_structured_json(self.config.world_data_path, source_type="world"))
        docs.extend(self._load_archdaily(self.config.archdaily_data_path))
        docs.extend(self._load_zlj_sections(self.config.knowledge_bwg_path))

        self.documents = [d.to_document() for d in docs if d.content.strip()]
        logger.info("设计理念ETL完成，共提取 %d 条语义单元", len(self.documents))
        return self.documents

    def _load_structured_json(self, folder: str, source_type: str) -> List[DesignConceptDocument]:
        path = Path(folder)
        if not path.exists():
            logger.warning("设计理念数据路径不存在: %s", folder)
            return []

        docs: List[DesignConceptDocument] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            concept = (data.get("concept&appearance") or data.get("concept") or "").strip()
            if not concept:
                continue

            project_name = data.get("name") or data.get("Project Title") or json_file.stem
            total_area = data.get("total_construction_area") or data.get("total_construction_area_sqm")
            height = data.get("building_height") or data.get("building_height_meters")

            metadata = {
                "source": str(json_file),
                "source_type": source_type,
                "project_name": project_name,
                "total_area": total_area,
                "total_area_num": self._parse_numeric(total_area),
                "building_height": height,
                "building_height_num": self._parse_numeric(height),
                "region": data.get("city") or data.get("Country") or data.get("country"),
                "project_type": data.get("Categories") or data.get("categories"),
            }
            docs.append(DesignConceptDocument(content=concept, metadata=metadata))

        logger.info("从%s提取理念 %d 条", folder, len(docs))
        return docs

    def _load_archdaily(self, folder: str) -> List[DesignConceptDocument]:
        path = Path(folder)
        if not path.exists():
            logger.warning("ArchDaily 数据路径不存在: %s", folder)
            return []

        docs: List[DesignConceptDocument] = []
        for json_file in path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as exc:
                logger.warning("读取%s失败: %s", json_file, exc)
                continue

            concept = (data.get("设计理念") or "").strip()
            if not concept:
                desc = data.get("Description")
                if isinstance(desc, list) and desc:
                    concept = "\n".join(desc[:2]).strip()

            if not concept:
                continue

            total_area = data.get("Area")
            metadata = {
                "source": str(json_file),
                "source_type": "archdaily",
                "project_name": data.get("Project Title", json_file.stem),
                "architects": data.get("Architects"),
                "categories": data.get("Categories"),
                "city": data.get("City"),
                "country": data.get("Country"),
                "project_type": data.get("Categories"),
                "total_area": total_area,
                "total_area_num": self._parse_numeric(total_area),
            }
            docs.append(DesignConceptDocument(content=concept, metadata=metadata))

        logger.info("从ArchDaily提取理念 %d 条", len(docs))
        return docs

    def _load_zlj_sections(self, file_path: str) -> List[DesignConceptDocument]:
        path = Path(file_path)
        if not path.exists():
            logger.warning("知识库文件不存在: %s", file_path)
            return []

        text = path.read_text(encoding="utf-8")
        section_text = self._extract_section(text, "## 布局与要求")
        if not section_text:
            return []

        docs: List[DesignConceptDocument] = []
        for header, body in self._split_subsections(section_text):
            if header not in self.TARGET_ZLJ_SECTIONS:
                continue
            cleaned = body.strip()
            if not cleaned:
                continue
            metadata = {
                "source": str(path),
                "source_type": "knowledge_base",
                "project_name": "博物馆知识库",
                "section": header.replace("#", "").strip(),
            }
            content = f"{header}\n{cleaned}"
            docs.append(DesignConceptDocument(content=content, metadata=metadata))

        logger.info("从%s提取知识段落 %d 条", file_path, len(docs))
        return docs

    @staticmethod
    def _extract_section(text: str, header: str) -> str:
        pattern = re.compile(rf"^{re.escape(header)}\s*$", re.MULTILINE)
        match = pattern.search(text)
        if not match:
            return ""
        start = match.end()
        next_header = re.compile(r"^##\s+.+", re.MULTILINE)
        next_match = next_header.search(text, start)
        end = next_match.start() if next_match else len(text)
        return text[start:end]

    @staticmethod
    def _split_subsections(section_text: str) -> List[tuple[str, str]]:
        chunks = re.split(r"(^###\s+.+$)", section_text, flags=re.MULTILINE)
        pairs: List[tuple[str, str]] = []
        current_header: Optional[str] = None
        for chunk in chunks:
            if not chunk:
                continue
            if chunk.startswith("###"):
                current_header = chunk.strip()
                continue
            if current_header:
                pairs.append((current_header, chunk))
        return pairs

    @staticmethod
    def _parse_numeric(value: Optional[object]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[\d.]+", value.replace(",", ""))
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None


class DesignConceptVectorStore:
    """FAISS-backed semantic index for design concept documents."""

    def __init__(self, config: DesignConceptConfig):
        self.config = config
        self.embedding = get_embedding(model_name=config.embedding_model)
        self.vectorstore: Optional[FAISS] = None

    def load(self) -> bool:
        index_dir = Path(self.config.index_save_path)
        if not index_dir.exists():
            return False
        try:
            self.vectorstore = FAISS.load_local(
                str(index_dir),
                self.embedding,
                allow_dangerous_deserialization=True
            )
            logger.info("已加载设计理念向量索引: %s", index_dir)
            return True
        except Exception as exc:
            logger.warning("加载设计理念向量索引失败，将重新构建: %s", exc)
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("没有可用于构建索引的设计理念文档")

        index_dir = Path(self.config.index_save_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        logger.info("正在构建设计理念向量索引（文档=%d）...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(str(index_dir))
        logger.info("设计理念索引保存至: %s", index_dir)

    def ensure_ready(self, documents_loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        documents = documents_loader()
        self.build(documents)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("向量索引尚未加载")
        final_k = top_k if top_k and top_k > 0 else 4
        search_k = max(final_k, 8)
        raw_results = self.vectorstore.similarity_search(query, k=search_k)
        if not filters:
            return raw_results[:final_k]
        filtered = [doc for doc in raw_results if self._match_filters(doc, filters)]
        return filtered[:final_k] if filtered else raw_results[:final_k]

    @staticmethod
    def _match_filters(doc: Document, filters: Dict[str, object]) -> bool:
        meta = doc.metadata or {}
        area = meta.get("total_area_num")
        height = meta.get("building_height_num")

        min_area = filters.get("min_area")
        max_area = filters.get("max_area")
        min_height = filters.get("min_height")
        max_height = filters.get("max_height")
        category = filters.get("category")

        if min_area is not None and (area is None or area < float(min_area)):
            return False
        if max_area is not None and (area is not None and area > float(max_area)):
            return False
        if min_height is not None and (height is None or height < float(min_height)):
            return False
        if max_height is not None and (height is not None and height > float(max_height)):
            return False
        if category is not None:
            categories = meta.get("project_type") or meta.get("categories") or []
            if isinstance(categories, str):
                categories = [c.strip() for c in categories.split(",")]
            if isinstance(categories, list):
                normalized = {c.lower() for c in categories}
                if category.lower() not in normalized:
                    return False
        return True


class DesignConceptPromptBuilder:
    """Builds structured prompts for the design concept report."""

    SYSTEM_PROMPT_TEMPLATE = (
        "你是一个专业的建筑策划顾问。你的任务是为《{project_name} 设计理念策划专报》提供内容。"  # noqa: E501
        "该项目的核心特征是：{project_features}。\n\n"
        "请基于以下参考案例（Context）以及你掌握的建筑理论，从四个维度展开：\n"
        "{context_block}\n\n"
        "### 1. 理念溯源 (Origins & Archetypes)\n"
        "- 归纳科技馆建筑常见理念来源，并结合本项目特征锁定最适切的理论源头。\n\n"
        "### 2. 相似案例理念参照 (Case Benchmarking)\n"
        "- 从 Context 中挑选 2-3 个最相关案例，逐一说明其理念生成逻辑与对本项目的启示，必须显式引用案例名称。\n\n"
        "### 3. 设计趋势研判 (Future Trends)\n"
        "- 分析 2015 年以来科技馆理念演变趋势，若 Context 偏旧，可结合行业共识补充最新洞察。\n\n"
        "### 4. 本项目概念生成 (Concept Generation)\n"
        "- 生成 2 个理念方案，并包含“核心隐喻 / 造型意向 / 空间氛围”。\n\n"
        "输出格式：必须使用 Markdown，结构清晰，便于直接发送给甲方或设计团队。"
    )

    USER_PROMPT_TEMPLATE = (
        "请按照系统指令，撰写《{project_name} 设计理念策划专报》。"
    )

    def build_prompt(
        self,
        project_name: str,
        project_features: str,
        retrieved_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context_block = self._format_context(retrieved_docs)
        system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            project_name=project_name,
            project_features=project_features or "（未提供特征）",
            context_block=context_block,
        )
        user_prompt = self.USER_PROMPT_TEMPLATE.format(project_name=project_name)
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _format_context(self, docs: Sequence[Document]) -> str:
        if not docs:
            return "Context:\n（暂无检索案例，可结合行业知识自行补充对标与趋势。）"

        formatted: List[str] = []
        for idx, doc in enumerate(docs, 1):
            formatted.append(self._format_single_context(idx, doc))
        return "Context:\n" + "\n".join(formatted)

    def _format_single_context(self, idx: int, doc: Document) -> str:
        meta = doc.metadata or {}
        name = str(meta.get("project_name") or f"案例{idx}")
        year = self._extract_year(meta)
        location = meta.get("region") or meta.get("city") or meta.get("country") or "未知地区"
        typology = meta.get("project_type") or meta.get("categories") or "类型未明"
        source_label = meta.get("source_type") or "unknown"
        snippet = doc.page_content.strip()
        snippet = snippet.replace("\n", " ")
        snippet = snippet[:600] + "..." if len(snippet) > 600 else snippet
        return (
            f"[{year}] {name}（{location} | {typology} | 来源:{source_label}）\n"
            f"{snippet}\n"
            "—— 请引用此案例名称进行对标"
        )

    @staticmethod
    def _extract_year(metadata: Dict[str, object]) -> str:
        year_fields = (
            "year",
            "Year",
            "completion_year",
            "completionYear",
            "year_completed",
            "built_year",
            "建成时间",
            "建成年份",
        )
        for field in year_fields:
            value = metadata.get(field)
            if not value:
                continue
            if isinstance(value, (int, float)):
                year = int(value)
                if 1900 <= year <= 2100:
                    return str(year)
            if isinstance(value, str):
                match = re.search(r"(19|20)\d{2}", value)
                if match:
                    return match.group(0)
        return "未知年份"


class DesignConceptGenerator:
    """High-level facade that orchestrates ETL, vector search, and prompt-driven generation."""

    def __init__(self, config: DesignConceptConfig):
        self.config = config
        self.etl = DesignConceptETL(config)
        self.vector_store = DesignConceptVectorStore(config)
        self.prompt_builder = DesignConceptPromptBuilder()
        self._llm_module: Optional[GenerationIntegrationModule] = None

    def ensure_index(self, rebuild: bool = False) -> None:
        self.vector_store.ensure_ready(self.etl.load_documents, rebuild=rebuild)

    def _ensure_llm(self) -> None:
        desired_temperature = self._resolve_creative_temperature()
        if (
            self._llm_module is not None
            and abs(self._llm_module.temperature - desired_temperature) < 1e-6
        ):
            return
        self._llm_module = GenerationIntegrationModule(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            temperature=desired_temperature,
            max_tokens=self.config.max_tokens,
        )

    def _resolve_creative_temperature(self) -> float:
        base_temp = self.config.temperature if self.config.temperature is not None else 0.4
        return max(0.3, min(0.5, base_temp))

    def retrieve_contexts(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, object]] = None,
    ) -> List[Document]:
        if self.vector_store.vectorstore is None:
            raise RuntimeError("请先构建或加载设计理念向量索引")
        return self.vector_store.search(query, top_k or self.config.top_k, filters=filters)

    def generate(
        self,
        project_name: str,
        project_features: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, object]] = None,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        if not project_name:
            raise ValueError("project_name 不能为空")
        search_query = query or project_features or project_name
        contexts = self.retrieve_contexts(search_query, top_k=top_k, filters=filters)
        prompt = self.prompt_builder.build_prompt(project_name, project_features, contexts)

        if dry_run:
            return {"prompt": prompt, "contexts": contexts}

        self._ensure_llm()
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        # LangChain 保证这里返回 str，但为安全起见做一次显式转换，
        # 避免上游节点在判断 truthy 值时受到类型或 None 的影响。
        raw = chain.invoke({})
        response = "" if raw is None else str(raw)
        return {"prompt": prompt, "contexts": contexts, "response": response}
