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
try:  # Prefer modern package to avoid deprecation
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:  # Fall back for environments not yet upgraded
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import DesignConceptConfig
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
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
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
    """Builds prompts aligned with the design brief requirements."""

    SYSTEM_PROMPT = (
        "你是一位经验丰富的建筑策划总师和建筑理论家。"  # noqa: E501
        "你的专长是分析建筑项目背景，并从过往的优秀案例中提炼出具有深度、创新性且可落地的设计理念。"
        "你的输出必须专业、逻辑严密，并具备建筑学术语言风格。"
    )

    JSON_SCHEMA = (
        "{{\n"
        "  \"analysis\": \"...\",\n"
        "  \"directions\": [\n"
        "    {{\n"
        "      \"title\": \"理念方向一：[四字或短语]\",\n"
        "      \"concept_description\": \"...\",\n"
        "      \"spatial_strategy\": \"...\",\n"
        "      \"inspiration_source\": \"...\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
    )

    def build_prompt(
        self,
        project_name: str,
        project_features: str,
        retrieved_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context = self._format_context(retrieved_docs)
        user_prompt = f"""# 任务背景\n"""
        user_prompt += f"**项目名称/类型**: {project_name}\n"
        user_prompt += f"**项目关键特征**: {project_features}\n\n"
        user_prompt += "# 参考上下文 (Retrieved Context)\n"
        user_prompt += "以下是从知识库中检索到的类似优秀项目的【设计理念】片段：\n---\n"
        user_prompt += f"{context}\n---\n\n"
        user_prompt += "# 生成任务\n"
        user_prompt += "请结合【项目关键特征】和【参考上下文】，为该项目生成一份“设计理念建议书”。\n"
        user_prompt += "请不要直接照抄参考文案，而是要分析这些案例背后的设计逻辑，并迁移到当前项目中。\n\n"
        user_prompt += "# 输出要求\n"
        user_prompt += "请严格按照以下 JSON 格式输出：\n"
        user_prompt += self.JSON_SCHEMA

        return {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": user_prompt,
        }

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "(未检索到上下文，参考语料为空)"

        formatted = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            name = meta.get("project_name", f"案例{idx}")
            src_type = meta.get("source_type", "unknown")
            area = meta.get("total_area") or meta.get("total_area_num")
            region = meta.get("region") or meta.get("city") or meta.get("country")
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            formatted.append(
                f"【案例{idx} | {name} | 来源:{src_type} | 区域:{region or '未知'} | 面积:{area or '未知'}】\n"
                f"{snippet}\n"
            )
        return "\n".join(formatted)


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

    def _ensure_llm(self):
        if self._llm_module is None:
            self._llm_module = GenerationIntegrationModule(
                provider=self.config.llm_provider,
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

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
        response = chain.invoke({})
        return {"prompt": prompt, "contexts": contexts, "response": response}
