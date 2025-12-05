"""Design Concept RAG pipeline utilities."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
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
        final_k = top_k if top_k and top_k > 0 else 12
        search_k = max(final_k, 16)
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
    """Builds structured prompts for the design concept JSON workflow."""

    
    def build_concept_sourcing_prompt(
        self,
        project_name: str,
        project_features: str,
        retrieved_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context_block = self._format_context(retrieved_docs)
        context_block_safe = context_block.replace("{", "{{").replace("}", "}}")
        system_prompt = (
            "你是一名熟悉中国《科学技术馆建设标准》《博物馆建筑设计规范》的建筑策划顾问。\n"
            "目标：为项目“{project_name}”提炼设计依据，严格区分【规范事实】与【分析推演】。\n"
            "请优先引用 Context 中的 GB/规范文本（标注为 [Standard Mandate]），并确保每条均为原文摘录。"
        ).format(
            project_name=project_name,
            project_features=project_features or "（未提供特征）",
        )
        user_prompt = (
            "从下方 Context 中检索规范与案例，生成 JSON（必须可 json.loads）：\n"
            "{{{{\n"
            "  \"standard_mandates\": [\n"
            "    {{{{\"source\": \"规范名称\", \"clause\": \"条款编号/标题\", \"content\": \"原文摘录\"}}}}\n"
            "  ],\n"
            "  \"theoretical_basis\": \"基于建筑类型学/叙事/场所精神的分析（若缺失写 Not Specified）\"\n"
            "}}}}\n\n"
            "要求：\n"
            "- 仅输出 JSON，不要解释；\n"
            "- standard_mandates 至少2条，若缺失填 Not Specified；\n"
            "- Content 必须是 Context 原文片段（50-120字）。\n\n"
            "Context:\n{context_block}".format(context_block=context_block_safe)
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def build_benchmarking_prompt(
        self,
        project_name: str,
        project_features: str,
        benchmark_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context_block = self._format_context(benchmark_docs)
        context_block_safe = context_block.replace("{", "{{").replace("}", "}}")
        system_prompt = (
            "You are an architectural historian conducting a forensic analysis. Do NOT summarize.\n"
            "Project: {project_name}. Features: {project_features}.\n"
            "From Context, extract **facts** and structured details. If a field is missing, write 'Not Specified'."
        ).format(
            project_name=project_name,
            project_features=project_features or "(not provided)",
        )
        user_prompt = (
            "Output JSON array (len=3) strictly: \n"
            "[\n  {{{{\n    \"case_name\": \"...\",\n    \"metadata\": {{{{\"architect\": \"...\", \"year\": \"...\", \"location\": \"城市, 国家\"}}}},\n    \"similarity_logic\": \"为何相似，规模/气候/区位\",\n    \"design_details\": {{{{\n      \"form_logic\": [\"至少3条：隐喻/几何/体量关系\"],\n      \"materiality\": [\"至少2条：幕墙/结构/生态技术\"],\n      \"spatial_features\": [\"至少3条：中庭/流线/界面\"]\n    }}}},\n    \"evidence_snippet\": \"引用 Context 中的原文句子\"\n  }}}}\n]\n\n"
            "Instructions:\n"
            "1. Identify Architect & Year explicitly (Fact).\n"
            "2. form_logic: metaphors/geometry/volume; materiality: facade+structure+ecotech; spatial_features: atrium/circulation/openings.\n"
            "3. evidence_snippet must be a quoted sentence from Context.\n"
            "4. Use bullet-like short items, no long paragraphs.\n\n"
            "Context:\n{context_block}".format(context_block=context_block_safe)
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def build_trend_prompt(
        self,
        project_name: str,
        project_features: str,
        trend_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context_block = self._format_context(trend_docs)
        context_block_safe = context_block.replace("{", "{{").replace("}", "}}")
        system_prompt = (
            "你是一名关注科普建筑前沿趋势的策划专家。\n"
            "请输出结构化 JSON，显式区分证据与分析。"
        )
        user_prompt = (
            "输出 JSON：\n"
            "{{{{\n"
            "  \"trend_list\": [\n"
            "    {{{{\n"
            "      \"trend_name\": \"...\",\n"
            "      \"description\": \"50-100字\",\n"
            "      \"evidence_cases\": [\"案例A (年份)\", \"案例B (年份)\"],\n"
            "      \"evidence_snippet\": \"Context 原文摘录\"\n"
            "    }}}}\n"
            "  ],\n"
            "  \"innovative_suggestions\": [\"建议1（含参考依据）\", \"建议2\"]\n"
            "}}}}\n\n"
            "要求：\n"
            "- 若找不到年份或案例名称，填 Not Specified；\n"
            "- evidence_snippet 必须引用 Context 原文；\n"
            "- description 用要点式短句，不要长段落。\n\n"
            "Context:\n{context_block}".format(context_block=context_block_safe)
        )
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def build_final_json_prompt(
        self,
        project_name: str,
        project_features: str,
        concept_struct: Dict[str, Any],
        benchmark_struct: List[Dict[str, Any]],
        trend_struct: Dict[str, Any],
    ) -> Dict[str, str]:
        """构建最终 JSON prompt，匹配高密度 Schema。"""
        def escape_braces(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        schema_description = (
            '{{{{"concept_sources": {{{{"standard_mandates": ["..."], "theoretical_basis": "..."}}}}, '
            '"benchmarking_cases": [{{{{"case_name": "...", "metadata": {{{{"architect": "...", "year": "...", "location": "..."}}}}, "similarity_logic": "...", "design_details": {{{{"form_logic": ["..."], "materiality": ["..."], "spatial_features": ["..."]}}}}, "evidence_snippet": "..."}}}}], '
            '"design_trends": {{{{"trend_list": [{{{{"trend_name": "...", "evidence_cases": ["..."], "description": "..."}}}}], "innovative_suggestions": ["..."]}}}}, '
            '"final_concept_proposal": "..."}}}}'
        )

        system_prompt = (
            "你是一名严谨的建筑策划顾问，整理并输出严格符合下述 Schema 的 JSON。\n"
            "- 仅输出 JSON 对象字符串，可被 json.loads 解析；\n"
            "- 保留上游的所有案例与证据；\n"
            "- 不要省略字段，缺失请填 Not Specified。\n\n"
            "Schema: {schema}"
        ).format(schema=schema_description)

        user_prompt = (
            "当前项目：{project_name} | 特征：{project_features}\n\n"
            "[溯源-结构化结果]\n{concept_struct}\n\n"
            "[对标-结构化结果]\n{benchmark_struct}\n\n"
            "[趋势-结构化结果]\n{trend_struct}\n\n"
            "请直接输出最终 JSON，确保字段与层级完全匹配 Schema。"
        ).format(
            project_name=escape_braces(project_name),
            project_features=escape_braces(project_features) if project_features else "未提供",
            concept_struct=escape_braces(json.dumps(concept_struct, ensure_ascii=False)),
            benchmark_struct=escape_braces(json.dumps(benchmark_struct, ensure_ascii=False)),
            trend_struct=escape_braces(json.dumps(trend_struct, ensure_ascii=False)),
        )
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
    """High-level facade that orchestrates ETL, graph-based workflow, and JSON generation."""

    def __init__(self, config: DesignConceptConfig):
        self.config = config
        self.etl = DesignConceptETL(config)
        self.vector_store = DesignConceptVectorStore(config)
        self.prompt_builder = DesignConceptPromptBuilder()
        self._llm_module: Optional[GenerationIntegrationModule] = None

        # 在生成器内部维护一个简单的状态字典，供 LangGraph 使用
        self._graph = self._build_graph()

    # ========== LLM 与索引基础设施 ==========
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

    @staticmethod
    def _strip_json_markers(text: str) -> str:
        """Remove common markdown fences around JSON payloads."""
        cleaned = text.strip()
        for pattern in [r"^```json\s*(.*)```$", r"^```\s*(.*)```$"]:
            m = re.match(pattern, cleaned, flags=re.DOTALL | re.IGNORECASE)
            if m:
                cleaned = m.group(1).strip()
                break
        return cleaned

    @classmethod
    def _safe_json_loads(cls, text: str) -> Optional[object]:
        cleaned = cls._strip_json_markers(text)
        try:
            return json.loads(cleaned)
        except Exception:
            logger.warning("JSON 解析失败，返回 None")
            return None

    # ========== LangGraph 状态定义 (TypedDict) ==========
    class _GraphState(TypedDict, total=False):
        """Graph state as TypedDict for LangGraph incremental updates."""
        project_name: str
        project_features: str
        user_project_info: str
        # 检索结果
        retrieved_sources: List[Document]
        retrieved_benchmarks: List[Document]
        retrieved_trends: List[Document]
        # 中间结构化结果
        concept_struct: Dict[str, Any]
        benchmark_struct: List[Dict[str, Any]]
        trend_struct: Dict[str, Any]
        # 最终 JSON
        final_json: Optional[Dict[str, Any]]

    # ========== LangGraph 节点实现 ==========
    def _node_concept_sourcing(self, state: "DesignConceptGenerator._GraphState") -> Dict[str, Any]:
        """溯源节点：只返回需要更新的字段。"""
        self._ensure_llm()
        query = "科学技术馆 设计理念 来源 标准 规范"
        contexts = self.vector_store.search(query, top_k=self.config.top_k or 12, filters=None)
        prompt = self.prompt_builder.build_concept_sourcing_prompt(
            state["project_name"],
            state.get("project_features", ""),
            contexts,
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        parsed = self._safe_json_loads(str(raw))
        concept_struct: Dict[str, Any] = {
            "standard_mandates": [],
            "theoretical_basis": "Not Specified",
        }
        if isinstance(parsed, dict):
            mandates = parsed.get("standard_mandates")
            if isinstance(mandates, list):
                concept_struct["standard_mandates"] = mandates
            basis = parsed.get("theoretical_basis")
            if isinstance(basis, str) and basis.strip():
                concept_struct["theoretical_basis"] = basis.strip()
        if not concept_struct["standard_mandates"]:
            fallback = {
                "source": "Not Specified",
                "clause": "Not Specified",
                "content": "Not Specified",
            }
            concept_struct["standard_mandates"] = [fallback, fallback]
        return {
            "retrieved_sources": contexts,
            "concept_struct": concept_struct,
        }

    def _node_benchmarking(self, state: "DesignConceptGenerator._GraphState") -> Dict[str, Any]:
        """对标节点：只返回需要更新的字段。"""
        self._ensure_llm()
        user_project_info = state.get("user_project_info", "")
        project_features = state.get("project_features", "")
        project_name = state["project_name"]

        target_area = self._extract_numeric_area(user_project_info)
        area_filters = self._build_area_filter(target_area)
        # location_keywords = self._extract_location_keywords(user_project_info)  # reserved for future

        filters: Dict[str, object] = {}
        if area_filters:
            filters.update(area_filters)

        query = user_project_info or project_features or project_name
        candidates = self.vector_store.search(query, top_k=self.config.top_k or 12, filters=filters)

        if len(candidates) < 3 and area_filters:
            logger.info("对标检索结果不足，降级为仅按规模过滤")
            candidates = self.vector_store.search(query, top_k=self.config.top_k or 12, filters=area_filters)

        prompt = self.prompt_builder.build_benchmarking_prompt(
            project_name,
            project_features,
            candidates,
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        parsed = self._safe_json_loads(str(raw))
        benchmark_struct: List[Dict[str, Any]] = []
        if isinstance(parsed, list):
            benchmark_struct = parsed
        return {
            "retrieved_benchmarks": candidates,
            "benchmark_struct": benchmark_struct,
        }

    def _node_trend(self, state: "DesignConceptGenerator._GraphState") -> Dict[str, Any]:
        """趋势节点：只返回需要更新的字段。"""
        self._ensure_llm()
        all_query = "科技馆 科普 博物馆 设计 趋势 新馆"
        all_docs = self.vector_store.search(all_query, top_k=32, filters=None)
        recent_docs = self._filter_by_year(all_docs, recent_years=7)
        if len(recent_docs) < 5:
            logger.info("近7年案例不足，扩大到近10年")
            recent_docs = self._filter_by_year(all_docs, recent_years=10)

        prompt = self.prompt_builder.build_trend_prompt(
            state["project_name"],
            state.get("project_features", ""),
            recent_docs,
        )
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        parsed = self._safe_json_loads(str(raw))
        trend_struct: Dict[str, Any] = {
            "trend_list": [],
            "innovative_suggestions": [],
        }
        if isinstance(parsed, dict):
            if isinstance(parsed.get("trend_list"), list):
                trend_struct["trend_list"] = parsed.get("trend_list")
            if isinstance(parsed.get("innovative_suggestions"), list):
                trend_struct["innovative_suggestions"] = parsed.get("innovative_suggestions")
        return {
            "retrieved_trends": recent_docs,
            "trend_struct": trend_struct,
        }

    def _node_final_json(self, state: "DesignConceptGenerator._GraphState") -> Dict[str, Any]:
        """最终 JSON 整合节点：纯 Python 拼接，无 LLM 调用。"""
        from datetime import datetime
        
        concept_struct = state.get("concept_struct", {
            "standard_mandates": [],
            "theoretical_basis": "Not Specified",
        })
        benchmark_struct = state.get("benchmark_struct", [])
        trend_struct = state.get("trend_struct", {
            "trend_list": [],
            "innovative_suggestions": [],
        })
        
        final_json = {
            "concept_sources": concept_struct,
            "benchmarking_cases": benchmark_struct,
            "design_trends": trend_struct,
            "generated_at": datetime.now().isoformat(),
        }
        return {"final_json": final_json}

    def _node_init(self, state: "DesignConceptGenerator._GraphState") -> Dict[str, Any]:
        """入口节点，用于支持后续节点并行执行。"""
        return {}

    # ========== Graph 构建与运行 ==========
    def _build_graph(self):
        graph = StateGraph(self._GraphState)
        graph.add_node("init", self._node_init)
        graph.add_node("concept_sourcing", self._node_concept_sourcing)
        graph.add_node("benchmarking", self._node_benchmarking)
        graph.add_node("trend", self._node_trend)
        graph.add_node("final_json", self._node_final_json)

        # 入口后并行跑三个工作节点，最后汇总到 JSON
        graph.set_entry_point("init")
        graph.add_edge("init", "concept_sourcing")
        graph.add_edge("init", "benchmarking")
        graph.add_edge("init", "trend")
        graph.add_edge("concept_sourcing", "final_json")
        graph.add_edge("benchmarking", "final_json")
        graph.add_edge("trend", "final_json")
        graph.add_edge("final_json", END)

        return graph.compile()

    # ========== 外部调用接口 ==========
    def generate(
        self,
        project_name: str,
        project_features: str,
        user_project_info: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        """运行“溯源-对标-趋势-整合”四段式图，并返回 JSON 结果。"""

        if not project_name:
            raise ValueError("project_name 不能为空")

        self.ensure_index(rebuild=False)

        # 初始状态为普通 dict，符合 TypedDict schema
        init_state: DesignConceptGenerator._GraphState = {
            "project_name": project_name,
            "project_features": project_features or "",
            "user_project_info": user_project_info or project_features or project_name,
        }

        final_state = self._graph.invoke(init_state)

        # LangGraph compiled graph 返回的是 dict 状态，而不是 dataclass 实例
        if isinstance(final_state, dict):
            json_result = final_state.get("final_json")
            if dry_run:
                return {
                    "retrieved_sources": final_state.get("retrieved_sources"),
                    "retrieved_benchmarks": final_state.get("retrieved_benchmarks"),
                    "retrieved_trends": final_state.get("retrieved_trends"),
                    "concept_struct": final_state.get("concept_struct", {}),
                    "benchmark_struct": final_state.get("benchmark_struct", []),
                    "trend_struct": final_state.get("trend_struct", {}),
                }

            return {
                "json_result": json_result,
                "raw_text": {
                    "concept_struct": final_state.get("concept_struct", {}),
                    "benchmark_struct": final_state.get("benchmark_struct", []),
                    "trend_struct": final_state.get("trend_struct", {}),
                },
            }

        # 回退：若未来返回的是 dataclass 实例
        if dry_run:
            return {
                "retrieved_sources": final_state.retrieved_sources,
                "retrieved_benchmarks": final_state.retrieved_benchmarks,
                "retrieved_trends": final_state.retrieved_trends,
                "concept_struct": final_state.concept_struct,
                "benchmark_struct": final_state.benchmark_struct,
                "trend_struct": final_state.trend_struct,
            }

        return {
            "json_result": final_state.final_json,
            "raw_text": {
                "concept_struct": final_state.concept_struct,
                "benchmark_struct": final_state.benchmark_struct,
                "trend_struct": final_state.trend_struct,
            },
        }

    # ========== 辅助函数：面积、区位与年份解析 ==========
    @staticmethod
    def _extract_numeric_area(text: str) -> Optional[float]:
        if not text:
            return None
        match = re.search(r"(\d+[\d,]*\.?\d*)\s*(平方米|m2|m²|sqm|平米|㎡)?", text)
        if not match:
            return None
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            return None

    @staticmethod
    def _build_area_filter(target_area: Optional[float]) -> Optional[Dict[str, float]]:
        if not target_area or target_area <= 0:
            return None
        delta = target_area * 0.3
        return {
            "min_area": max(target_area - delta, 0),
            "max_area": target_area + delta,
        }

    @staticmethod
    def _extract_location_keywords(text: str) -> List[str]:
        if not text:
            return []
        keywords = []
        for kw in ["滨水", "沿江", "河畔", "湖畔", "海边", "严寒", "寒冷", "南方", "北方", "山地", "丘陵"]:
            if kw in text:
                keywords.append(kw)
        return keywords

    @staticmethod
    def _filter_by_year(docs: Sequence[Document], recent_years: int = 7) -> List[Document]:
        if not docs:
            return []
        current_year = 2025
        min_year = current_year - recent_years
        filtered: List[Document] = []
        for doc in docs:
            meta = doc.metadata or {}
            year = None
            for key in ["Year", "year", "opening_date", "开馆时间", "建成时间", "completion_year"]:
                value = meta.get(key)
                if not value:
                    continue
                if isinstance(value, (int, float)):
                    year = int(value)
                    break
                if isinstance(value, str):
                    m = re.search(r"(19|20)\d{2}", value)
                    if m:
                        year = int(m.group(0))
                        break
            if year and min_year <= year <= current_year:
                filtered.append(doc)
        return filtered
