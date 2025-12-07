"""LangGraph dual-track pipeline for public service (前场公共服务区)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from config import PublicServiceConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import PublicServiceDataExtractor
from utils.indicator_analyzer import analyze_indicators, get_size_class

logger = logging.getLogger(__name__)


class PublicServiceState(TypedDict, total=False):
    """LangGraph state for public service generation."""

    normative_data: Dict[str, Any]
    design_data: Dict[str, Any]
    final_json: Dict[str, Any]
    _normative_prompt: Optional[ChatPromptTemplate]
    _design_prompt: Optional[ChatPromptTemplate]
    _normative_contexts: Optional[List[Document]]
    _design_contexts: Optional[List[Document]]


class PublicServiceVectorStore:
    """Vector store dedicated to public service area knowledge with filtering."""

    def __init__(self, config: PublicServiceConfig):
        self.config = config
        self.embedding = get_embedding(model_name=config.embedding_model)
        self.vectorstore: Optional[FAISS] = None

    def load(self) -> bool:
        try:
            self.vectorstore = FAISS.load_local(
                self.config.index_save_path,
                self.embedding,
                allow_dangerous_deserialization=True,
            )
            logger.info("已加载公共服务区索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("公共服务区文档为空，无法构建索引")
        logger.info("正在构建公共服务区索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("公共服务区索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("公共服务区索引尚未构建")
        final_k = top_k if top_k and top_k > 0 else 6
        search_k = max(final_k, 12)
        retrieved = self.vectorstore.similarity_search(query, k=search_k)
        if not filters:
            return retrieved[:final_k]
        filtered = [doc for doc in retrieved if self._match_filters(doc, filters)]
        return filtered[:final_k] if filtered else retrieved[:final_k]

    @staticmethod
    def _match_filters(doc: Document, filters: Dict[str, Any]) -> bool:
        meta = doc.metadata or {}
        for key, value in filters.items():
            if key == "source_type":
                allowed = value if isinstance(value, (list, tuple, set)) else [value]
                if meta.get("source_type") not in allowed:
                    return False
            else:
                if meta.get(key) != value:
                    return False
        return True


class PublicServiceGenerator:
    """Dual-track generator for front-of-house public service."""

    NORMATIVE_QUERIES = [
        "卫生间 数量",
        "无障碍",
        "门厅 面积",
        "通用设计",
    ]

    TREND_QUERIES = [
        "Museum Shop",
        "Cafe design",
        "Lobby atrium",
        "Ticket counter",
        "文创",
        "餐饮",
    ]

    def __init__(self, config: PublicServiceConfig):
        self.config = config
        self.extractor = PublicServiceDataExtractor(config)
        self.vector_store = PublicServiceVectorStore(config)
        self._llm_module: Optional[GenerationIntegrationModule] = None

    # ------------------------------------------------------------------
    # Infra helpers
    # ------------------------------------------------------------------
    def ensure_index(self, rebuild: bool = False) -> None:
        self.vector_store.ensure_ready(self.extractor.load_documents, rebuild=rebuild)

    def _ensure_llm(self) -> None:
        if self._llm_module is None:
            self._llm_module = GenerationIntegrationModule(
                provider=self.config.llm_provider,
                model_name=self.config.llm_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

    def _search_and_filter(self, queries: List[str], top_k: int, allowed_sources: Sequence[str]) -> List[Document]:
        results: List[Document] = []
        seen = set()
        for q in queries:
            docs = self.vector_store.search(q, top_k=top_k, filters={"source_type": allowed_sources})
            for doc in docs:
                key = doc.metadata.get("chunk_id") or doc.metadata.get("source") or doc.page_content[:80]
                if key in seen:
                    continue
                seen.add(key)
                results.append(doc)
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "[无具体证据]"
        formatted = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"案例{idx}"
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            snippet = snippet.replace("{", "{{").replace("}", "}}")
            formatted.append(f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}")
        return "\n\n".join(formatted)

    def _safe_indicator_snapshot(self, target_area: float) -> Dict[str, Any]:
        try:
            return analyze_indicators(target_area)
        except Exception as exc:  # noqa: BLE001
            logger.exception("indicator_analyzer 失败，使用回退: %s", exc)
            size = get_size_class(target_area if target_area else 0)
            return {
                "target_area": target_area,
                "target_classification": size,
            }

    # ------------------------------------------------------------------
    # Branch A: Normative / Compliance
    # ------------------------------------------------------------------
    def _run_normative_branch(
        self,
        project_name: str,
        target_area: float,
        dry_run: bool,
        top_k: int,
    ) -> Dict[str, Any]:
        indicator = self._safe_indicator_snapshot(target_area)
        class_info = indicator.get("target_classification") or {}
        target_level = class_info.get("class_name") or "中型馆"

        contexts = self._search_and_filter(self.NORMATIVE_QUERIES, top_k, ["gb_standard", "zlj"])

        if not contexts:
            fallback = {
                "project_scale_category": target_level,
                "lobby_capacity_guide": "[无具体证据]",
                "sanitary_facilities": {
                    "ratio_requirement": "[无具体证据]",
                    "accessibility_note": "[无具体证据]",
                },
                "compliance_source": [],
            }
            return {"normative_data": fallback, "prompt": None, "contexts": contexts}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Code Consultant. 仅基于上下文抽取强制性条款，引用规范名称与条款；若缺失写 'Standard not specified in context'。输出中文 JSON。",
                ),
                (
                    "human",
                    "项目: {project_name}\n目标面积: {target_area} m²\n等级: {level}\n上下文:\n{context}\n\n"
                    "输出 JSON: normative_requirements.project_scale_category, normative_requirements.lobby_capacity_guide, normative_requirements.sanitary_facilities.ratio_requirement, normative_requirements.sanitary_facilities.accessibility_note, normative_requirements.compliance_source[list]\n"
                    "缺失则填 'Standard not specified in context' 或 '[无具体证据]'。\n{format_instructions}",
                ),
            ]
        )

        llm_output: Dict[str, Any] = {}
        try:
            chain = prompt | self._llm_module.llm | parser
            llm_output = chain.invoke(
                {
                    "project_name": project_name,
                    "target_area": target_area,
                    "level": target_level,
                    "context": self._format_context(contexts)[:6000],
                    "format_instructions": fmt,
                }
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("规范分支 LLM 失败，使用回退: %s", exc)

        if not isinstance(llm_output, dict):
            llm_output = {}

        parsed = llm_output.get("normative_requirements") if isinstance(llm_output, dict) else None

        lobby_capacity = None
        sanitary = None
        sources = []
        if parsed:
            lobby_capacity = parsed.get("lobby_capacity_guide")
            sanitary = parsed.get("sanitary_facilities")
            sources = parsed.get("compliance_source") or []

        normative_data = {
            "project_scale_category": parsed.get("project_scale_category") if parsed else target_level,
            "lobby_capacity_guide": lobby_capacity or "[无具体证据]",
            "sanitary_facilities": sanitary
            if sanitary and isinstance(sanitary, dict)
            else {
                "ratio_requirement": "[无具体证据]",
                "accessibility_note": "[无具体证据]",
            },
            "compliance_source": sources or [],
        }

        return {"normative_data": normative_data, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Branch B: Empirical / Trends
    # ------------------------------------------------------------------
    def _run_design_branch(
        self,
        project_name: str,
        project_features: str,
        dry_run: bool,
        top_k: int,
    ) -> Dict[str, Any]:
        contexts = self._search_and_filter(self.TREND_QUERIES, top_k, ["archdaily", "china", "world"])

        if not contexts:
            fallback = {
                "entrance_lobby": {
                    "flow_strategy": "[无具体证据]",
                    "atmosphere": "[无具体证据]",
                },
                "commercial_planning": [],
                "amenity_innovation": "[无具体证据]",
            }
            return {"design_data": fallback, "prompt": None, "contexts": contexts}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an Architect. 仅基于案例上下文总结趋势与布局策略，输出中文 JSON；每条策略需给出 case_reference (项目名称或来源)，若缺失标记 '[General Knowledge]' 并 is_general_knowledge=true。",
                ),
                (
                    "human",
                    "项目: {project_name}\n特征: {project_features}\n上下文:\n{context}\n\n"
                    "输出 JSON: design_strategies.entrance_lobby.flow_strategy, design_strategies.entrance_lobby.atmosphere,\n"
                    "design_strategies.commercial_planning (数组: zone, location_logic, case_reference, is_general_knowledge[bool]),\n"
                    "design_strategies.amenity_innovation。\n{format_instructions}",
                ),
            ]
        )

        llm_output: Dict[str, Any] = {}
        try:
            chain = prompt | self._llm_module.llm | parser
            llm_output = chain.invoke(
                {
                    "project_name": project_name,
                    "project_features": project_features,
                    "context": self._format_context(contexts)[:6000],
                    "format_instructions": fmt,
                }
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("体验分支 LLM 失败，使用回退: %s", exc)

        if not isinstance(llm_output, dict):
            llm_output = {}

        parsed = llm_output.get("design_strategies") if isinstance(llm_output, dict) else None
        commercial = parsed.get("commercial_planning") if parsed and isinstance(parsed.get("commercial_planning"), list) else None
        if not commercial:
            commercial = [
                {
                    "zone": "Gift Shop",
                    "location_logic": "出口必经位置，结合动线末端布置",
                    "case_reference": "[General Knowledge]",
                    "is_general_knowledge": True,
                },
                {
                    "zone": "Cafe/Dining",
                    "location_logic": "首层或景观面，独立排烟",
                    "case_reference": "[General Knowledge]",
                    "is_general_knowledge": True,
                },
            ]

        entrance = parsed.get("entrance_lobby") if parsed else None
        amenity = parsed.get("amenity_innovation") if parsed else None

        for item in commercial:
            if not item.get("case_reference"):
                item["case_reference"] = "[General Knowledge]"
            if item.get("is_general_knowledge") is None and not contexts:
                item["is_general_knowledge"] = True

        design_data = {
            "entrance_lobby": entrance
            if entrance and isinstance(entrance, dict)
            else {
                "flow_strategy": "[无具体证据]",
                "atmosphere": "[无具体证据]",
            },
            "commercial_planning": commercial,
            "amenity_innovation": amenity if amenity else "[无具体证据]",
        }

        return {"design_data": design_data, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Merge / Synthesis
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(normative_data: Dict[str, Any], design_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "normative_requirements": normative_data or {},
            "design_strategies": design_data or {},
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        project_name: str,
        project_features: str,
        *,
        query: Optional[str] = None,
        target_area: Optional[float] = None,
        top_k: Optional[int] = None,
        rebuild_index: bool = False,
        dry_run: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        self.ensure_index(rebuild=rebuild_index)

        target_area = float(target_area) if target_area is not None else 20000.0
        top_k = top_k or self.config.top_k

        graph = StateGraph(PublicServiceState)

        def normative_node(state: PublicServiceState) -> Dict[str, Any]:
            result = self._run_normative_branch(project_name, target_area, dry_run, top_k)
            return {
                "normative_data": result.get("normative_data"),
                "_normative_prompt": result.get("prompt"),
                "_normative_contexts": result.get("contexts"),
            }

        def design_node(state: PublicServiceState) -> Dict[str, Any]:
            result = self._run_design_branch(project_name, project_features, dry_run, top_k)
            return {
                "design_data": result.get("design_data"),
                "_design_prompt": result.get("prompt"),
                "_design_contexts": result.get("contexts"),
            }

        def merge_node(state: PublicServiceState) -> Dict[str, Any]:
            merged = self._merge(state.get("normative_data") or {}, state.get("design_data") or {})
            return {"final_json": merged}

        graph.add_node("normative", normative_node)
        graph.add_node("design", design_node)
        graph.add_node("merge", merge_node)

        graph.add_edge(START, "normative")
        graph.add_edge(START, "design")
        graph.add_edge("normative", "merge")
        graph.add_edge("design", "merge")
        graph.add_edge("merge", END)

        app = graph.compile()
        final_state = app.invoke(PublicServiceState())

        final_json = final_state.get("final_json") or {}
        response_text = None if dry_run else json.dumps(final_json, ensure_ascii=False, indent=2)

        return {
            "response": response_text,
            "normative_data": final_state.get("normative_data"),
            "design_data": final_state.get("design_data"),
            "final_json": final_json,
            "prompts": {
                "normative": final_state.get("_normative_prompt"),
                "design": final_state.get("_design_prompt"),
            },
            "contexts": {
                "normative": final_state.get("_normative_contexts"),
                "design": final_state.get("_design_contexts"),
            },
        }
