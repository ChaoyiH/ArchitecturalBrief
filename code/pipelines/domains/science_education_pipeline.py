"""LangGraph dual-track pipeline for Science & Education (科教活动)."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from config import ScienceEducationConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import ScienceEducationDataExtractor
from utils.indicator_analyzer import analyze_indicators, calculate_compliance_specs, get_size_class

logger = logging.getLogger(__name__)


class ScienceEducationState(TypedDict, total=False):
    """LangGraph state for science education generation."""

    normative_data: Dict[str, Any]
    activity_data: Dict[str, Any]
    final_json: Dict[str, Any]
    _normative_prompt: Optional[ChatPromptTemplate]
    _activity_prompt: Optional[ChatPromptTemplate]
    _normative_contexts: Optional[List[Document]]
    _activity_contexts: Optional[List[Document]]


class ScienceEducationVectorStore:
    """FAISS-backed index with metadata filtering support."""

    def __init__(self, config: ScienceEducationConfig):
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
            logger.info("已加载科教活动索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("科教活动文档为空，无法构建索引")
        logger.info("正在构建科教活动索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("科教活动索引保存至: %s", self.config.index_save_path)

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
            raise RuntimeError("科教活动索引尚未构建")
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
            elif key == "category":
                categories = meta.get("project_type") or meta.get("categories") or []
                if isinstance(categories, str):
                    categories = [c.strip() for c in categories.split(",")]
                normalized = {c.lower() for c in categories} if isinstance(categories, list) else set()
                if str(value).lower() not in normalized:
                    return False
            elif key in {"min_area", "max_area", "min_height", "max_height"}:
                # Keep parity with DesignConceptVectorStore numeric guards
                area = meta.get("total_area_num")
                height = meta.get("building_height_num")
                if key == "min_area" and (area is None or area < float(value)):
                    return False
                if key == "max_area" and (area is not None and area > float(value)):
                    return False
                if key == "min_height" and (height is None or height < float(value)):
                    return False
                if key == "max_height" and (height is not None and height > float(value)):
                    return False
            else:
                if meta.get(key) != value:
                    return False
        return True


class ScienceEducationGenerator:
    """Dual-track (规范 + 案例) generator for 科教活动."""

    NORMATIVE_QUERIES = [
        "科技馆 教育用房 指标",
        "实验室 安全 规范",
        "培训教室 面积",
    ]

    TREND_QUERIES = [
        "Science Education Program",
        "Workshop design",
        "Chemistry Lab",
        "Maker Space",
        "科普活动",
        "研学",
    ]

    def __init__(self, config: ScienceEducationConfig):
        self.config = config
        self.extractor = ScienceEducationDataExtractor(config)
        self.vector_store = ScienceEducationVectorStore(config)
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
            name = meta.get("project_name") or meta.get("doc_name") or f"片段{idx}"
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
            compliance = calculate_compliance_specs(target_area or 0, size.get("class_name"))
            return {
                "target_area": target_area,
                "target_classification": size,
                "compliance": compliance,
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
        compliance_info = indicator.get("compliance") or {}
        target_level = class_info.get("class_name") or "中型馆"
        compliance_source = compliance_info.get("standard_source") or "建标 101-2007"

        area_ranges = compliance_info.get("function_area_ranges") or {}
        edu_range = area_ranges.get("exhibition_education") or {}
        ratio_hint = None
        if edu_range:
            ratio_hint = f"展览教育占比: {edu_range.get('percent_min', '-')}%-{edu_range.get('percent_max', '-')}%"

        contexts = self._search_and_filter(self.NORMATIVE_QUERIES, top_k, ["gb_standard", "zlj"])

        if not contexts:
            fallback_data = {
                "target_level": target_level,
                "compliance_source": compliance_source,
                "mandatory_rooms": [],
                "safety_specs": [],
            }
            return {"normative_data": fallback_data, "prompt": None, "contexts": contexts}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是科教活动合规官。仅依据下方标准/规范上下文抽取信息，输出中文 JSON。缺失字段用'[无具体证据]'或'缺失'，严禁编造标准。",
                ),
                (
                    "human",
                    "项目: {project_name}\n目标面积: {target_area} m²\n等级: {level}\n{ratio_hint}\n上下文:\n{context}\n\n"
                    "请输出 JSON，字段: normative_requirements.target_level, normative_requirements.compliance_source, \n"
                    "normative_requirements.mandatory_rooms (列表: room_type, min_area, note 需含证据来源), normative_requirements.safety_specs (列表字符串)。\n"
                    "缺失请填'[无具体证据]'。\n{format_instructions}",
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
                    "ratio_hint": ratio_hint or "",
                    "context": self._format_context(contexts)[:6000],
                    "format_instructions": fmt,
                }
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("规范分支 LLM 失败，使用回退: %s", exc)

        if not isinstance(llm_output, dict):
            llm_output = {}

        parsed = llm_output.get("normative_requirements") if isinstance(llm_output, dict) else None

        mandatory_rooms = []
        if parsed and isinstance(parsed.get("mandatory_rooms"), list):
            mandatory_rooms = parsed.get("mandatory_rooms")
        elif edu_range:
            mandatory_rooms = [
                {
                    "room_type": "通用教室",
                    "min_area": "60-80 m²",
                    "note": "基于展教占比推估，[通用知识]",
                },
                {
                    "room_type": "实验/创客空间",
                    "min_area": "80-120 m²",
                    "note": "需排风与耐污地面，[通用知识]",
                },
            ]

        safety_specs = []
        if parsed and isinstance(parsed.get("safety_specs"), list):
            safety_specs = parsed.get("safety_specs")
        else:
            safety_specs = [
                "化学/湿式实验需独立排风与洗眼器 [通用知识]",
                "教室宜设置双出口与应急照明 [通用知识]",
            ]

        normative_data = {
            "target_level": parsed.get("target_level") if parsed else target_level,
            "compliance_source": parsed.get("compliance_source") if parsed else compliance_source,
            "mandatory_rooms": mandatory_rooms,
            "safety_specs": safety_specs,
        }

        return {"normative_data": normative_data, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Branch B: Empirical / Activities
    # ------------------------------------------------------------------
    def _run_activity_branch(
        self,
        project_name: str,
        project_features: str,
        dry_run: bool,
        top_k: int,
    ) -> Dict[str, Any]:
        contexts = self._search_and_filter(self.TREND_QUERIES, top_k, ["archdaily", "china", "world"])

        if not contexts:
            fallback = {
                "trends": [],
                "suggested_curriculum_zones": [],
            }
            return {"activity_data": fallback, "prompt": None, "contexts": contexts}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是科教活动策划人。仅依据案例上下文提出活动与空间需求，输出 JSON，缺失时标记 is_general_knowledge=true 且 reference_case 用 '[General Knowledge]'。",
                ),
                (
                    "human",
                    "项目: {project_name}\n特征: {project_features}\n上下文:\n{context}\n\n"
                    "输出 JSON: activity_programming.trends (3-5 条，每条含 activity_name, spatial_need, reference_case, is_general_knowledge[bool]),\n"
                    "activity_programming.suggested_curriculum_zones (列表字符串)。\n{format_instructions}",
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
            logger.exception("活动分支 LLM 失败，使用回退: %s", exc)

        if not isinstance(llm_output, dict):
            llm_output = {}

        parsed = llm_output.get("activity_programming") if isinstance(llm_output, dict) else None
        trends = parsed.get("trends") if parsed and isinstance(parsed.get("trends"), list) else None
        if not trends:
            trends = [
                {
                    "activity_name": "开放玻璃实验室",
                    "spatial_need": "临展厅或走廊可视，需隔音与安全隔断",
                    "reference_case": "[General Knowledge]",
                    "is_general_knowledge": True,
                },
                {
                    "activity_name": "创客工坊/机器人编程",
                    "spatial_need": "提供地面电源、储物、可清洗桌面",
                    "reference_case": "[General Knowledge]",
                    "is_general_knowledge": True,
                },
            ]

        for item in trends:
            if not item.get("reference_case"):
                item["reference_case"] = "[General Knowledge]"
            if item.get("is_general_knowledge") is None and not contexts:
                item["is_general_knowledge"] = True

        suggested_zones = parsed.get("suggested_curriculum_zones") if parsed else None
        if not suggested_zones:
            suggested_zones = [
                "生命科学区（湿式实验室）",
                "工程与创客区（Maker Space）",
                "科学表演区（可视化舞台）",
            ]

        activity_data = {
            "trends": trends,
            "suggested_curriculum_zones": suggested_zones,
        }

        return {"activity_data": activity_data, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Merge / Synthesis
    # ------------------------------------------------------------------
    @staticmethod
    def _build_spatial_matrix(trends: List[Dict[str, Any]], rooms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        matrix: List[Dict[str, Any]] = []
        if not trends or not rooms:
            return matrix
        for activity in trends:
            name = activity.get("activity_name") or "未命名活动"
            need = (activity.get("spatial_need") or "").lower()
            candidate = rooms[0].get("room_type") if rooms else "通用教室"
            for room in rooms:
                rt_lower = (room.get("room_type") or "").lower()
                if "实验" in name or "lab" in need:
                    if "实验" in rt_lower or "lab" in rt_lower:
                        candidate = room.get("room_type")
                        break
                if "maker" in need or "创客" in name:
                    if "创客" in rt_lower or "工坊" in rt_lower:
                        candidate = room.get("room_type")
                        break
                if "教室" in rt_lower and candidate == rooms[0].get("room_type"):
                    candidate = room.get("room_type")
            matrix.append(
                {
                    "activity_name": name,
                    "candidate_room": candidate,
                    "rationale": "基于空间需求与房间类型匹配",
                }
            )
        return matrix

    def _merge(self, normative_data: Dict[str, Any], activity_data: Dict[str, Any]) -> Dict[str, Any]:
        matrix = self._build_spatial_matrix(
            activity_data.get("trends") or [],
            normative_data.get("mandatory_rooms") or [],
        )
        return {
            "normative_requirements": normative_data or {},
            "activity_programming": activity_data or {},
            "spatial_integration_matrix": matrix,
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
    ) -> Dict[str, Any]:
        self.ensure_index(rebuild=rebuild_index)

        target_area = float(target_area) if target_area is not None else 20000.0
        top_k = top_k or self.config.top_k

        graph = StateGraph(ScienceEducationState)

        def normative_node(state: ScienceEducationState) -> Dict[str, Any]:
            result = self._run_normative_branch(project_name, target_area, dry_run, top_k)
            return {
                "normative_data": result.get("normative_data"),
                "_normative_prompt": result.get("prompt"),
                "_normative_contexts": result.get("contexts"),
            }

        def activity_node(state: ScienceEducationState) -> Dict[str, Any]:
            result = self._run_activity_branch(project_name, project_features, dry_run, top_k)
            return {
                "activity_data": result.get("activity_data"),
                "_activity_prompt": result.get("prompt"),
                "_activity_contexts": result.get("contexts"),
            }

        def merge_node(state: ScienceEducationState) -> Dict[str, Any]:
            merged = self._merge(state.get("normative_data") or {}, state.get("activity_data") or {})
            return {"final_json": merged}

        graph.add_node("normative", normative_node)
        graph.add_node("activity", activity_node)
        graph.add_node("merge", merge_node)

        graph.add_edge(START, "normative")
        graph.add_edge(START, "activity")
        graph.add_edge("normative", "merge")
        graph.add_edge("activity", "merge")
        graph.add_edge("merge", END)

        app = graph.compile()
        final_state = app.invoke(ScienceEducationState())

        final_json = final_state.get("final_json") or {}
        response_text = None if dry_run else json.dumps(final_json, ensure_ascii=False, indent=2)

        return {
            "response": response_text,
            "normative_data": final_state.get("normative_data"),
            "activity_data": final_state.get("activity_data"),
            "final_json": final_json,
            "prompts": {
                "normative": final_state.get("_normative_prompt"),
                "activity": final_state.get("_activity_prompt"),
            },
            "contexts": {
                "normative": final_state.get("_normative_contexts"),
                "activity": final_state.get("_activity_contexts"),
            },
        }
