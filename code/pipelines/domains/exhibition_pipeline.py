"""Exhibition pipeline with LangGraph dual-track (Foundation + Vision)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import END, START, StateGraph

from config import ExhibitionConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import ExhibitionDataExtractor
from utils.indicator_analyzer import calculate_compliance_specs, get_size_class

logger = logging.getLogger(__name__)


class ExhibitionState(TypedDict, total=False):
    """LangGraph state for exhibition generation."""

    core_planning: Dict[str, any]
    trend_analysis: Dict[str, any]
    final_json: Dict[str, any]


class ExhibitionVectorStore:
    """Vector index dedicated to exhibition space knowledge."""

    def __init__(self, config: ExhibitionConfig):
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
            logger.info("已加载展览空间索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("展览空间文档为空，无法构建索引")
        logger.info("正在构建设计展览空间索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("展览空间索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("展览空间索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class ExhibitionGenerator:
    """Dual-track Exhibition generator with evidence-first enforcement."""

    FOUNDATION_QUERIES = [
        "常设展厅 功能占比",
        "展览教育 功能占比",
        "展教主题",
        "常设展厅 净高 柱网 荷载",
        "常设展厅 防火 疏散",
    ]

    TREND_QUERIES = [
        "Science Museum Exhibition Trend",
        "Immersive interactive narrative museum",
        "open exploration flow museum",
        "沉浸式 交互 叙事 科技馆",
    ]

    def __init__(self, config: ExhibitionConfig):
        self.config = config
        self.extractor = ExhibitionDataExtractor(config)
        self.vector_store = ExhibitionVectorStore(config)
        self._llm_module: Optional[GenerationIntegrationModule] = None
        self._docs_cache: Optional[List[Document]] = None

    # ------------------------------------------------------------------
    # Infra
    # ------------------------------------------------------------------
    def ensure_index(self, rebuild: bool = False) -> None:
        self.vector_store.ensure_ready(self.extractor.load_documents, rebuild=rebuild)
        self._docs_cache = None

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
            retrieved = self.vector_store.search(q, top_k=top_k)
            for doc in retrieved:
                if allowed_sources and doc.metadata.get("source_type") not in allowed_sources:
                    continue
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

    # ------------------------------------------------------------------
    # Branch A: Foundation
    # ------------------------------------------------------------------
    def _run_foundation_branch(
        self,
        project_name: str,
        target_area: float,
        dry_run: bool,
        top_k: int,
    ) -> Dict[str, any]:
        size_info = get_size_class(target_area)
        compliance_specs = calculate_compliance_specs(target_area, size_info.get("class_name"))

        area_ranges = compliance_specs.get("function_area_ranges", {}) or {}
        exh_range = area_ranges.get("exhibition_education") or {}
        ratio_text = None
        if exh_range:
            ratio_text = f"展览教育 {exh_range.get('percent_min', '-')}-{exh_range.get('percent_max', '-')}%"

        contexts = self._search_and_filter(self.FOUNDATION_QUERIES, top_k, ["gb_standard", "zlj", "china"])

        if dry_run and not contexts:
            core_planning = {
                "standard_compliance": {
                    "mandatory_zones": ["[No Specific Evidence]"] ,
                    "area_ratios": ratio_text or "[General Knowledge]",
                    "spatial_requirements": "[No Specific Evidence]",
                },
                "baseline_themes": [
                    {
                        "theme_name": "[General Knowledge]",
                        "typical_ratio": "[General Knowledge]",
                        "description": "[General Knowledge]",
                        "reference": "[No Specific Evidence]",
                    }
                ],
            }
            return {"core_planning": core_planning, "contexts": contexts, "prompt": None}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名合规官，仅基于提供的上下文抽取信息，输出必须为中文。若上下文为空，对应字段填入 '缺失' 或 '[无具体证据]'，严禁编造GB条款。"
                    " CRITICAL: 输出必须为简体中文，如检索内容为英文需翻译，不得输出英文句子。",
                ),
                (
                    "human",
                    "项目: {project_name}\n目标面积: {target_area} m²\n等级: {level}\n功能配比: {ratio_text}\n上下文:\n{context}\n\n请输出 JSON，字段: standard_compliance.mandatory_zones, standard_compliance.area_ratios, standard_compliance.spatial_requirements, baseline_themes (列表: theme_name, typical_ratio, description, reference)。\n缺失则填 '缺失' 或 '[无具体证据]'。\n{format_instructions}",
                ),
            ]
        )
        context_text = self._format_context(contexts)
        llm_output: Dict[str, any] = {}
        try:
            llm_output = prompt | self._llm_module.llm | parser
            llm_output = llm_output.invoke(
                {
                    "project_name": project_name,
                    "target_area": target_area,
                    "level": size_info.get("class_name"),
                    "ratio_text": ratio_text or "",
                    "context": context_text[:6000],
                    "format_instructions": fmt,
                }
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Foundation LLM failed, falling back: %s", exc)

        standard = llm_output.get("standard_compliance") or {
            "mandatory_zones": ["展览教育用房", "公共服务配套"],
            "area_ratios": ratio_text or "[通用知识]",
            "spatial_requirements": "[通用知识]",
        }
        baseline = llm_output.get("baseline_themes") or [
            {
                "theme_name": "科学与技术基础",
                "typical_ratio": "[通用知识]",
                "description": "基本科学、工程与工业文明的常设叙事。",
                "reference": "[通用知识]",
            }
        ]

        core_planning = {
            "standard_compliance": standard,
            "baseline_themes": baseline,
        }

        return {"core_planning": core_planning, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Branch B: Vision / Trends
    # ------------------------------------------------------------------
    def _run_trend_branch(
        self,
        project_name: str,
        project_features: str,
        dry_run: bool,
        top_k: int,
    ) -> Dict[str, any]:
        contexts = self._search_and_filter(self.TREND_QUERIES, top_k, ["archdaily", "world"])
        if dry_run and not contexts:
            trend_analysis = {
                "global_trends": [
                    {
                        "trend_name": "[General Knowledge]",
                        "description": "[General Knowledge]",
                        "evidence_cases": [],
                        "is_general_knowledge": True,
                    }
                ],
                "innovative_features": [],
            }
            return {"trend_analysis": trend_analysis, "contexts": contexts, "prompt": None}

        self._ensure_llm()
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是一名策展人，仅基于上下文提炼3-5条全球趋势，输出必须为中文。每条趋势需包含 evidence_snippet 引文；若无相关案例，则 evidence_cases 为空并标记 is_general_knowledge=true，不得编造。"
                    " CRITICAL: 输出必须为简体中文，如检索内容为英文需翻译，不得输出英文句子。",
                ),
                (
                    "human",
                    "项目: {project_name}\n特征: {project_features}\n上下文:\n{context}\n\n请输出 JSON: global_trends (trend_name, description, evidence_cases[list of {{case_name, snippet}}], is_general_knowledge[bool]); innovative_features[list]。\n若上下文缺失，则标记为行业通用认知（未验证），并用中文表述。\n{format_instructions}",
                ),
            ]
        )
        context_text = self._format_context(contexts)
        llm_output: Dict[str, any] = {}
        try:
            chain = prompt | self._llm_module.llm | parser
            llm_output = chain.invoke(
                {
                    "project_name": project_name,
                    "project_features": project_features,
                    "context": context_text[:6000],
                    "format_instructions": fmt,
                }
            ) or {}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Trend LLM failed, using general knowledge: %s", exc)

        if not isinstance(llm_output, dict):
            llm_output = {}

        trends = llm_output.get("global_trends") if isinstance(llm_output, dict) else None
        trends = trends or [
            {
                "trend_name": "沉浸式叙事",
                "description": "强调光影、交互与故事线的沉浸体验。",
                "evidence_cases": [],
                "is_general_knowledge": True,
            }
        ]
        for item in trends:
            if not item.get("evidence_cases"):
                item["is_general_knowledge"] = True

        trend_analysis = {
            "global_trends": trends,
            "innovative_features": llm_output.get("innovative_features") or [],
        }
        return {"trend_analysis": trend_analysis, "prompt": prompt, "contexts": contexts}

    # ------------------------------------------------------------------
    # Merge node
    # ------------------------------------------------------------------
    @staticmethod
    def _merge(core_planning: Dict[str, any], trend_analysis: Dict[str, any]) -> Dict[str, any]:
        return {
            "core_planning_data": core_planning or {},
            "trend_analysis_data": trend_analysis or {},
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "[No Specific Evidence]"
        formatted = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"案例{idx}"
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            formatted.append(f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}")
        return "\n\n".join(formatted)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        project_name: str,
        project_features: str,
        *,
        target_area: Optional[float] = None,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        rebuild_index: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        self.ensure_index(rebuild=rebuild_index)

        target_area = float(target_area) if target_area is not None else 20000.0
        top_k = top_k or self.config.top_k

        graph = StateGraph(ExhibitionState)

        def core_node(state: ExhibitionState) -> Dict[str, any]:
            result = self._run_foundation_branch(project_name, target_area, dry_run, top_k)
            return {"core_planning": result.get("core_planning"), "_core_prompt": result.get("prompt"), "_core_contexts": result.get("contexts")}

        def trend_node(state: ExhibitionState) -> Dict[str, any]:
            result = self._run_trend_branch(project_name, project_features, dry_run, top_k)
            return {"trend_analysis": result.get("trend_analysis"), "_trend_prompt": result.get("prompt"), "_trend_contexts": result.get("contexts")}

        def merge_node(state: ExhibitionState) -> Dict[str, any]:
            merged = self._merge(state.get("core_planning") or {}, state.get("trend_analysis") or {})
            return {"final_json": merged}

        graph.add_node("core", core_node)
        graph.add_node("trend", trend_node)
        graph.add_node("merge", merge_node)

        graph.add_edge(START, "core")
        graph.add_edge(START, "trend")
        graph.add_edge("core", "merge")
        graph.add_edge("trend", "merge")
        graph.add_edge("merge", END)

        app = graph.compile()
        final_state = app.invoke(ExhibitionState())

        final_json = final_state.get("final_json") or {}
        response_text = None if dry_run else json.dumps(final_json, ensure_ascii=False, indent=2)

        return {
            "response": response_text,
            "core_planning": final_state.get("core_planning"),
            "trend_analysis": final_state.get("trend_analysis"),
            "final_json": final_json,
            "prompts": {
                "core": final_state.get("_core_prompt"),
                "trend": final_state.get("_trend_prompt"),
            },
            "contexts": {
                "core": final_state.get("_core_contexts"),
                "trend": final_state.get("_trend_contexts"),
            },
        }

