"""
Central Hub (综合大厅/中庭) Pipeline - Parallel StateGraph Implementation.

采用 Fan-Out / Fan-In 架构，并行执行三个专业分支：
1. Typology (形态溯源): 科技馆中庭的空间原型归纳
2. Benchmarking (精准对标): 基于规模的同量级案例分析
3. Trends (趋势分析): 近年中庭形态演变趋势

最终在 Assembly 节点整合为结构化 JSON 输出。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from config import CentralHubConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import CentralHubDataExtractor

logger = logging.getLogger(__name__)


# ==============================================================================
# Prompt Templates - 空间形态学导向
# ==============================================================================

class CentralHubPrompts:
    """Centralized prompt templates for Central Hub module."""

    # 通用形态学约束 (Negative Constraint)
    MORPHOLOGY_CONSTRAINT = """
【重要约束】
- 请聚焦于 **空间形态学** 层面：空间体量(Volume)、光环境(Light)、界面开放度(Interface)、流线关系(Connectivity)。
- **忽略**：具体电梯品牌、普通导视系统、非结构性装饰品，除非它们决定了空间形态。
"""

    # ========== Branch A: Typology ==========
    TYPOLOGY_SYSTEM = f"""你是一名专注于科技馆空间形态学的建筑策划顾问。
你的任务是归纳科技馆中庭/综合大厅的典型空间原型。
{MORPHOLOGY_CONSTRAINT}
请基于参考案例，提炼出 4-6 种典型的中庭形态原型。"""

    TYPOLOGY_USER = """请分析下方 Context 中的科技馆/博物馆中庭案例，归纳出 4-6 种典型的空间形态原型。

对于每种原型，请说明：
1. 原型名称（中英文）
2. 空间特征描述（体量、光环境、界面、流线）
3. 典型案例名称

Context:
{context}

输出格式：结构化的中文分析，便于后续整理为 JSON。"""

    # ========== Branch B: Benchmarking ==========
    BENCHMARKING_SYSTEM = f"""你是一名负责科技馆项目前期策划的建筑顾问。
你的任务是基于项目规模，分析同量级案例如何处理中庭空间的尺度感。
{MORPHOLOGY_CONSTRAINT}
重点关注：规模决定了中庭是做"巨大的震撼空间"还是"精致的导流空间"。"""

    BENCHMARKING_USER = """当前项目：{project_name}
项目规模：约 {total_area} 平方米
项目特征：{project_features}

请从下方 Context 中选择 3 个规模相近的案例，分析它们如何处理中庭空间的尺度感。

对于每个案例，请说明：
1. 案例名称
2. 与本项目相似的原因（规模、区位等）
3. 中庭形态特征（体量处理、光环境、界面开放度、流线组织）

Context:
{context}

输出格式：结构化的中文分析，便于后续整理为 JSON。"""

    # ========== Branch C: Trends ==========
    TRENDS_SYSTEM = f"""你是一名关注科技馆与科普建筑前沿趋势的建筑策划专家。
你的任务是分析近年（2019-2025）中庭形态设计的新趋势。
{MORPHOLOGY_CONSTRAINT}
关注新锐趋势，如"中庭即展厅"、"气候响应性中庭"、"去中心化公共空间"等。"""

    TRENDS_USER = """请分析下方 Context 中近年（2019-2025）竣工的科技馆/博物馆案例，总结中庭形态设计的最新趋势。

请重点关注：
1. 中庭与展览空间的复合化趋势
2. 生态/气候响应性设计
3. 去中心化与多核空间组织
4. 数字化与沉浸式体验的空间支撑

Context:
{context}

输出格式：趋势综述 + 2-4 条关键发展方向。"""

    # ========== Final Assembly ==========
    ASSEMBLY_SYSTEM = """你是一名严谨的建筑策划顾问，擅长将复杂分析整理为结构化 JSON。
你的任务是将三个分支的分析结果整合为一个严格符合 JSON 语法的对象。

必须满足以下要求：
1. 输出内容只能是一个 JSON 对象字符串，不要包含任何解释性文字、注释或 Markdown 代码块标记；
2. 字符串必须可以被 Python 的 json.loads 成功解析；
3. 键名和层级必须严格符合下述 Schema；
4. 所有字符串值必须使用双引号。

JSON Schema:
{{
  "morphology_panorama": {{
    "summary": "关于科技馆中庭空间形态的综述...",
    "common_archetypes": [
      {{"name": "空间原型名称", "description": "简述...", "example_case": "典型案例名"}}
    ]
  }},
  "benchmarking_analysis": {{
    "logic": "基于项目面积(X万平米)的对标分析...",
    "cases": [
      {{
        "case_name": "案例A",
        "similarity": "面积接近，均为城市中心型",
        "morphology_feature": "采用了垂直峡谷式设计，旨在..."
      }}
    ]
  }},
  "design_trends": {{
    "trend_summary": "近五年设计趋势分析...",
    "key_directions": ["趋势1: 生态化", "趋势2: 复合化"]
  }},
  "final_strategy": "基于上述分析，建议本项目中庭采用[某种形态]，理由是..."
}}"""

    ASSEMBLY_USER = """当前项目：{project_name}
项目规模：约 {total_area} 平方米

下面是三个分支的分析结果，请整合为一个 JSON 对象：

[形态溯源分析]
{typology_result}

[规模对标分析]
{benchmark_result}

[趋势洞察]
{trend_result}

请直接输出最终 JSON 字符串（不需要任何额外说明或 Markdown 标记）。"""


# ==============================================================================
# LangGraph State Definition
# ==============================================================================

class CentralHubState(TypedDict, total=False):
    """Graph state for Central Hub parallel pipeline."""
    # Inputs
    project_name: str
    project_features: str
    project_location: str
    total_area: float  # 从用户输入解析出的总面积 (平方米)

    # Branch Outputs (Independent storage - no conflict)
    typology_result: str     # node_typology 的中间生成结果
    benchmark_result: str    # node_benchmarking 的中间生成结果
    trend_result: str        # node_trends 的中间生成结果

    # Retrieved contexts (for debugging/dry_run)
    typology_contexts: List[Document]
    benchmark_contexts: List[Document]
    trend_contexts: List[Document]

    # Final Output
    final_json: Optional[Dict[str, Any]]


# ==============================================================================
# Vector Store
# ==============================================================================

class CentralHubVectorStore:
    """Vector store handler for central hub knowledge."""

    def __init__(self, config: CentralHubConfig):
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
            logger.info("已加载综合大厅索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("综合大厅文档为空，无法构建索引")
        logger.info("正在构建综合大厅索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("综合大厅索引保存至: %s", self.config.index_save_path)

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
        """Search with optional metadata filters."""
        if self.vectorstore is None:
            raise RuntimeError("综合大厅索引尚未构建")

        # 先获取更多候选，再做后过滤
        search_k = max(top_k * 3, 20) if filters else top_k
        raw_results = self.vectorstore.similarity_search(query, k=search_k)

        if not filters:
            return raw_results[:top_k]

        filtered = [doc for doc in raw_results if self._match_filters(doc, filters)]
        return filtered[:top_k] if filtered else raw_results[:top_k]

    @staticmethod
    def _match_filters(doc: Document, filters: Dict[str, Any]) -> bool:
        """Apply metadata filters."""
        meta = doc.metadata or {}

        # Area filter
        min_area = filters.get("min_area")
        max_area = filters.get("max_area")
        if min_area is not None or max_area is not None:
            area = CentralHubVectorStore._parse_area(meta)
            if area is not None:
                if min_area is not None and area < min_area:
                    return False
                if max_area is not None and area > max_area:
                    return False

        # Year filter
        min_year = filters.get("min_year")
        max_year = filters.get("max_year")
        if min_year is not None or max_year is not None:
            year = CentralHubVectorStore._parse_year(meta)
            if year is not None:
                if min_year is not None and year < min_year:
                    return False
                if max_year is not None and year > max_year:
                    return False

        return True

    @staticmethod
    def _parse_area(meta: Dict[str, Any]) -> Optional[float]:
        """Extract numeric area from metadata."""
        for key in ["total_construction_area_sqm", "total_construction_area", "Area", "area"]:
            value = meta.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                match = re.search(r"([\d,]+\.?\d*)", value.replace(",", ""))
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        pass
        return None

    @staticmethod
    def _parse_year(meta: Dict[str, Any]) -> Optional[int]:
        """Extract year from metadata."""
        for key in ["Year", "year", "opening_date", "开馆时间", "建成时间", "completion_year"]:
            value = meta.get(key)
            if value is None:
                continue
            if isinstance(value, (int, float)):
                return int(value)
            if isinstance(value, str):
                match = re.search(r"(19|20)\d{2}", value)
                if match:
                    return int(match.group(0))
        return None


# ==============================================================================
# Main Generator Class with LangGraph
# ==============================================================================

class CentralHubGenerator:
    """Parallel StateGraph-based generator for Central Hub spatial morphology analysis."""

    def __init__(self, config: CentralHubConfig):
        self.config = config
        self.extractor = CentralHubDataExtractor(config)
        self.vector_store = CentralHubVectorStore(config)
        self._llm_module: Optional[GenerationIntegrationModule] = None
        self._graph = self._build_graph()

    # ========== Infrastructure ==========
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

    # ========== Context Formatting ==========
    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "(未检索到参考内容)"
        formatted: List[str] = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"片段{idx}"
            year = CentralHubVectorStore._parse_year(meta)
            area = CentralHubVectorStore._parse_area(meta)
            year_str = f" | 年份:{year}" if year else ""
            area_str = f" | 面积:{area:.0f}㎡" if area else ""
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            formatted.append(f"【案例{idx} | 来源:{src} | 名称:{name}{year_str}{area_str}】\n{snippet}")
        return "\n\n".join(formatted)

    # ========== LangGraph Nodes ==========
    def _node_init(self, state: CentralHubState) -> Dict[str, Any]:
        """初始化节点：确保 LLM 就绪，准备并行执行。"""
        self._ensure_llm()
        logger.info("初始化节点完成，准备并行执行三个分析分支")
        return {}  # 不修改状态，仅触发后续并行节点

    def _node_typology(self, state: CentralHubState) -> Dict[str, Any]:
        """形态溯源节点：归纳科技馆中庭的典型空间原型。"""
        # LLM 已在 _node_init 中初始化
        logger.info("开始执行: 形态溯源分析")

        query = "科技馆 博物馆 中庭 综合大厅 空间形态 类型 原型"
        contexts = self.vector_store.search(query, top_k=self.config.top_k or 8)

        if not contexts:
            return {
                "typology_contexts": [],
                "typology_result": "No specific data found for typology analysis.",
            }

        context_text = self._format_context(contexts)
        prompt = CentralHubPrompts.TYPOLOGY_USER.format(context=context_text)

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", CentralHubPrompts.TYPOLOGY_SYSTEM),
            ("human", prompt),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        
        logger.info("完成: 形态溯源分析")
        return {
            "typology_contexts": contexts,
            "typology_result": str(raw),
        }

    def _node_benchmarking(self, state: CentralHubState) -> Dict[str, Any]:
        """规模对标节点：基于项目面积筛选同量级案例。"""
        logger.info("开始执行: 规模对标分析")

        total_area = state.get("total_area", 0)
        project_name = state.get("project_name", "")
        project_features = state.get("project_features", "")

        # 构建面积过滤器 (±30%)
        filters: Dict[str, Any] = {}
        if total_area and total_area > 0:
            delta = total_area * 0.3
            filters["min_area"] = max(total_area - delta, 0)
            filters["max_area"] = total_area + delta

        query = f"{project_name} 中庭 综合大厅 规模 尺度"
        contexts = self.vector_store.search(query, top_k=self.config.top_k or 8, filters=filters)

        # Fallback: 如果过滤后结果太少，放宽条件
        if len(contexts) < 3 and filters:
            logger.info("对标检索结果不足，放宽面积过滤条件")
            contexts = self.vector_store.search(query, top_k=self.config.top_k or 8)

        if not contexts:
            return {
                "benchmark_contexts": [],
                "benchmark_result": "No specific data found for benchmarking analysis.",
            }

        context_text = self._format_context(contexts)
        area_display = f"{total_area / 10000:.1f}万" if total_area > 10000 else f"{total_area:.0f}"
        prompt = CentralHubPrompts.BENCHMARKING_USER.format(
            project_name=project_name,
            total_area=area_display,
            project_features=project_features or "(未提供)",
            context=context_text,
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", CentralHubPrompts.BENCHMARKING_SYSTEM),
            ("human", prompt),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""

        logger.info("完成: 规模对标分析")
        return {
            "benchmark_contexts": contexts,
            "benchmark_result": str(raw),
        }

    def _node_trends(self, state: CentralHubState) -> Dict[str, Any]:
        """趋势分析节点：检索 2019-2025 年案例，分析最新趋势。"""
        logger.info("开始执行: 趋势分析")

        # 时间过滤：2019-2025
        filters = {"min_year": 2019, "max_year": 2025}
        query = "科技馆 博物馆 中庭 综合大厅 新建 趋势 设计"
        contexts = self.vector_store.search(query, top_k=self.config.top_k or 10, filters=filters)

        # Fallback: 如果结果太少，放宽到 2015-2025
        if len(contexts) < 3:
            logger.info("近年案例不足，扩大到 2015-2025 年")
            filters = {"min_year": 2015, "max_year": 2025}
            contexts = self.vector_store.search(query, top_k=self.config.top_k or 10, filters=filters)

        if not contexts:
            return {
                "trend_contexts": [],
                "trend_result": "No specific data found for trend analysis.",
            }

        context_text = self._format_context(contexts)
        prompt = CentralHubPrompts.TRENDS_USER.format(context=context_text)

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", CentralHubPrompts.TRENDS_SYSTEM),
            ("human", prompt),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""

        logger.info("完成: 趋势分析")
        return {
            "trend_contexts": contexts,
            "trend_result": str(raw),
        }

    def _node_assembly(self, state: CentralHubState) -> Dict[str, Any]:
        """最终整合节点：将三个分支结果整合为 JSON。"""
        logger.info("开始执行: 最终整合")

        project_name = state.get("project_name", "")
        total_area = state.get("total_area", 0)
        typology_result = state.get("typology_result", "")
        benchmark_result = state.get("benchmark_result", "")
        trend_result = state.get("trend_result", "")

        # 转义前置节点结果中的花括号，防止被 ChatPromptTemplate 误解为变量
        def escape_braces(s: str) -> str:
            return s.replace("{", "{{").replace("}", "}}")

        area_display = f"{total_area / 10000:.1f}万" if total_area > 10000 else f"{total_area:.0f}"
        prompt = CentralHubPrompts.ASSEMBLY_USER.format(
            project_name=project_name,
            total_area=area_display,
            typology_result=escape_braces(typology_result) if typology_result else "(无数据)",
            benchmark_result=escape_braces(benchmark_result) if benchmark_result else "(无数据)",
            trend_result=escape_braces(trend_result) if trend_result else "(无数据)",
        )

        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", CentralHubPrompts.ASSEMBLY_SYSTEM),
            ("human", prompt),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or "{}"
        text = str(raw).strip()

        # 尝试解析 JSON，清理 Markdown 包裹
        for pattern in [
            r"^```json\s*(.*)```$",
            r"^```\s*(.*)```$",
        ]:
            m = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                break

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("综合大厅最终节点返回的 JSON 解析失败，将返回原始文本")
            parsed = {"raw_response": text}

        return {"final_json": parsed}

    # ========== Graph Construction ==========
    def _build_graph(self):
        """Build true parallel Fan-Out / Fan-In topology.
        
        Graph structure:
            init ──┬── typology ────┐
                   ├── benchmarking ┼── assembly ── END
                   └── trends ──────┘
        
        All three analysis nodes run in PARALLEL after init.
        """
        graph = StateGraph(CentralHubState)

        # Add nodes
        graph.add_node("init", self._node_init)
        graph.add_node("typology", self._node_typology)
        graph.add_node("benchmarking", self._node_benchmarking)
        graph.add_node("trends", self._node_trends)
        graph.add_node("assembly", self._node_assembly)

        # True Fan-Out: init -> [typology, benchmarking, trends] in parallel
        graph.set_entry_point("init")
        graph.add_edge("init", "typology")
        graph.add_edge("init", "benchmarking")
        graph.add_edge("init", "trends")
        
        # Fan-In: all three analysis nodes -> assembly
        graph.add_edge("typology", "assembly")
        graph.add_edge("benchmarking", "assembly")
        graph.add_edge("trends", "assembly")
        
        graph.add_edge("assembly", END)

        return graph.compile()

    # ========== Public Interface ==========
    def generate(
        self,
        project_name: str,
        project_features: str,
        total_area: Optional[float] = None,
        project_location: Optional[str] = None,
        rebuild_index: bool = False,
        dry_run: bool = False,
        # Legacy parameters for backward compatibility
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行并行 Fan-Out/Fan-In 图，生成综合大厅空间形态分析 JSON。

        Args:
            project_name: 项目名称
            project_features: 项目特征描述
            total_area: 总建筑面积 (平方米)
            project_location: 项目区位
            rebuild_index: 是否重建索引
            dry_run: 仅预览检索结果，不调用 LLM

        Returns:
            包含 json_result 或中间结果的字典
        """
        if not project_name:
            raise ValueError("project_name 不能为空")

        self.ensure_index(rebuild=rebuild_index)

        # 初始化状态
        init_state: CentralHubState = {
            "project_name": project_name,
            "project_features": project_features or "",
            "project_location": project_location or "",
            "total_area": total_area or 0,
        }

        final_state = self._graph.invoke(init_state)

        if isinstance(final_state, dict):
            json_result = final_state.get("final_json")

            if dry_run:
                return {
                    "typology_contexts": final_state.get("typology_contexts", []),
                    "benchmark_contexts": final_state.get("benchmark_contexts", []),
                    "trend_contexts": final_state.get("trend_contexts", []),
                    "typology_result": final_state.get("typology_result", ""),
                    "benchmark_result": final_state.get("benchmark_result", ""),
                    "trend_result": final_state.get("trend_result", ""),
                }

            # 为了兼容旧接口，同时返回 response 字段
            response_str = json.dumps(json_result, ensure_ascii=False, indent=2) if json_result else ""
            return {
                "json_result": json_result,
                "response": response_str,
                "raw_text": {
                    "typology_result": final_state.get("typology_result", ""),
                    "benchmark_result": final_state.get("benchmark_result", ""),
                    "trend_result": final_state.get("trend_result", ""),
                },
            }

        return {"json_result": None, "error": "Unexpected state type"}


# ==============================================================================
# Backward Compatibility
# ==============================================================================

class CentralHubPromptBuilder:
    """Legacy prompt builder - kept for backward compatibility."""
    pass

