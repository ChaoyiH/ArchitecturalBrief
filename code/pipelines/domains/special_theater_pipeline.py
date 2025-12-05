"""Special effects theater planning pipeline with structured parallel nodes."""

from __future__ import annotations

import logging
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from config import SpecialTheaterConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import SpecialTheaterDataExtractor
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class SpecialTheaterVectorStore:
    """Vector store wrapper for special theater knowledge."""

    def __init__(self, config: SpecialTheaterConfig):
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
            logger.info("已加载特效影院索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("特效影院文档为空，无法构建索引")
        logger.info("正在构建特效影院索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("特效影院索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("特效影院索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class SpecialTheaterPromptBuilder:
    """Builds structured prompts for typology, trends, suitability."""

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "(未检索到参考内容)"
        formatted: List[str] = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"片段{idx}"
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            snippet = snippet.replace("{", "{{").replace("}", "}}")
            formatted.append(f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}")
        return "\n".join(formatted)

    def build_typology_prompt(self, project_name: str, project_features: str, docs: Sequence[Document]) -> Dict[str, str]:
        ctx = self._format_context(docs)
        system = (
            "You are a Cinema Technology Consultant for a Science Museum."
            " Identify common special theater typologies from evidence. Output JSON only."
        )
        user_raw = (
            "从 Context 提取特效影厅类型，输出 JSON：\n"
            "{\n"
            "  \"summary\": \"30-60字概述主流类型\",\n"
            "  \"types\": [\n"
            "    {\"name\": \"类型名\", \"description\": \"用途/特征\", \"evidence\": \"原文摘录\"}\n"
            "  ]\n"
            "}\n\n"
            "约束：必须是有效 JSON；缺失填 Not Specified；引用必须来自 Context。\n\n"
            f"Context:\n{ctx}"
        )
        user = self._escape_braces(user_raw)
        return {"system_prompt": system, "user_prompt": user}

    def build_trend_prompt(self, project_name: str, project_features: str, docs: Sequence[Document]) -> Dict[str, str]:
        ctx = self._format_context(docs)
        system = (
            "You are a Cinema Technology Consultant for a Science Museum."
            " Analyze trends; separate older (<2015) vs newer (>=2015). Output JSON only."
        )
        user_raw = (
            "输出 JSON：\n"
            "{\n"
            "  \"trend_list\": [\n"
            "    {\"trend_name\": \"...\", \"description\": \"50-80字\", \"evidence_case\": \"案例+年份\", \"time_window\": \"<2015 或 >=2015\"}\n"
            "  ],\n"
            "  \"actionable_strategies\": [\"结合趋势的可实施策略\"]\n"
            "}\n\n"
            "若检索不足，写 'Trend data is insufficient based on current retrieval'.\n"
            f"Context:\n{ctx}"
        )
        user = self._escape_braces(user_raw)
        return {"system_prompt": system, "user_prompt": user}

    def build_suitability_prompt(self, project_name: str, project_features: str, docs: Sequence[Document]) -> Dict[str, str]:
        ctx = self._format_context(docs)
        system = (
            "You are a Cinema Technology Consultant for a Science Museum."
            " Recommend theater mix for the project. Output JSON only."
        )
        user_raw = (
            f"项目：{project_name or '未命名'} | 特征：{project_features or '未提供'}\n"
            "基于 Context，输出 JSON：\n"
            "{\n"
            "  \"recommendations\": [\n"
            "    {\"theater_type\": \"类型\", \"recommended_capacity\": \"座位/规模\", \"core_function\": \"定位\", \"rationale\": \"基于竞品/定位的理由\"}\n"
            "  ],\n"
            "  \"spatial_requirements\": [\"层高/跨度/隔音等关键要求\"]\n"
            "}\n\n"
            "要求：必须有效 JSON；理由必须引用 Context 证据或说明缺失。\n\n"
            f"Context:\n{ctx}"
        )
        user = self._escape_braces(user_raw)
        return {"system_prompt": system, "user_prompt": user}


class SpecialTheaterGenerator:
    """Parallel graph for special theater typology/trend/suitability with Python assembly."""

    class _State(TypedDict, total=False):
        project_name: str
        project_features: str
        query: str
        # retrieval
        retrieved_docs: List[Document]
        # branch outputs
        typology_data: Dict[str, Any]
        trend_data: Dict[str, Any]
        suitability_data: Dict[str, Any]
        # final
        final_markdown: str

    def __init__(self, config: SpecialTheaterConfig):
        self.config = config
        self.extractor = SpecialTheaterDataExtractor(config)
        self.vector_store = SpecialTheaterVectorStore(config)
        self.prompt_builder = SpecialTheaterPromptBuilder()
        self._llm_module: Optional[GenerationIntegrationModule] = None
        self._graph = self._build_graph()

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

    @staticmethod
    def _strip_json_markers(text: str) -> str:
        cleaned = text.strip()
        for pattern in [r"^```json\s*(.*)```$", r"^```\s*(.*)```$"]:
            m = re.match(pattern, cleaned, flags=re.DOTALL | re.IGNORECASE)
            if m:
                cleaned = m.group(1).strip()
                break
        return cleaned

    @classmethod
    def _safe_json_loads(cls, text: str, fallback: Any) -> Any:
        cleaned = cls._strip_json_markers(text)
        try:
            parsed = json.loads(cleaned)
            return parsed
        except Exception:
            logger.warning("JSON 解析失败，返回 fallback")
            return fallback

    def _expand_queries(self, project_name: str, project_features: str, base_query: Optional[str]) -> List[str]:
        base = base_query or project_name or project_features or "特效影院"
        tech_terms = [
            "IMAX", "Giant Screen", "巨幕", "Dome", "Planetarium", "球幕", "天象厅",
            "4D", "5D", "Motion Ride", "动感影院", "Flying Theater", "飞行影院",
            "VR", "AR", "Immersive", "沉浸式", "激光", "8K"
        ]
        seeds = [
            base,
            f"{base} 特效影厅",
            f"{project_name} 特效影院 配置" if project_name else "科技馆 特效影院 配置",
            f"{project_features} 影厅 类型" if project_features else "科普馆 影厅 类型",
        ]
        for term in tech_terms:
            seeds.append(f"特效影院 {term}")
        unique = []
        seen = set()
        for s in seeds:
            c = s.strip()
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def retrieve_contexts(
        self,
        project_name: str,
        project_features: str,
        query: Optional[str],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        queries = self._expand_queries(project_name, project_features, query)
        collected: Dict[str, Document] = {}
        limit = top_k or self.config.top_k
        for q in queries:
            docs = self.vector_store.search(q, top_k=limit)
            for doc in docs:
                key = doc.metadata.get("chunk_id") or doc.metadata.get("source") or doc.page_content[:50]
                if key not in collected:
                    collected[key] = doc
            if len(collected) >= limit:
                break
        return list(collected.values())[:limit]

    # ========== Graph Nodes ==========
    def _node_init(self, state: "SpecialTheaterGenerator._State") -> Dict[str, Any]:
        self._ensure_llm()
        contexts = self.retrieve_contexts(
            state.get("project_name", ""),
            state.get("project_features", ""),
            state.get("query", None),
            self.config.top_k,
        )
        return {"retrieved_docs": contexts}

    def _node_typology(self, state: "SpecialTheaterGenerator._State") -> Dict[str, Any]:
        prompt = self.prompt_builder.build_typology_prompt(
            state.get("project_name", ""),
            state.get("project_features", ""),
            state.get("retrieved_docs", []),
        )
        chat = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        parsed = self._safe_json_loads(str(raw), {"summary": "Not Specified", "types": []})
        if not isinstance(parsed, dict):
            parsed = {"summary": "Not Specified", "types": []}
        return {"typology_data": parsed}

    def _node_trends(self, state: "SpecialTheaterGenerator._State") -> Dict[str, Any]:
        prompt = self.prompt_builder.build_trend_prompt(
            state.get("project_name", ""),
            state.get("project_features", ""),
            state.get("retrieved_docs", []),
        )
        chat = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        parsed = self._safe_json_loads(str(raw), {"trend_list": [], "actionable_strategies": []})
        if not isinstance(parsed, dict):
            parsed = {"trend_list": [], "actionable_strategies": []}
        return {"trend_data": parsed}

    def _node_suitability(self, state: "SpecialTheaterGenerator._State") -> Dict[str, Any]:
        prompt = self.prompt_builder.build_suitability_prompt(
            state.get("project_name", ""),
            state.get("project_features", ""),
            state.get("retrieved_docs", []),
        )
        chat = ChatPromptTemplate.from_messages([
            ("system", prompt["system_prompt"]),
            ("human", prompt["user_prompt"]),
        ])
        chain = chat | self._llm_module.llm | StrOutputParser()
        raw = chain.invoke({}) or ""
        fallback = {"recommendations": [], "spatial_requirements": []}
        parsed = self._safe_json_loads(str(raw), fallback)
        if not isinstance(parsed, dict):
            parsed = fallback
        return {"suitability_data": parsed}

    def _node_assembly(self, state: "SpecialTheaterGenerator._State") -> Dict[str, Any]:
        typ = state.get("typology_data", {"summary": "Not Specified", "types": []})
        trend = state.get("trend_data", {"trend_list": [], "actionable_strategies": []})
        suit = state.get("suitability_data", {"recommendations": [], "spatial_requirements": []})

        def fmt_types(types: List[Dict[str, Any]]) -> str:
            lines = []
            for t in types:
                name = t.get("name", "类型未明")
                desc = t.get("description", "Not Specified")
                ev = t.get("evidence", "")
                line = f"- **{name}**：{desc}"
                if ev:
                    line += f"；Evidence：{ev}"
                lines.append(line)
            return "\n".join(lines) if lines else "- Not Specified"

        def fmt_trends(trend_list: List[Dict[str, Any]]) -> str:
            lines = []
            for t in trend_list:
                lines.append(f"- **{t.get('trend_name','趋势未明')}** ({t.get('time_window','未标明')}): {t.get('description','Not Specified')} | Evidence: {t.get('evidence_case','Not Specified')}")
            return "\n".join(lines) if lines else "- Trend data is insufficient based on current retrieval"

        def fmt_table(recs: List[Dict[str, Any]]) -> str:
            if not recs:
                return "| 影厅类型 | 建议座位数/规模 | 核心功能定位 | 推荐理由 (基于竞品分析) |\n| :--- | :--- | :--- | :--- |\n| Not Specified | Not Specified | Not Specified | Not Specified |"
            rows = ["| 影厅类型 | 建议座位数/规模 | 核心功能定位 | 推荐理由 (基于竞品分析) |", "| :--- | :--- | :--- | :--- |"]
            for r in recs:
                rows.append(
                    f"| {r.get('theater_type','未明')} | {r.get('recommended_capacity','Not Specified')} | {r.get('core_function','Not Specified')} | {r.get('rationale','Not Specified')} |"
                )
            return "\n".join(rows)

        md = [
            "## 4. 特效影厅体系规划 (Special Theaters Planning)",
            "", "### 4.1 行业主流影厅类型分析", typ.get("summary", "Not Specified"), fmt_types(typ.get("types", [])),
            "", "### 4.2 建设趋势洞察", fmt_trends(trend.get("trend_list", [])),
            "", "### 4.3 济南科技馆影厅配置建议", fmt_table(suit.get("recommendations", [])),
            "", "### 4.4 空间技术要求概要", "- " + "\n- ".join(suit.get("spatial_requirements", []) or ["层高/跨度/隔音要求：Not Specified"]),
            "", f"生成时间: {datetime.now().isoformat()}",
        ]

        return {"final_markdown": "\n".join(md)}

    def _build_graph(self):
        graph = StateGraph(self._State)
        graph.add_node("init", self._node_init)
        graph.add_node("typology", self._node_typology)
        graph.add_node("trends", self._node_trends)
        graph.add_node("suitability", self._node_suitability)
        graph.add_node("assembly", self._node_assembly)

        graph.set_entry_point("init")
        # 并行执行三条分支，assembly 会在最后一次触发时得到完整数据
        graph.add_edge("init", "typology")
        graph.add_edge("init", "trends")
        graph.add_edge("init", "suitability")
        graph.add_edge("typology", "assembly")
        graph.add_edge("trends", "assembly")
        graph.add_edge("suitability", "assembly")
        graph.add_edge("assembly", END)
        return graph.compile()

    def generate(
        self,
        project_name: str,
        project_features: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        rebuild_index: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        self.ensure_index(rebuild=rebuild_index)
        init_state: SpecialTheaterGenerator._State = {
            "project_name": project_name,
            "project_features": project_features,
            "query": query or "",
        }
        final_state = self._graph.invoke(init_state)

        if dry_run:
            return {
                "typology_data": final_state.get("typology_data"),
                "trend_data": final_state.get("trend_data"),
                "suitability_data": final_state.get("suitability_data"),
                "retrieved_docs": final_state.get("retrieved_docs"),
            }

        # 返回 response 以兼容 design_generator 打印逻辑
        return {
            "response": final_state.get("final_markdown", ""),
            "json": {
                "typology_data": final_state.get("typology_data"),
                "trend_data": final_state.get("trend_data"),
                "suitability_data": final_state.get("suitability_data"),
            },
        }
