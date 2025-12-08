"""Facade-style entrypoint for hybrid QA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DEFAULT_CONFIG
from core.generation_integration import GenerationIntegrationModule
from .case_engine import CaseAnalyst, CaseResult
from .norm_engine import NormRetriever
from .router import QAIntent, Router

logger = logging.getLogger(__name__)


class QAPipeline:
    """Hybrid QA pipeline combining case + norm retrieval."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.llm_provider = llm_provider or DEFAULT_CONFIG.llm_provider
        self.llm_model = llm_model or DEFAULT_CONFIG.llm_model

        self.router = Router(self.llm_provider, self.llm_model)
        self.case_engine = CaseAnalyst()
        self.norm_engine = NormRetriever()
        self.generator = GenerationIntegrationModule(
            provider=self.llm_provider,
            model_name=self.llm_model,
            temperature=0.2,
            max_tokens=2048,
        )

    def answer(self, query: str) -> str:
        intent = self.router.classify(query)

        case_result: Optional[CaseResult] = None
        norm_docs: List[Document] = []

        if intent in (QAIntent.INTENT_CASE, QAIntent.INTENT_HYBRID):
            case_result = self.case_engine.query(query)

        if intent in (QAIntent.INTENT_NORM, QAIntent.INTENT_HYBRID):
            norm_docs = self.norm_engine.retrieve(query, top_k=5)

        return self._synthesize(query, intent, case_result, norm_docs)

    # ------------------------------------------------------------------
    # Synthesizer
    # ------------------------------------------------------------------
    def _synthesize(
        self,
        query: str,
        intent: QAIntent,
        case_result: Optional[CaseResult],
        norm_docs: List[Document],
    ) -> str:
        case_context = self._format_case_context(case_result)
        norm_context = self._format_norm_context(norm_docs)

        system_prompt = (
            "你是一名严谨的建筑策划师。请基于提供的上下文回答。"
            "引用数据时必须说明来源（是规范还是案例）。若无依据，请回答不知道。"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "用户问题: {question}\n\n"
                    "【案例事实】\n{case_context}\n\n"
                    "【规范依据】\n{norm_context}\n\n"
                    "请结合上下文回答，缺失信息请说明。",
                ),
            ]
        )

        chain = prompt | self.generator.llm | StrOutputParser()
        try:
            return chain.invoke(
                {
                    "question": query,
                    "case_context": case_context or "(暂无案例匹配)",
                    "norm_context": norm_context or "(暂无规范匹配)",
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("QA 生成失败: %s", exc)
            return "生成回答时出现问题，请稍后再试。"

    @staticmethod
    def _format_case_context(case_result: Optional[CaseResult]) -> str:
        if not case_result:
            return ""

        lines: List[str] = []
        if case_result.records:
            for idx, item in enumerate(case_result.records, 1):
                lines.append(
                    f"[案例{idx}] {item.get('project_name', '')} | {item.get('location', '')} | "
                    f"面积: {item.get('area', '')} ㎡ | 来源: {Path(item.get('source', '')).name}"
                )

        docs = case_result.documents[:5] if case_result.documents else []
        for doc in docs:
            meta = doc.metadata or {}
            name = meta.get("project_name") or meta.get("doc_name") or meta.get("source")
            source = Path(meta.get("source", "")).name if meta.get("source") else meta.get("doc_name", "")
            lines.append(f"[案例内容] {name} ({source})\n{doc.page_content}")

        return "\n".join(lines)

    @staticmethod
    def _format_norm_context(norm_docs: List[Document]) -> str:
        if not norm_docs:
            return ""

        lines: List[str] = []
        for idx, doc in enumerate(norm_docs, 1):
            meta = doc.metadata or {}
            source = meta.get("doc_name") or Path(meta.get("source", "")).name
            lines.append(f"[规范{idx}] {source}\n{doc.page_content}")
        return "\n\n".join(lines)


__all__ = ["QAPipeline"]
