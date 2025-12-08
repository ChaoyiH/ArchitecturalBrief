"""Case analysis engine for hybrid structured/semantic queries."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DEFAULT_CONFIG
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from core.retrieval_optimization import RetrievalOptimizationModule
from utils.json_data_loader import load_case_database

logger = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Container for case retrieval outputs."""

    mode: str
    records: List[Dict[str, Any]]
    documents: List[Document]
    query_expr: Optional[str] = None


class CaseAnalyst:
    """Handle factual/statistical queries against case datasets."""

    STRUCTURE_KEYWORDS = [
        "最大",
        "最小",
        "排名",
        "top",
        "TOP",
        "面积",
        "year",
        "年份",
        "地点",
        "城市",
        "超过",
        "以上",
        "以下",
        "大于",
        "小于",
    ]
    SEMANTIC_HINTS = ["流线", "立面", "材质", "设计", "体验", "布局", "动线", " facade", "material"]

    def __init__(
        self,
        embedding_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> None:
        self.df: pd.DataFrame = load_case_database().copy()
        self.embedding_model = embedding_model or DEFAULT_CONFIG.embedding_model
        self.llm_provider = llm_provider or DEFAULT_CONFIG.llm_provider
        self.llm_model = llm_model or DEFAULT_CONFIG.llm_model

        self._llm_module: Optional[GenerationIntegrationModule] = None
        self._documents: Optional[List[Document]] = None
        self._retriever: Optional[RetrievalOptimizationModule] = None

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def query(self, user_input: str) -> CaseResult:
        """Run structured/semantic/hybrid query."""
        use_structured = self._should_use_structured(user_input)
        semantic_needed = self._should_use_semantic(user_input)

        if use_structured:
            structured = self._run_structured_query(user_input)

            if not structured.records:
                logger.warning("Structured query yielded no results, falling back to semantic search.")
                semantic_docs = self._semantic_search(user_input, None)
                return CaseResult(
                    mode="semantic_fallback",
                    records=[],
                    documents=semantic_docs,
                    query_expr=structured.query_expr,
                )

            if semantic_needed and structured.documents:
                # Hybrid: semantic re-ranking within structured subset
                semantic_docs = self._semantic_search(user_input, structured.documents)
                return CaseResult(
                    mode="hybrid",
                    records=structured.records,
                    documents=semantic_docs,
                    query_expr=structured.query_expr,
                )
            return structured

        # Pure semantic branch
        semantic_docs = self._semantic_search(user_input, None)
        return CaseResult(mode="semantic", records=[], documents=semantic_docs, query_expr=None)

    # ------------------------------------------------------------------
    # Structured query branch
    # ------------------------------------------------------------------
    def _run_structured_query(self, user_input: str) -> CaseResult:
        expr = self._build_pandas_query(user_input)
        if not expr or not self._is_safe_query(expr):
            logger.warning("LLM 生成的查询表达式为空或不安全，跳过结构化检索: %s", expr)
            return CaseResult(mode="structured", records=[], documents=[], query_expr=expr)

        try:
            filtered = self.df.query(expr, engine="python")
        except Exception as exc:  # noqa: BLE001
            logger.warning("执行 Pandas 查询失败 (%s)，表达式: %s", exc, expr)
            filtered = pd.DataFrame(columns=self.df.columns)

        records = self._df_to_records(filtered.head(20))
        documents = self._df_to_documents(filtered.head(50))
        return CaseResult(mode="structured", records=records, documents=documents, query_expr=expr)

    def _build_pandas_query(self, user_input: str) -> Optional[str]:
        self._ensure_llm()
        prompt = ChatPromptTemplate.from_template(
            """
你是一名数据筛选助手。请根据用户问题生成一段 Pandas DataFrame 的 query 表达式，
仅使用以下列名：project_name, location, architects, year, area。

规则：
- 只输出一行表达式，不要解释。
- 如需字符串匹配，可使用 str.contains('关键词', case=False)。
- area 为浮点数（单位 m2），year 可为字符串。
用户问题: {question}
            """
        )
        chain = prompt | self._llm_module.llm | StrOutputParser()
        try:
            expr = chain.invoke({"question": user_input}).strip()
            return expr.splitlines()[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("生成 Pandas query 失败: %s", exc)
            return None

    @staticmethod
    def _is_safe_query(expr: str) -> bool:
        lowered = expr.lower()
        banned = ["__", "import", "eval", "exec", "os.", "sys.", "@", "open("]
        if any(b in lowered for b in banned):
            return False
        return bool(re.fullmatch(r"[\w\s\'\"\&\|\<\>\=\.\-\(\)\,\~]+", expr))

    # ------------------------------------------------------------------
    # Semantic query branch
    # ------------------------------------------------------------------
    def _semantic_search(self, user_input: str, docs: Optional[List[Document]]) -> List[Document]:
        target_docs = docs or self._ensure_documents()
        if not target_docs:
            return []

        if docs is None:
            # Reuse cached retriever for full corpus
            retriever = self._ensure_retriever()
        else:
            # Build a temporary retriever for the narrowed subset
            embedding = get_embedding(self.embedding_model)
            vectorstore = FAISS.from_documents(target_docs, embedding=embedding)
            retriever = RetrievalOptimizationModule(vectorstore, target_docs)

        return retriever.hybrid_search(user_input, top_k=5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _should_use_structured(self, question: str) -> bool:
        lowered = question.lower()
        return any(k.lower() in lowered for k in self.STRUCTURE_KEYWORDS) or bool(
            re.search(r"\d", question)
        )

    def _should_use_semantic(self, question: str) -> bool:
        lowered = question.lower()
        return any(k.lower() in lowered for k in self.SEMANTIC_HINTS)

    def _ensure_llm(self) -> None:
        if self._llm_module is None:
            self._llm_module = GenerationIntegrationModule(
                provider=self.llm_provider,
                model_name=self.llm_model,
                temperature=0.1,
                max_tokens=2048,
            )

    def _ensure_documents(self) -> List[Document]:
        if self._documents is not None:
            return self._documents

        docs: List[Document] = []
        for row in self.df.itertuples():
            content_parts = [
                f"项目: {getattr(row, 'project_name', '')}",
                f"地点: {getattr(row, 'location', '')}",
                f"设计: {getattr(row, 'architects', '')}",
                f"年份: {getattr(row, 'year', '')}",
                f"面积: {getattr(row, 'area', '')} ㎡",
                f"描述: {getattr(row, 'description', '')}",
            ]
            doc = Document(
                page_content=" | ".join(content_parts),
                metadata={
                    "project_name": getattr(row, "project_name", ""),
                    "location": getattr(row, "location", ""),
                    "architects": getattr(row, "architects", ""),
                    "year": getattr(row, "year", ""),
                    "area": getattr(row, "area", None),
                    "source": getattr(row, "source", ""),
                    "source_type": "case",
                },
            )
            docs.append(doc)

        self._documents = docs
        return docs

    def _ensure_retriever(self) -> RetrievalOptimizationModule:
        if self._retriever is not None:
            return self._retriever
        embedding = get_embedding(self.embedding_model)
        vectorstore = FAISS.from_documents(self._ensure_documents(), embedding=embedding)
        self._retriever = RetrievalOptimizationModule(vectorstore, self._ensure_documents())
        return self._retriever

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            records.append(
                {
                    "project_name": row.get("project_name", ""),
                    "location": row.get("location", ""),
                    "architects": row.get("architects", ""),
                    "year": row.get("year", ""),
                    "area": row.get("area", None),
                    "source": row.get("source", ""),
                }
            )
        return records

    @staticmethod
    def _df_to_documents(df: pd.DataFrame) -> List[Document]:
        docs: List[Document] = []
        for _, row in df.iterrows():
            content = (
                f"项目: {row.get('project_name', '')} | 地点: {row.get('location', '')} | "
                f"设计: {row.get('architects', '')} | 年份: {row.get('year', '')} | "
                f"面积: {row.get('area', '')} ㎡ | 描述: {row.get('description', '')}"
            )
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "project_name": row.get("project_name", ""),
                        "location": row.get("location", ""),
                        "architects": row.get("architects", ""),
                        "year": row.get("year", ""),
                        "area": row.get("area", None),
                        "source": row.get("source", ""),
                        "source_type": "case",
                    },
                )
            )
        return docs


__all__ = ["CaseAnalyst", "CaseResult"]
