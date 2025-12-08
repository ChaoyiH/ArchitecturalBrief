"""Normative retrieval engine constrained to gb/zlj corpora."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config import DEFAULT_CONFIG, DATA_ROOT
from core.index_construction import IndexConstructionModule
from core.retrieval_optimization import RetrievalOptimizationModule
from utils.data_preparation import DataPreparationModule

logger = logging.getLogger(__name__)


class NormRetriever:
    """Retrieve standards/requirements with scope limited to gb/zlj datasets."""

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self.data_paths = [str(DATA_ROOT / "gb"), str(DATA_ROOT / "zlj")]
        self.index_path = index_path or DEFAULT_CONFIG.index_save_path
        self.embedding_model = embedding_model or DEFAULT_CONFIG.embedding_model

        self.data_module = DataPreparationModule(self.data_paths)
        self.index_module = IndexConstructionModule(
            model_name=self.embedding_model,
            index_save_path=self.index_path,
        )
        self._retriever: Optional[RetrievalOptimizationModule] = None
        self._chunks: Optional[List[Document]] = None

    def retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        self._ensure_retriever()
        return self._retriever.hybrid_search(question, top_k=top_k) if self._retriever else []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_retriever(self) -> None:
        if self._retriever is not None:
            return

        chunks = self._load_chunks()
        vectorstore = self.index_module.load_index()
        if vectorstore is None:
            logger.info("未找到既有规范索引，使用当前 chunks 构建临时索引")
            vectorstore = self.index_module.build_vector_index(chunks)

        self._retriever = RetrievalOptimizationModule(vectorstore, chunks)

    def _load_chunks(self) -> List[Document]:
        if self._chunks is not None:
            return self._chunks

        logger.info("加载 gb/zlj 规范文档并进行分块...")
        self.data_module.load_documents()
        self._chunks = self.data_module.chunk_documents()
        return self._chunks


__all__ = ["NormRetriever"]
