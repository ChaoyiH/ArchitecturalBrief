"""Pipeline for generating special effects theater planning briefs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import SpecialTheaterConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import SpecialTheaterDataExtractor
from core.generation_integration import GenerationIntegrationModule
from prompts import (
    SPECIAL_THEATER_SYSTEM,
    SPECIAL_THEATER_JSON_SCHEMA,
    SPECIAL_THEATER_USER_TEMPLATE,
)

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
    """Builds prompts for special theater design using centralized prompt registry."""

    def build_prompt(
        self,
        project_name: str,
        project_features: str,
        retrieved_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context = self._format_context(retrieved_docs)
        # 使用 prompts.py 中的模板，转义花括号以兼容 ChatPromptTemplate
        json_schema_escaped = SPECIAL_THEATER_JSON_SCHEMA.replace("{", "{{").replace("}", "}}")
        user_prompt = SPECIAL_THEATER_USER_TEMPLATE.format(
            project_name=project_name,
            project_features=project_features,
            context=context,
            json_schema=json_schema_escaped,
        )
        # 转义 user_prompt 中的花括号
        user_prompt = user_prompt.replace("{", "{{").replace("}", "}}")

        return {
            "system_prompt": SPECIAL_THEATER_SYSTEM,
            "user_prompt": user_prompt,
        }

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


class SpecialTheaterGenerator:
    """Facade for special theater generation."""

    def __init__(self, config: SpecialTheaterConfig):
        self.config = config
        self.extractor = SpecialTheaterDataExtractor(config)
        self.vector_store = SpecialTheaterVectorStore(config)
        self.prompt_builder = SpecialTheaterPromptBuilder()
        self._llm_module: Optional[GenerationIntegrationModule] = None

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

    def _expand_queries(
        self,
        project_name: str,
        project_features: str,
        base_query: Optional[str],
    ) -> List[str]:
        base = base_query or project_name or project_features or "特效影院"
        seeds = [
            base,
            f"{project_name} 特效影院 配置" if project_name else "科技馆 特效影院 配置",
            "球幕影院 建筑高度 要求",
            "IMAX影院 座位数 案例",
            "博物馆 4D影院 设计规范",
            f"{project_features} 影院 组合" if project_features else "特效影院 组合",
        ]
        unique: List[str] = []
        seen = set()
        for term in seeds:
            cleaned = term.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                unique.append(cleaned)
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
        contexts = self.retrieve_contexts(project_name, project_features, query, top_k)
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
