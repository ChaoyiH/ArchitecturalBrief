"""Pipeline for planning science & education integrations."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import ScienceEducationConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import ScienceEducationDataExtractor
from core.generation_integration import GenerationIntegrationModule
from prompts import (
    SCIENCE_EDUCATION_SYSTEM,
    SCIENCE_EDUCATION_JSON_SCHEMA,
    SCIENCE_EDUCATION_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class ScienceEducationVectorStore:
    """Vector store for science & education knowledge."""

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

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("科教活动索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class ScienceEducationPromptBuilder:
    SYSTEM_PROMPT = SCIENCE_EDUCATION_SYSTEM

    JSON_SCHEMA = (
        "{\n"
        "  \"education_concept\": \"一句话概括教育理念（如：从'参观'走向'探究'，馆校深度融合）。\",\n"
        "  \"signature_activities\": [\n"
        "    {\n"
        "      \"name\": \"建议活动名称（如：奇妙化学实验秀）\",\n"
        "      \"format\": \"活动形式（如：现场演示/互动体验）\",\n"
        "      \"spatial_requirement\": \"对空间的要求（如：需配有排风设施的开放舞台，或需大跨度中庭）。\"\n"
        "    },\n"
        "    {\n"
        "      \"name\": \"...\",\n"
        "      \"format\": \"...\",\n"
        "      \"spatial_requirement\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"spatial_integration\": {\n"
        "    \"embedded_labs\": \"关于在展厅内设置‘玻璃盒子’实验室或开放工坊的建议。\",\n"
        "    \"public_performance\": \"关于利用门厅/中庭进行科学表演的空间利用策略。\"\n"
        "  },\n"
        "  \"dedicated_education_zone\": {\n"
        "    \"room_configuration\": \"独立教育区建议设置的房间类型及数量（如：2间通用教室，1间机器人工作室）。\",\n"
        "    \"zoning_strategy\": \"教育区在建筑中的位置建议（如：独立首层入口，方便夜间或周末单独开放）。\"\n"
        "  }\n"
        "}\n"
    )

    def build_prompt(
        self,
        project_name: str,
        project_features: str,
        retrieved_docs: Sequence[Document],
    ) -> Dict[str, str]:
        context = self._format_context(retrieved_docs)
        escaped_schema = self.JSON_SCHEMA.replace("{", "{{").replace("}", "}}")
        user_prompt = SCIENCE_EDUCATION_USER_TEMPLATE.format(
            project_name=project_name,
            project_features=project_features,
            context=context,
            json_schema=escaped_schema,
        )
        return {
            "system_prompt": self.SYSTEM_PROMPT,
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


class ScienceEducationGenerator:
    """Facade for science education planning."""

    def __init__(self, config: ScienceEducationConfig):
        self.config = config
        self.extractor = ScienceEducationDataExtractor(config)
        self.vector_store = ScienceEducationVectorStore(config)
        self.prompt_builder = ScienceEducationPromptBuilder()
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
        base = base_query or project_name or project_features or "科教活动"
        seeds = [
            base,
            "科技馆 科普活动 案例",
            "博物馆 教育空间 设计规范",
            "科学实验室 通风要求",
            "研学流线 组织",
            f"{project_features} 科教活动" if project_features else "科教活动 特色",
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
