"""Pipeline for generating public service area design briefs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import PublicServiceConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import PublicServiceDataExtractor
from core.generation_integration import GenerationIntegrationModule
from prompts import (
    PUBLIC_SERVICE_SYSTEM,
    PUBLIC_SERVICE_JSON_SCHEMA,
    PUBLIC_SERVICE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class PublicServiceVectorStore:
    """Vector store dedicated to public service area knowledge."""

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

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("公共服务区索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class PublicServicePromptBuilder:
    SYSTEM_PROMPT = PUBLIC_SERVICE_SYSTEM

    JSON_SCHEMA = (
        "{\n"
        "  \"entrance_lobby\": {\n"
        "    \"flow_strategy\": \"描述入馆流线的组织策略（如：单向流线、分层检票等）。\",\n"
        "    \"spatial_requirements\": \"门厅/综合大厅的空间尺度建议（面积、净高）及氛围营造。\",\n"
        "    \"key_facilities\": [\"列出必备设施，如：智能储物柜、自动取票机、咨询台\"]\n"
        "  },\n"
        "  \"amenities_standard\": {\n"
        "    \"restroom_config\": \"关于卫生间配置的具体建议（如：依据规范建议男女厕位比例、第三卫生间设置）。\",\n"
        "    \"accessibility\": \"无障碍设计要求（坡道、电梯、盲道等）。\",\n"
        "    \"special_care\": \"母婴室、医务室等关怀设施的要求。\"\n"
        "  },\n"
        "  \"commercial_dining\": [\n"
        "    {\n"
        "      \"type\": \"餐饮/咖啡\",\n"
        "      \"location\": \"建议位置（如：顶层景观区、首层临街等）\",\n"
        "      \"design_note\": \"设计要点（如：独立出入口、排烟要求）。\"\n"
        "    },\n"
        "    {\n"
        "      \"type\": \"文创商店\",\n"
        "      \"location\": \"建议位置（如：出口必经之路）\",\n"
        "      \"design_note\": \"设计要点。\"\n"
        "    }\n"
        "  ],\n"
        "  \"rest_area_concept\": \"关于非经营性公共休息座椅、视听区的布置理念。\"\n"
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
        user_prompt = PUBLIC_SERVICE_USER_TEMPLATE.format(
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
        formatted = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"案例{idx}"
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            snippet = snippet.replace("{", "{{").replace("}", "}}")
            formatted.append(f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}")
        return "\n".join(formatted)


class PublicServiceGenerator:
    """High-level facade for public service area planning."""

    def __init__(self, config: PublicServiceConfig):
        self.config = config
        self.extractor = PublicServiceDataExtractor(config)
        self.vector_store = PublicServiceVectorStore(config)
        self.prompt_builder = PublicServicePromptBuilder()
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
        base = base_query or project_name or project_features or "公共服务区"
        seeds = [
            base,
            f"{project_features} 门厅流线设计" if project_features else "博物馆 门厅流线设计",
            "博物馆 卫生间 规范 数量",
            "博物馆 纪念品商店 位置",
            "无障碍设计规范",
            f"{project_name} 综合大厅 规模" if project_name else "综合大厅 规模",
        ]
        # remove empty & duplicates
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
