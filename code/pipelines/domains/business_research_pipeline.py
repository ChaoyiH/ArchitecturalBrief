"""Pipeline for generating business & research area briefs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import BusinessResearchConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import BusinessResearchDataExtractor
from core.generation_integration import GenerationIntegrationModule
from prompts import (
    BUSINESS_RESEARCH_SYSTEM,
    BUSINESS_RESEARCH_JSON_SCHEMA,
    BUSINESS_RESEARCH_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class BusinessResearchVectorStore:
    """Vector store for business & research knowledge."""

    def __init__(self, config: BusinessResearchConfig):
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
            logger.info("已加载业务科研区索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("业务科研区文档为空，无法构建索引")
        logger.info("正在构建业务科研区索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("业务科研区索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("业务科研区索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class BusinessResearchPromptBuilder:
    SYSTEM_PROMPT = BUSINESS_RESEARCH_SYSTEM

    JSON_SCHEMA = (
        "{\n"
        "  \"collection_management\": {\n"
        "    \"process_flow\": \"描述库前区的工作流线组织建议（如：洁污分区）。\",\n"
        "    \"key_rooms\": [\"列出必备房间，如：卸货平台、缓冲间、鉴选室、摄影室等\"],\n"
        "    \"security_level\": \"关于该区域安防等级和门禁控制的建议。\"\n"
        "  },\n"
        "  \"research_conservation\": {\n"
        "    \"labs_requirements\": \"各类修复室/实验室的物理环境要求（如：北向采光、独立排风系统、地面承重等）。\",\n"
        "    \"equipment_space\": \"对大型仪器设备空间的预留建议。\"\n"
        "  },\n"
        "  \"admin_office\": {\n"
        "    \"zoning\": \"行政区与业务区的关系建议（如：集中布置或分散布置）。\",\n"
        "    \"layout_style\": \"办公空间形式建议（如：大空间与独立办公室结合）。\"\n"
        "  },\n"
        "  \"circulation_strategy\": \"关于内部流线（员工/藏品）与外部流线（观众）彻底分离的策略描述。\"\n"
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
        user_prompt = BUSINESS_RESEARCH_USER_TEMPLATE.format(
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
            formatted.append(
                f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}"
            )
        return "\n".join(formatted)


class BusinessResearchGenerator:
    """Facade for business & research task generation."""

    def __init__(self, config: BusinessResearchConfig):
        self.config = config
        self.extractor = BusinessResearchDataExtractor(config)
        self.vector_store = BusinessResearchVectorStore(config)
        self.prompt_builder = BusinessResearchPromptBuilder()
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
        base = base_query or project_name or project_features or "业务科研区"
        seeds = [
            base,
            "博物馆 库前区 流程",
            "文物修复室 设计要求",
            "博物馆 办公区 流线",
            "藏品 摄影室 采光",
            f"{project_features} 行政办公" if project_features else "行政办公 布局",
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
