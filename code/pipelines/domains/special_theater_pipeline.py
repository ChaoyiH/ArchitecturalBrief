"""Pipeline for generating special effects theater planning briefs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:  # pragma: no cover
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

from config import SpecialTheaterConfig
from utils.data_preparation import SpecialTheaterDataExtractor
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class SpecialTheaterVectorStore:
    """Vector store wrapper for special theater knowledge."""

    def __init__(self, config: SpecialTheaterConfig):
        self.config = config
        self.embedding = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
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
    SYSTEM_PROMPT = (
        "你是一位专业的文化建筑视听顾问和工艺设计师。"
        "你的任务是规划博物馆/科技馆的特效影院系统。"
        "你需要根据项目规模推荐合适的影院组合（如：巨幕+球幕+4D），并给出具体的空间工艺要求（净高、视线设计、声学隔离）。"
    )

    JSON_SCHEMA = (
        "{\n"
        "  \"theater_configuration\": [\n"
        "    {\n"
        "      \"type\": \"推荐影院类型1（如：IMAX球幕影院）\",\n"
        "      \"capacity_suggestion\": \"建议座位数（如：200-250座）\",\n"
        "      \"screen_spec\": \"建议屏幕规格（如：直径23米倾斜式球幕）\",\n"
        "      \"feature_description\": \"该影院的体验特点及科普价值。\"\n"
        "    },\n"
        "    {\n"
        "      \"type\": \"推荐影院类型2（如：4D动感影院）\",\n"
        "      \"capacity_suggestion\": \"...\",\n"
        "      \"screen_spec\": \"...\",\n"
        "      \"feature_description\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"spatial_requirements\": {\n"
        "    \"clear_height\": \"针对所选影院的最大净高需求（如：球幕厅需净高25米以上）。\",\n"
        "    \"structure_span\": \"建议的大跨度结构参数。\",\n"
        "    \"acoustic_isolation\": \"关于影院与其他安静展区之间的隔声/减振策略。\"\n"
        "  },\n"
        "  \"operational_layout\": {\n"
        "    \"access_strategy\": \"如何实现影院的单独对外开放（夜间运营）流线。\",\n"
        "    \"support_rooms\": \"放映机房、排队等候区、3D眼镜分发回收区的布置建议。\"\n"
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
        schema = self.JSON_SCHEMA.replace("{", "{{").replace("}", "}}")
        user_prompt = [
            "# 任务背景",
            f"项目名称: {project_name}",
            f"项目特征: {project_features}",
            "",
            "# 知识库检索结果",
            "以下是关于特效影院（IMAX、球幕、4D等）的配置标准、案例数据和设计规范：",
            "---",
            context,
            "---",
            "",
            "# 生成任务",
            "请为该项目编写《特效影院区空间设计策划书》。",
            "请综合考虑：",
            f"1.  **影院选型**: 根据项目定位（如{project_features}），推荐配置哪些类型的特效影院（如：特大型馆通常配置IMAX球幕）。",
            "2.  **规模建议**: 各个影院的建议座位数和银幕/球幕直径。",
            "3.  **空间工艺**: 对应的建筑层高要求（非常关键，球幕通常需要穿越多层）、结构跨度要求。",
            "4.  **布局策略**: 影院应如何布置以方便独立运营（闭馆后单独开放）并解决隔声问题。",
            "",
            "# 输出要求",
            "请严格按照以下 JSON 格式输出：",
            schema,
        ]
        return {
            "system_prompt": self.SYSTEM_PROMPT,
            "user_prompt": "\n".join(user_prompt),
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
