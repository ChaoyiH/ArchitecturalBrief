"""Pipeline for generating exhibition space design requirements."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import ExhibitionConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import ExhibitionDataExtractor
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class ExhibitionVectorStore:
    """Vector index dedicated to exhibition space knowledge."""

    def __init__(self, config: ExhibitionConfig):
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
            logger.info("已加载展览空间索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("展览空间文档为空，无法构建索引")
        logger.info("正在构建设计展览空间索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("展览空间索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("展览空间索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class ExhibitionPromptBuilder:
    SYSTEM_PROMPT = (
        "你是一位精通博物馆与科技馆设计的资深建筑师和展陈策划专家。"
        "你需要根据项目背景，结合国家规范（硬指标）和优秀案例（软策略），"
        "输出一份专业、可落地的《陈列展览区空间设计任务书》。"
        "你的输出必须包含具体的空间参数、流线策略和功能分区建议，严禁泛泛而谈。"
    )

    JSON_SCHEMA = (
        "{\n"
        "  \"spatial_parameters\": {\n"
        "    \"ceiling_height\": \"建议主要展厅净高范围（如：首层X米，标准层Y米），并引用规范或案例依据。\",\n"
        "    \"column_grid\": \"建议柱网尺寸（如：Xm*Ym），以适应大型展项布置。\",\n"
        "    \"floor_load\": \"建议楼面荷载值（kN/m2），特别是针对重型展品区。\"\n"
        "  },\n"
        "  \"layout_strategy\": {\n"
        "    \"organization_type\": \"推荐的空间组合形式（如：大厅式、串联式、放射式），并说明理由。\",\n"
        "    \"circulation_flow\": \"观众参观流线建议（如：单向强制流线、自由选择流线），以及如何处理人流高峰。\"\n"
        "  },\n"
        "  \"functional_zoning\": [\n"
        "    {\n"
        "      \"zone_name\": \"推荐展区1名称（如：儿童科技乐园）\",\n"
        "      \"floor_suggestion\": \"建议楼层（如：首层）\",\n"
        "      \"area_concept\": \"该展区的设计概念和空间特征描述。\",\n"
        "      \"reference\": \"参考了哪个案例的设置。\"\n"
        "    },\n"
        "    {\n"
        "      \"zone_name\": \"推荐展区2名称\",\n"
        "      \"floor_suggestion\": \"...\",\n"
        "      \"area_concept\": \"...\",\n"
        "      \"reference\": \"...\"\n"
        "    },\n"
        "    {\n"
        "      \"zone_name\": \"推荐展区3名称\",\n"
        "      \"floor_suggestion\": \"...\",\n"
        "      \"area_concept\": \"...\",\n"
        "      \"reference\": \"...\"\n"
        "    }\n"
        "  ],\n"
        "  \"environment_requirements\": \"关于光环境（自然光/人工光控制）和声学环境的具体要求。\"\n"
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
            "# 知识库检索结果 (Retrieved Context)",
            "以下是从规范标准、设计资料和类似案例中检索到的相关信息：",
            "---",
            context,
            "---",
            "",
            "# 生成任务",
            (
                "请为该项目编写“陈列展览区空间设计要求”。请综合考虑作为"
                f"“{project_features}”的特殊性（例如科技馆对层高和荷载要求通常高于一般博物馆）。"
            ),
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


class ExhibitionGenerator:
    """High-level facade for exhibition space requirement generation."""

    def __init__(self, config: ExhibitionConfig):
        self.config = config
        self.extractor = ExhibitionDataExtractor(config)
        self.vector_store = ExhibitionVectorStore(config)
        self.prompt_builder = ExhibitionPromptBuilder()
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
        core = base_query or project_name or project_features or "展览空间"
        terms = [
            core,
            f"{project_name} 展厅净高 规范" if project_name else "展厅净高 规范",
            f"{project_name} 展览流线 案例" if project_name else "展览流线 案例",
            f"{project_name} 常设展览 内容" if project_name else "常设展览 内容",
            f"{project_features} 展厅荷载" if project_features else "展厅荷载",
            f"{project_features} 陈列柱网" if project_features else "陈列柱网",
        ]
        seen = set()
        expanded = []
        for term in terms:
            if term and term not in seen:
                seen.add(term)
                expanded.append(term)
        return expanded

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
                key = (
                    doc.metadata.get("chunk_id")
                    or doc.metadata.get("source")
                    or doc.page_content[:50]
                )
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

