"""Pipeline for generating central hall & circulation hub briefs."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import CentralHubConfig
from core.embedding_manager import get_embedding
from utils.data_preparation import CentralHubDataExtractor
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class CentralHubVectorStore:
    """Vector store handler for central hub knowledge."""

    def __init__(self, config: CentralHubConfig):
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
            logger.info("已加载综合大厅索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("综合大厅文档为空，无法构建索引")
        logger.info("正在构建综合大厅索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("综合大厅索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("综合大厅索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class CentralHubPromptBuilder:
    SYSTEM_PROMPT = (
        "你是一位擅长公共空间塑造的建筑设计大师。"
        "你的任务是为博物馆/科技馆策划“综合大厅（中庭）”。这是一个集交通枢纽、仪式感展示和环境调节于一体的核心空间。"
        "你需要平衡空间的震撼力（视觉焦点）与功能性（人流集散）。"
    )

    JSON_SCHEMA = (
        "{\n"
        "  \"spatial_concept\": {\n"
        "    \"theme_name\": \"为大厅起一个主题名（如：时空隧道、生态峡谷）\",\n"
        "    \"form_description\": \"描述大厅的空间形态（如：X层通高的中庭，通过流线型栏板引导视线）。\",\n"
        "    \"atmosphere\": \"描述空间氛围（如：明亮、科技感、甚至带有神圣感）。\"\n"
        "  },\n"
        "  \"scale_reference\": {\n"
        "    \"height_suggestion\": \"建议通高高度（如：24m，贯穿1-4层）。\",\n"
        "    \"area_suggestion\": \"建议核心区面积范围。\",\n"
        "    \"rationale\": \"基于规范或参考案例的理由。\"\n"
        "  },\n"
        "  \"circulation_hub\": {\n"
        "    \"vertical_transport\": \"主要垂直交通工具的选型与布局建议（如：飞天梯、螺旋坡道）。\",\n"
        "    \"visual_connection\": \"如何建立大厅与各层展厅的视线联系。\"\n"
        "  },\n"
        "  \"feature_element\": {\n"
        "    \"type\": \"建议的标志性元素（如：悬挂展品、互动媒体墙、巨型雕塑）。\",\n"
        "    \"description\": \"该元素的具体描述及其承载的文化/科技寓意。\"\n"
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
            "以下是关于综合大厅、中庭和序厅的规范指标、设计资料及优秀案例：",
            "---",
            context,
            "---",
            "",
            "# 生成任务",
            "请为该项目编写《综合大厅与核心空间策划书》。",
            "请重点关注：",
            "1.  **空间形态**: 大厅的形态特征（如：通高空间、穹顶、线性长廊）。",
            "2.  **视觉焦点**: 建议设置何种标志性装置或艺术品（如：上海科技馆的“生命之卵”）。",
            "3.  **垂直交通**: 核心楼梯/扶梯的布置方式，如何引导观众向上层展厅流动。",
            "4.  **物理环境**: 采光（天窗/幕墙）与通风策略。",
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


class CentralHubGenerator:
    """Facade for central hub task generation."""

    def __init__(self, config: CentralHubConfig):
        self.config = config
        self.extractor = CentralHubDataExtractor(config)
        self.vector_store = CentralHubVectorStore(config)
        self.prompt_builder = CentralHubPromptBuilder()
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
        base = base_query or project_name or project_features or "综合大厅"
        seeds = [
            base,
            f"{project_name} 综合大厅 案例" if project_name else "博物馆 综合大厅 案例",
            "博物馆 中庭 设计手法",
            "科技馆 序厅 标志性展项",
            "综合大厅 面积指标",
            f"{project_features} 中庭 枢纽" if project_features else "中庭 交通 枢纽",
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
