"""Operation (business & operations) value-loop planner with evidence-first enforcement."""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from config import BusinessResearchConfig
from core.embedding_manager import get_embedding
from core.generation_integration import GenerationIntegrationModule
from utils.data_preparation import BusinessResearchDataExtractor

logger = logging.getLogger(__name__)


class OperationVectorStore:
    """Vector store for operation (commercial & edu) evidence."""

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
            logger.info("已加载商业/运营索引: %s", self.config.index_save_path)
            return True
        except Exception:
            return False

    def build(self, documents: Sequence[Document]) -> None:
        if not documents:
            raise ValueError("商业/运营文档为空，无法构建索引")
        logger.info("正在构建商业/运营索引 (文档=%d)...", len(documents))
        self.vectorstore = FAISS.from_documents(list(documents), self.embedding)
        self.vectorstore.save_local(self.config.index_save_path)
        logger.info("商业/运营索引保存至: %s", self.config.index_save_path)

    def ensure_ready(self, loader, rebuild: bool = False) -> None:
        if not rebuild and self.load():
            return
        docs = loader()
        self.build(docs)

    def search(self, query: str, top_k: int) -> List[Document]:
        if self.vectorstore is None:
            raise RuntimeError("商业/运营索引尚未构建")
        return self.vectorstore.similarity_search(query, k=top_k)


class OperationPromptBuilder:
    """Prompt enforcing evidence/opinion separation with structured JSON schema."""

    @staticmethod
    def _escape_braces(text: str) -> str:
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _format_context(docs: Sequence[Document]) -> str:
        if not docs:
            return "(未检索到参考内容)"
        formatted: List[str] = []
        for idx, doc in enumerate(docs, 1):
            meta = doc.metadata or {}
            src = meta.get("source_type", "unknown")
            name = meta.get("project_name") or meta.get("doc_name") or f"案例{idx}"
            snippet = doc.page_content.strip()
            snippet = snippet[:800] + "..." if len(snippet) > 800 else snippet
            snippet = snippet.replace("{", "{{").replace("}", "}}")
            formatted.append(f"【片段{idx} | 来源:{src} | 名称:{name}】\n{snippet}")
        return "\n".join(formatted)

    def build_prompt(self, project_name: str, project_features: str, docs: Sequence[Document]) -> Dict[str, str]:
        ctx = self._format_context(docs)
        system = (
            "You are a Museum Cultural Economy Consultant. Your goal is to create value loops, not merely split floor areas."
            " Enforce evidence-first: every recommendation must point to retrieved evidence."
            " Separate facts (case evidence) from opinions (project-specific strategy)."
            " Uphold public-benefit framing: commercial services must serve science education, not pure profit."
        )

        schema = (
            "{\n"
            "  \"global_evidence\": [\n"
            "    {\"category\": \"文创与零售 / Education / etc.\", \"description\": \"Summary...\", \"cases\": [\"Case A\", \"Case B\"]}\n"
            "  ],\n"
            "  \"spatial_strategy\": [\n"
            "    {\"type\": \"主题餐饮 / 研学教室 / etc.\", \"proposal\": \"Suggestion...\", \"reference\": \"Evidence...\"}\n"
            "  ],\n"
            "  \"tailored_recommendations\": [\n"
            "    {\"topic\": \"IP开发 / 夜间经济 / etc.\", \"strategy\": \"Core strategy...\", \"rationale\": \"Why...\"}\n"
            "  ],\n"
            "  \"technical_requirements\": [\"Requirement 1\", \"Requirement 2\"]\n"
            "}"
        )

        user_raw = (
            f"项目：{project_name or '未命名'} | 特征：{project_features or '未提供'}\n"
            "请仅用中文输出，严格返回 JSON（不要添加 Markdown、标题、列表符号、代码块或 ```json 包裹）。\n"
            "值中禁止使用 Markdown 粗体/标题（如 **、##），除非必要的规范编号。\n"
            "严禁幻觉：仅依据 Context，缺证据则填写空字符串/空数组并可注明“未检索到相关证据”。\n"
            "如 Context 含有《建筑设计防火规范》或类似条文，须在 technical_requirements 中体现疏散/人员密度要求。\n"
            "若未检索到“元宇宙商业”等概念，不得编造。\n"
            "Context:\n" + ctx + "\n\n"
            "输出必须完全符合以下 JSON 结构，键名不可更改，可为空：\n"
            f"{schema}\n"
        )

        user = self._escape_braces(user_raw)
        return {"system_prompt": system, "user_prompt": user}


class OperationGenerator:
    """Operation generator with parallel query expansion."""

    def __init__(self, config: BusinessResearchConfig):
        self.config = config
        self.extractor = BusinessResearchDataExtractor(config)
        self.vector_store = OperationVectorStore(config)
        self.prompt_builder = OperationPromptBuilder()
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
        base = base_query or project_name or project_features or "科技馆 商业运营"
        physical = [
            "museum cafe shop layout",
            "科技馆 餐厅 商店 布局",
            "museum bookstore giftshop 独立动线",
            "museum mixed-use lobby retail"
        ]
        revenue = [
            "museum revenue models",
            "科技馆 研学 收入 文创",
            "science museum membership program",
            "museum event rental auditorium"
        ]
        trends = [
            "museum night tour immersive dining",
            "科技馆 夜游 沉浸式",
            "science show cafe activation",
            "rooftop bar museum public"
        ]
        seeds = [base] + physical + revenue + trends
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
        response_text = response.strip() if isinstance(response, str) else str(response)
        parsed = self._parse_json_response(response_text)

        return {
            "prompt": prompt,
            "contexts": contexts,
            "response": response_text,
            "json_result": parsed,
        }

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, object]:
        if not text:
            return {}
        cleaned = text.strip()
        cleaned = re.sub(r"^```json|```$", "", cleaned, flags=re.IGNORECASE).strip()
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {"data": parsed}