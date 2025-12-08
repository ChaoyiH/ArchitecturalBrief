"""Intent router for QA pipeline."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DEFAULT_CONFIG
from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class QAIntent(str, Enum):
    INTENT_CASE = "case"
    INTENT_NORM = "norm"
    INTENT_HYBRID = "hybrid"


class Router:
    """LLM-based router to decide query intent."""

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        fast_llm_provider: str = "google",
        fast_llm_model: str = "gemini-flash-latest",
    ) -> None:
        # 使用快速模型作为路由器默认
        self.llm_provider = fast_llm_provider or llm_provider or DEFAULT_CONFIG.llm_provider
        self.llm_model = fast_llm_model or llm_model or DEFAULT_CONFIG.llm_model
        self.llm_module = GenerationIntegrationModule(
            provider=self.llm_provider,
            model_name=self.llm_model,
            temperature=0.0,
            max_tokens=512,
        )

    def classify(self, question: str) -> QAIntent:
        template = """
判断用户问题类型，输出以下之一：case / norm / hybrid。
- case: 具体建筑案例、面积、地点、年份、项目名称、统计类排名。
- norm: 建筑规范、标准、红线、强条、疏散、防火、指引。
- hybrid: 同时涉及案例和规范或需要对比两者。
用户问题: {question}
仅输出类别单词。
        """

        try:
            if self.llm_provider == "google":
                prompt_text = template.format(question=question).strip()
                result = self.llm_module.llm.invoke(prompt_text).strip().lower()
            else:
                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | self.llm_module.llm | StrOutputParser()
                result = chain.invoke({"question": question}).strip().lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning("路由 LLM 调用失败，使用启发式: %s", exc)
            return self._heuristic_route(question)

        if "norm" in result:
            return QAIntent.INTENT_NORM
        if "hybrid" in result:
            return QAIntent.INTENT_HYBRID
        if "case" in result:
            return QAIntent.INTENT_CASE

        return self._heuristic_route(question)

    @staticmethod
    def _heuristic_route(question: str) -> QAIntent:
        lowered = question.lower()
        norm_keys = ["规范", "标准", "红线", "条文", "fire", "疏散", "防火", "指标"]
        case_keys = ["案例", "项目", "面积", "最大", "最小", "排名", "位置", "哪", "多少"]

        if any(k.lower() in lowered for k in norm_keys):
            return QAIntent.INTENT_NORM
        if any(k.lower() in lowered for k in case_keys):
            return QAIntent.INTENT_CASE
        return QAIntent.INTENT_HYBRID


__all__ = ["Router", "QAIntent"]
