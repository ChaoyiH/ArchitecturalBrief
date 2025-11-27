"""Module that assembles module outputs into a final design brief."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from config import DEFAULT_CONFIG
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from core.generation_integration import GenerationIntegrationModule

logger = logging.getLogger(__name__)


class BriefAssemblyPipeline:
    """LLM-driven assembler that turns module JSON into a markdown design brief."""

    SYSTEM_PROMPT = (
        "你是一位拥有20年经验的建筑策划总师。"
        "你的任务是依据各个专业顾问提供的详细数据（JSON格式），撰写一份结构严谨、逻辑清晰、文风专业的《建筑设计任务书》。"
        "你需要将零散的数据点串联成通顺的段落，并使用 Markdown 格式进行排版。"
    )

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 6000,
    ) -> None:
        cfg = DEFAULT_CONFIG
        self.provider = provider or cfg.llm_provider
        self.model_name = model_name or cfg.llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm_module: Optional[GenerationIntegrationModule] = None

    def _ensure_llm(self) -> None:
        if self._llm_module is None:
            self._llm_module = GenerationIntegrationModule(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    @staticmethod
    def _format_section(value: Any) -> str:
        if value is None:
            return "(暂无数据)"
        if isinstance(value, str):
            return value.strip() or "(暂无数据)"
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            return str(value)

    def _build_user_prompt(
        self,
        project_name: str,
        project_features: str,
        sections: Dict[str, Any],
    ) -> str:
        prompt = [
            "# 项目背景",
            f"项目名称: {project_name}",
            f"项目特征: {project_features}",
            "",
            "# 各专业顾问输入数据",
            "以下是各分项策划的详细要求：",
            "",
            "## 1. 设计理念与愿景",
            self._format_section(sections.get("concept")),
            "",
            "## 2. 核心空间与中庭",
            self._format_section(sections.get("central_hub")),
            "",
            "## 3. 展览空间体系",
            self._format_section(sections.get("exhibition")),
            "",
            "## 4. 特效影院配置",
            self._format_section(sections.get("special_theater")),
            "",
            "## 5. 科教与研学",
            self._format_section(sections.get("science_education")),
            "",
            "## 6. 公共服务与运营",
            self._format_section(sections.get("public_service")),
            "",
            "## 7. 业务科研与后勤",
            self._format_section(sections.get("business_research")),
            "",
            "# 撰写任务",
            f"请汇总上述信息，编写一份完整的《{project_name} 建筑设计任务书》。",
            "",
            "# 格式要求 (Markdown)",
            "1.  **项目概况**: 简述项目背景和核心定位（基于设计理念的 analysis 部分）。",
            "2.  **设计愿景 (Design Vision)**: 整合“设计理念”中的 directions，用富有感染力的语言描述。",
            "3.  **核心空间 (The Hub)**: 描述综合大厅的空间意象和交通组织。",
            "4.  **功能分区详述 (Functional Program)**:",
            "    * **陈列展览区**: 详细列出各展厅名称、面积建议、空间要求（引用 exhibiton 数据）。",
            "    * **影院与表演区**: 描述影院配置和空间工艺（引用 special_theater 数据）。",
            "    * **教育与活动区**: 描述实验室、教室及空间融合策略（引用 science_education 数据）。",
            "    * **公共服务区**: 描述门厅、餐饮、商业及人性化设施（引用 public_service 数据）。",
            "    * **业务与后勤区**: 描述办公、修复、库房流线（引用 business_research 数据）。",
            "5.  **关键技术指标总结**: 将文中提到的净高、跨度、荷载、环境要求汇总成一个表格。",
            "",
            "请保持专业、客观、指导性强的语调。",
        ]
        return "\n".join(prompt)

    def generate_brief(
        self,
        project_name: str,
        project_features: str,
        sections: Dict[str, Any],
    ) -> Dict[str, str]:
        self._ensure_llm()
        user_prompt = self._build_user_prompt(project_name, project_features, sections)
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "{user_prompt}"),
        ])
        chain = chat_prompt | self._llm_module.llm | StrOutputParser()
        response = chain.invoke({"user_prompt": user_prompt})
        return {
            "prompt": {
                "system_prompt": self.SYSTEM_PROMPT,
                "user_prompt": user_prompt,
            },
            "response": response,
        }
