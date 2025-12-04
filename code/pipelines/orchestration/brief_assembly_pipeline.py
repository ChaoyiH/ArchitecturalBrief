"""
确定性任务书组装模块

将各模块的 JSON/字典输出按固定模板拼接为完整的 Markdown 设计任务书。
不使用 LLM，执行速度为毫秒级，完整保留上游模块产生的所有数据细节。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BriefAssemblyPipeline:
    """确定性模板拼接器，将模块输出组装为 Markdown 设计任务书。"""

    # 章节顺序与 key 映射
    # 注意：这里仅控制 Markdown 中的章节顺序与标题文案
    SECTION_ORDER = [
        ("indicators", "经济技术指标 (Technical Indicators)"),
        ("concept", "设计理念 (Design Concept)"),
        # 主要功能设计要求
        ("exhibition", "主要功能设计要求 / 展陈体系 (Exhibition)"),
        ("science_education", "主要功能设计要求 / 科研教学 (Science Education)"),
        ("public_service", "主要功能设计要求 / 公共服务 (Public Service)"),
        ("business_research", "主要功能设计要求 / 业务科研与后勤 (Business & Research)"),
        # 重点空间
        ("special_theater", "重点空间 / 特效影院 (Special Theater)"),
        ("central_hub", "重点空间 / 综合大厅与中庭 (Central Hub & Atrium)"),
    ]

    def __init__(self) -> None:
        """初始化（无需 LLM 配置）。"""
        pass

    @staticmethod
    def _dict_to_markdown(data: Any, level: int = 3) -> str:
        """
        递归将字典/列表/字符串转换为 Markdown 格式文本。

        Args:
            data: 待转换的数据（dict, list, str, 或其他）
            level: 当前标题层级（默认 3，即 ###）

        Returns:
            格式化后的 Markdown 字符串
        """
        if data is None:
            return "_（暂无数据）_\n"

        if isinstance(data, str):
            text = data.strip()
            return f"{text}\n" if text else "_（暂无数据）_\n"

        if isinstance(data, bool):
            return f"{'是' if data else '否'}\n"

        if isinstance(data, (int, float)):
            return f"{data}\n"

        if isinstance(data, list):
            if not data:
                return "_（暂无数据）_\n"
            lines = []
            for item in data:
                if isinstance(item, dict):
                    # 列表中的字典，展开为子项
                    item_md = BriefAssemblyPipeline._dict_to_markdown(item, level + 1)
                    lines.append(item_md)
                elif isinstance(item, str):
                    lines.append(f"- {item.strip()}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines) + "\n"

        if isinstance(data, dict):
            if not data:
                return "_（暂无数据）_\n"

            lines = []
            header_prefix = "#" * min(level, 6)  # 最多 6 级标题

            for key, value in data.items():
                # 将 key 格式化为更友好的标题
                title = BriefAssemblyPipeline._format_key_as_title(key)

                if isinstance(value, dict):
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(BriefAssemblyPipeline._dict_to_markdown(value, level + 1))
                elif isinstance(value, list):
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(BriefAssemblyPipeline._dict_to_markdown(value, level + 1))
                elif isinstance(value, str) and len(value) > 100:
                    # 长文本单独作为段落
                    lines.append(f"{header_prefix} {title}\n")
                    lines.append(f"{value.strip()}\n")
                else:
                    # 短文本/数字/布尔值作为行内描述
                    formatted_value = BriefAssemblyPipeline._dict_to_markdown(value, level + 1).strip()
                    lines.append(f"**{title}**: {formatted_value}\n")

            return "\n".join(lines) + "\n"

        # 其他类型直接转字符串
        return f"{data}\n"

    @staticmethod
    def _format_key_as_title(key: str) -> str:
        """
        将字典的 key 格式化为更友好的标题。

        Args:
            key: 原始 key（如 snake_case 或 camelCase）

        Returns:
            格式化后的标题
        """
        title = key.replace("_", " ")
        title = title.title()
        return title

    def assemble_generated_content(
        self,
        project_name: str,
        project_features: str,
        context: Dict[str, Any],
    ) -> str:
        """
        将各模块输出按模板拼接为完整的 Markdown 任务书。

        Args:
            project_name: 项目名称
            project_features: 项目特征描述
            context: 各模块输出字典，key 为模块名，value 为该模块生成的内容

        Returns:
            完整的 Markdown 文档字符串
        """
        lines: List[str] = []

        # 标题
        lines.append(f"# {project_name} 建筑设计任务书\n")
        lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append("---\n")

        # 1. 项目概况
        lines.append("## 1. 项目概况\n")
        lines.append(f"{project_features}\n")
        lines.append("")

        # 2-8. 各专业章节
        section_num = 2
        for key, title in self.SECTION_ORDER:
            lines.append(f"## {section_num}. {title}\n")

            section_data = context.get(key)
            if section_data is None:
                lines.append("_（该模块暂无输出）_\n")
            elif isinstance(section_data, str):
                if section_data.startswith("生成失败"):
                    lines.append(f"> ⚠️ {section_data}\n")
                else:
                    lines.append(f"{section_data.strip()}\n")
            else:
                lines.append(self._dict_to_markdown(section_data, level=3))

            lines.append("")
            section_num += 1

        # 尾部
        lines.append("---\n")
        lines.append("*本任务书由 RAG 系统自动生成，仅供参考。*\n")

        return "\n".join(lines)

    def generate_brief(
        self,
        project_name: str,
        project_features: str,
        sections: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        生成设计任务书（接口兼容）。

        Args:
            project_name: 项目名称
            project_features: 项目特征
            sections: 各模块输出字典

        Returns:
            包含 "response" 键的字典，值为生成的 Markdown 文本
        """
        logger.info("开始组装任务书（确定性模板拼接）...")
        markdown = self.assemble_generated_content(project_name, project_features, sections)
        logger.info("任务书组装完成，长度: %d 字符", len(markdown))

        return {
            "response": markdown,
        }
