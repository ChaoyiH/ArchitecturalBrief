"""
全局状态定义 (State Definition)

此模块定义了 LangGraph 图中流转的全局状态对象。
所有节点都从此状态读取输入、写入输出。

关于并行写入：
- LangGraph 默认对字典进行浅合并 (shallow merge)
- 每个节点只返回增量字段（如 {"exhibition": result}），框架自动合并
- 不同节点写入不同字段，不存在覆盖冲突
- 若多个节点写入同一字段，需自定义 Reducer（本架构已避免此情况）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class BriefGenerationInput(TypedDict, total=False):
    """用户输入的初始参数，用于启动图执行。"""

    project_name: str
    project_features: str
    query: Optional[str]
    top_k: Optional[int]
    rebuild_index: bool
    dry_run: bool
    filters: Optional[Dict[str, Any]]
    llm_provider: Optional[str]
    llm_model: Optional[str]
    target_area: Optional[float]
    total_area: Optional[float]
    project_location: Optional[str]
    user_project_info: Optional[str]


class ModuleOutput(TypedDict, total=False):
    """单个模块的输出结构。"""

    prompt: Optional[Dict[str, str]]
    contexts: Optional[List[Any]]
    response: Optional[str]
    # 可选的结构化 JSON 输出（例如设计理念模块的最终结果）
    json_result: Optional[Dict[str, Any]]
    error: Optional[str]


class BriefGenerationState(TypedDict, total=False):
    """
    全局状态对象 - 贯穿整个图执行的数据总线。

    Attributes:
        input: 用户输入参数（不可变）
        design: 设计理念模块输出
        indicators: 经济技术指标分析模块输出
        central_hub: 综合大厅模块输出
        exhibition: 展览空间模块输出
        special_theater: 特效影院模块输出
        science_education: 科教活动模块输出
        public_service: 公共服务模块输出
        business_research: 业务科研模块输出（后勤/BOH）
        operation: 商业运营模块输出（前场/FOH）
        assembled_brief: 最终组装的任务书 Markdown
        errors: 各模块运行错误记录
        execution_log: 执行日志（可选，用于调试）
    """

    # ========== 输入层 ==========
    input: BriefGenerationInput

    # ========== 模块输出层（第一阶段并行） ==========
    design: ModuleOutput
    indicators: ModuleOutput
    central_hub: ModuleOutput
    exhibition: ModuleOutput
    special_theater: ModuleOutput
    science_education: ModuleOutput
    public_service: ModuleOutput
    business_research: ModuleOutput
    operation: ModuleOutput

    # ========== 组装输出层（第二阶段） ==========
    assembled_brief: Optional[str]
    assembled_json: Optional[Dict[str, Any]]

    # ========== 元信息 ==========
    errors: Dict[str, str]
    execution_log: List[str]


def create_initial_state(
    project_name: str,
    project_features: str,
    query: Optional[str] = None,
    top_k: Optional[int] = None,
    rebuild_index: bool = False,
    dry_run: bool = False,
    filters: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    target_area: Optional[float] = None,
    total_area: Optional[float] = None,
    project_location: Optional[str] = None,
    user_project_info: Optional[str] = None,
) -> BriefGenerationState:
    """
    工厂函数：创建初始状态对象。

    Args:
        project_name: 项目名称
        project_features: 项目特征描述
        query: 自定义检索查询
        top_k: 检索数量
        rebuild_index: 是否重建索引
        dry_run: 是否仅预览 Prompt
        filters: 检索过滤条件

    Returns:
        初始化好的 BriefGenerationState
    """
    return BriefGenerationState(
        input=BriefGenerationInput(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            rebuild_index=rebuild_index,
            dry_run=dry_run,
            filters=filters,
            llm_provider=llm_provider,
            llm_model=llm_model,
            target_area=target_area,
            total_area=total_area,
            project_location=project_location,
            user_project_info=user_project_info,
        ),
        design={},
        indicators={},
        central_hub={},
        exhibition={},
        special_theater={},
        science_education={},
        public_service={},
        business_research={},
        operation={},
        assembled_brief=None,
        assembled_json=None,
        errors={},
        execution_log=[],
    )


# 模块名到状态字段的映射（便于动态访问）
MODULE_STATE_KEYS = [
    "design",
    "indicators",
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
    "operation",
]

# 模块名到任务书章节键的映射
SECTION_KEY_MAP = {
    "design": "concept",
    "indicators": "indicators",
    "central_hub": "central_hub",
    "exhibition": "exhibition",
    "special_theater": "special_theater",
    "science_education": "science_education",
    "public_service": "public_service",
    "business_research": "business_research",
    "operation": "operation",
}
