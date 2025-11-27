"""
Graph Engine 包

基于 LangGraph 的异步图执行引擎，用于并行生成任务书各模块内容。

主要导出：
- BriefGenerationState: 全局状态类型定义
- create_initial_state: 创建初始状态的工厂函数
- run_graph: 便捷的图执行函数
- compile_graph: 编译图（支持自定义 checkpointer）
- get_app: 获取默认编译后的应用实例

使用示例:
    from pipelines.graph_engine import create_initial_state, run_graph

    state = create_initial_state(
        project_name="某科技馆",
        project_features="占地面积 15000 平方米...",
        rebuild_index=False,
        dry_run=False,
    )

    result = await run_graph(state)
    print(result["assembled_brief"])
"""

from .state import (
    BriefGenerationState,
    BriefGenerationInput,
    ModuleOutput,
    create_initial_state,
    MODULE_STATE_KEYS,
    SECTION_KEY_MAP,
)

from .graph import (
    build_brief_generation_graph,
    compile_graph,
    get_app,
    run_graph,
)

from .nodes import (
    NODE_REGISTRY,
    PARALLEL_NODES,
)

__all__ = [
    # 状态相关
    "BriefGenerationState",
    "BriefGenerationInput",
    "ModuleOutput",
    "create_initial_state",
    "MODULE_STATE_KEYS",
    "SECTION_KEY_MAP",
    # 图相关
    "build_brief_generation_graph",
    "compile_graph",
    "get_app",
    "run_graph",
    # 节点相关
    "NODE_REGISTRY",
    "PARALLEL_NODES",
]
