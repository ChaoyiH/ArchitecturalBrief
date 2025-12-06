"""
图构建模块 (Graph Construction)

使用 LangGraph StateGraph 构建任务书生成的异步执行图。

图结构：
    START
      │
      ├──> design ─────────────┐
      ├──> central_hub ────────┤
      ├──> exhibition ─────────┤
      ├──> special_theater ────┼──> assembly ──> END
      ├──> science_education ──┤
    ├──> public_service ─────┤
    └──> operation ──────────┘

设计说明：
- 所有领域模块从 START 并行启动
- 所有领域模块完成后，汇聚到 assembly 节点
- assembly 完成后流向 END

使用 LangGraph 的 fanout/fanin 模式实现并行执行。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from .nodes import (
    NODE_REGISTRY,
    PARALLEL_NODES,
    assembly_node,
    operation_node,
    central_hub_node,
    design_node,
    exhibition_node,
    indicator_node,
    public_service_node,
    science_education_node,
    special_theater_node,
)
from .state import BriefGenerationState

logger = logging.getLogger(__name__)


def build_brief_generation_graph() -> StateGraph:
    """
    构建任务书生成的 StateGraph。

    Returns:
        未编译的 StateGraph 实例（需调用 .compile() 获取可执行对象）
    """
    # 创建图，指定状态类型
    graph = StateGraph(BriefGenerationState)

    # 添加所有领域节点
    graph.add_node("design", design_node)
    graph.add_node("indicators", indicator_node)
    graph.add_node("central_hub", central_hub_node)
    graph.add_node("exhibition", exhibition_node)
    graph.add_node("special_theater", special_theater_node)
    graph.add_node("science_education", science_education_node)
    graph.add_node("public_service", public_service_node)
    graph.add_node("operation", operation_node)

    # 添加组装节点
    graph.add_node("assembly", assembly_node)

    # === 边定义 ===
    # 从 START 并行分发到所有领域节点
    for node_name in PARALLEL_NODES:
        graph.add_edge(START, node_name)

    # 从所有领域节点汇聚到 assembly
    for node_name in PARALLEL_NODES:
        graph.add_edge(node_name, "assembly")

    # assembly 完成后结束
    graph.add_edge("assembly", END)

    logger.debug("任务书生成图构建完成")
    return graph


def compile_graph(checkpointer=None):
    """
    编译图为可执行应用。

    Args:
        checkpointer: 可选的检查点器，用于状态持久化和恢复

    Returns:
        编译后的 LangGraph 应用对象
    """
    graph = build_brief_generation_graph()

    compile_kwargs: Dict[str, Any] = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    app = graph.compile(**compile_kwargs)
    logger.info("LangGraph 应用编译完成")
    return app


# 预编译的默认应用实例（无检查点）
# 使用方式: from pipelines.graph_engine.graph import app; await app.ainvoke(state)
_default_app = None


def get_app():
    """
    获取默认的编译后应用实例（懒加载单例）。

    Returns:
        编译后的 LangGraph 应用
    """
    global _default_app
    if _default_app is None:
        _default_app = compile_graph()
    return _default_app


async def run_graph(initial_state: BriefGenerationState) -> BriefGenerationState:
    """运行图并返回最终状态，并带有一次自动补跑失败模块的机制。

    当前策略：
    1. 先完整执行一次图，得到初始结果；
    2. 检查各模块输出中的 ``error`` 字段；
    3. 如果存在失败模块，则以第一次的结果作为新初始状态，再运行一次完整图，
       作为"补跑"（仅一次）；
    4. 始终返回最后一次执行的状态。

    这样可以在不修改现有节点实现的前提下，对偶发性超时等错误做一次
    自动重试，同时保持架构改动最小。

    Args:
        initial_state: 初始状态字典

    Returns:
        执行完成后的最终状态
    """

    # 在并行任务启动前预加载 embedding 模型
    from .nodes import ensure_embedding_loaded
    ensure_embedding_loaded()

    app = get_app()

    # 第一次执行
    state = await app.ainvoke(initial_state)

    # 检查是否存在需要补跑的模块：
    # 约定模块输出结构为 {"response": ..., "error": ...}
    failed_modules = []
    for key, value in state.items():
        if isinstance(value, dict):
            err = value.get("error")
            if isinstance(err, str) and err.strip():
                failed_modules.append(key)

    if not failed_modules:
        return state

    logger.info("检测到需要补跑的模块: %s", ", ".join(sorted(failed_modules)))

    # 以第一次执行结果作为新初始状态，执行一次补跑
    retry_state = await app.ainvoke(state)
    return retry_state
