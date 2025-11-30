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
      └──> business_research ──┘

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
    business_research_node,
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
    graph.add_node("business_research", business_research_node)

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
    """
    便捷函数：运行图并返回最终状态。

    Args:
        initial_state: 初始状态字典

    Returns:
        执行完成后的最终状态
    """
    # 在并行任务启动前预加载 embedding 模型
    from .nodes import ensure_embedding_loaded
    ensure_embedding_loaded()
    
    app = get_app()
    result = await app.ainvoke(initial_state)
    return result
