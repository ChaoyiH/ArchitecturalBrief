"""
节点包装器 (Node Wrappers)

采用适配器模式，将现有的同步 Pipeline 类包装为 LangGraph 兼容的异步节点函数。
每个节点函数：
- 输入：全局状态 (BriefGenerationState)
- 输出：增量字典，仅包含该模块产生的数据

设计原则：
- 不修改原有 Pipeline 内部逻辑
- 使用 asyncio.to_thread 将同步调用转为异步，实现非阻塞并发
- 所有异常被捕获并记录到 errors 字段
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

from config import (
    DEFAULT_BUSINESS_RESEARCH_CONFIG,
    DEFAULT_CENTRAL_HUB_CONFIG,
    DEFAULT_DESIGN_CONCEPT_CONFIG,
    DEFAULT_EXHIBITION_CONFIG,
    DEFAULT_PUBLIC_SERVICE_CONFIG,
    DEFAULT_SCIENCE_EDUCATION_CONFIG,
    DEFAULT_SPECIAL_THEATER_CONFIG,
)
from pipelines.domains.business_research_pipeline import BusinessResearchGenerator
from pipelines.domains.central_hub_pipeline import CentralHubGenerator
from pipelines.domains.design_concept_pipeline import DesignConceptGenerator
from pipelines.domains.exhibition_pipeline import ExhibitionGenerator
from pipelines.domains.public_service_pipeline import PublicServiceGenerator
from pipelines.domains.science_education_pipeline import ScienceEducationGenerator
from pipelines.domains.special_theater_pipeline import SpecialTheaterGenerator
from pipelines.orchestration.brief_assembly_pipeline import BriefAssemblyPipeline
from utils.indicator_analyzer import analyze_indicators

from .state import BriefGenerationState, ModuleOutput, SECTION_KEY_MAP

logger = logging.getLogger(__name__)

# 预加载标记
_embedding_preloaded = False


def ensure_embedding_loaded() -> None:
    """
    确保 embedding 模型已预加载（在并行任务启动前调用）。
    
    这可以避免多个线程同时初始化 HuggingFaceEmbeddings 导致的
    PyTorch meta tensor 错误。
    """
    global _embedding_preloaded
    if _embedding_preloaded:
        return
    
    from core.embedding_manager import preload_embedding
    logger.info("预加载共享 embedding 模型...")
    preload_embedding()
    _embedding_preloaded = True
    logger.info("embedding 模型预加载完成")


def _should_skip_module(state: BriefGenerationState, module_key: str) -> bool:
    """检查指定模块是否已成功完成，可在补跑时跳过。

    判定规则：
    - 若状态中不存在该模块 key，则不能跳过；
    - 若存在 error 且非空，则不能跳过；
    - 若 response 存在且为非空字符串，则认为是一次干净成功，可跳过；
    - 其它情况一律视为需要重新执行。
    """

    output = state.get(module_key)  # type: ignore[assignment]
    if not isinstance(output, dict):
        return False

    error = output.get("error")
    if isinstance(error, str) and error.strip():
        return False

    response = output.get("response")
    if isinstance(response, str) and response.strip():
        return True

    return False


def _parse_json_response(response_text: str | None) -> Any:
    """尝试将模块输出解析为 JSON，失败则返回原始文本。"""
    if not response_text:
        return {}
    text = response_text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.debug("模块输出非 JSON，按原始文本处理")
        return text


# =============================================================================
# 设计理念节点
# =============================================================================


def _run_design_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行设计理念生成。"""
    if _should_skip_module(state, "design"):
        logger.info("🎨 设计理念模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = DesignConceptGenerator(DEFAULT_DESIGN_CONCEPT_CONFIG)
        generator.ensure_index(rebuild=inp.get("rebuild_index", False))
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            filters=inp.get("filters"),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("设计理念模块执行失败")
        return ModuleOutput(error=str(exc))


async def design_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步设计理念节点。"""
    logger.info("🎨 开始执行: 设计理念模块")
    output = await asyncio.to_thread(_run_design_sync, state)
    logger.info("🎨 完成: 设计理念模块")
    return {"design": output}


# =============================================================================
# 综合大厅节点
# =============================================================================


def _run_central_hub_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行综合大厅生成。"""
    if _should_skip_module(state, "central_hub"):
        logger.info("🏛️ 综合大厅模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = CentralHubGenerator(DEFAULT_CENTRAL_HUB_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("综合大厅模块执行失败")
        return ModuleOutput(error=str(exc))


async def central_hub_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步综合大厅节点。"""
    logger.info("🏛️ 开始执行: 综合大厅模块")
    output = await asyncio.to_thread(_run_central_hub_sync, state)
    logger.info("🏛️ 完成: 综合大厅模块")
    return {"central_hub": output}


# =============================================================================
# 展览空间节点
# =============================================================================


def _run_exhibition_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行展览空间生成。"""
    if _should_skip_module(state, "exhibition"):
        logger.info("🖼️ 展览空间模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = ExhibitionGenerator(DEFAULT_EXHIBITION_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("展览空间模块执行失败")
        return ModuleOutput(error=str(exc))


async def exhibition_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步展览空间节点。"""
    logger.info("🖼️ 开始执行: 展览空间模块")
    output = await asyncio.to_thread(_run_exhibition_sync, state)
    logger.info("🖼️ 完成: 展览空间模块")
    return {"exhibition": output}


# =============================================================================
# 特效影院节点
# =============================================================================


def _run_special_theater_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行特效影院生成。"""
    if _should_skip_module(state, "special_theater"):
        logger.info("🎬 特效影院模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = SpecialTheaterGenerator(DEFAULT_SPECIAL_THEATER_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("特效影院模块执行失败")
        return ModuleOutput(error=str(exc))


async def special_theater_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步特效影院节点。"""
    logger.info("🎬 开始执行: 特效影院模块")
    output = await asyncio.to_thread(_run_special_theater_sync, state)
    logger.info("🎬 完成: 特效影院模块")
    return {"special_theater": output}


# =============================================================================
# 科教活动节点
# =============================================================================


def _run_science_education_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行科教活动生成。"""
    if _should_skip_module(state, "science_education"):
        logger.info("🔬 科教活动模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = ScienceEducationGenerator(DEFAULT_SCIENCE_EDUCATION_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("科教活动模块执行失败")
        return ModuleOutput(error=str(exc))


async def science_education_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步科教活动节点。"""
    logger.info("🔬 开始执行: 科教活动模块")
    output = await asyncio.to_thread(_run_science_education_sync, state)
    logger.info("🔬 完成: 科教活动模块")
    return {"science_education": output}


# =============================================================================
# 公共服务节点
# =============================================================================


def _run_public_service_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行公共服务生成。"""
    if _should_skip_module(state, "public_service"):
        logger.info("🚻 公共服务模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = PublicServiceGenerator(DEFAULT_PUBLIC_SERVICE_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("公共服务模块执行失败")
        return ModuleOutput(error=str(exc))


async def public_service_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步公共服务节点。"""
    logger.info("🚻 开始执行: 公共服务模块")
    output = await asyncio.to_thread(_run_public_service_sync, state)
    logger.info("🚻 完成: 公共服务模块")
    return {"public_service": output}


# =============================================================================
# 业务科研节点
# =============================================================================


def _run_business_research_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行业务科研生成。"""
    if _should_skip_module(state, "business_research"):
        logger.info("📊 业务科研模块已成功完成，本次补跑将跳过执行")
        return ModuleOutput()

    inp = state.get("input", {})
    try:
        generator = BusinessResearchGenerator(DEFAULT_BUSINESS_RESEARCH_CONFIG)
        result = generator.generate(
            project_name=inp.get("project_name", ""),
            project_features=inp.get("project_features", ""),
            query=inp.get("query"),
            top_k=inp.get("top_k"),
            rebuild_index=inp.get("rebuild_index", False),
            dry_run=inp.get("dry_run", False),
        )
        return ModuleOutput(
            prompt=result.get("prompt"),
            contexts=result.get("contexts"),
            response=result.get("response"),
        )
    except Exception as exc:
        logger.exception("业务科研模块执行失败")
        return ModuleOutput(error=str(exc))


async def business_research_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步业务科研节点。"""
    logger.info("📊 开始执行: 业务科研模块")
    output = await asyncio.to_thread(_run_business_research_sync, state)
    logger.info("📊 完成: 业务科研模块")
    return {"business_research": output}


# =============================================================================
# 经济技术指标节点
# =============================================================================


def _extract_target_area(input_data: Dict[str, Any]) -> float:
    """从输入中提取建筑面积，支持多种字段名与类型。"""

    candidates = [
        "target_area",
        "building_area",
        "gross_floor_area",
    ]

    value = None
    for key in candidates:
        if key in input_data and input_data[key] is not None:
            value = input_data[key]
            break

    if value is None:
        return 0.0

    # 兼容字符串/数字
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            return float(text)
        except ValueError:
            logger.warning("无法解析输入面积字段为数值: %r", value)
            return 0.0

    logger.warning("未知的面积字段类型: %r", type(value))
    return 0.0


def _run_indicators_sync(state: BriefGenerationState) -> ModuleOutput:
    """同步执行经济技术指标分析。"""

    inp = state.get("input", {})
    try:
        target_area = _extract_target_area(inp)
        if target_area <= 0:
            raise ValueError("未提供有效的建筑面积参数（target_area/building_area/gross_floor_area）。")

        result = analyze_indicators(target_area)
        # 将字典结果序列化为 JSON 字符串，方便下游统一处理
        response_text = json.dumps(result, ensure_ascii=False, indent=2)
        return ModuleOutput(response=response_text)
    except Exception as exc:
        logger.exception("经济技术指标模块执行失败")
        return ModuleOutput(error=str(exc))


async def indicator_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步经济技术指标节点。"""

    logger.info("📐 开始执行: 经济技术指标模块")
    output = await asyncio.to_thread(_run_indicators_sync, state)
    logger.info("📐 完成: 经济技术指标模块")
    return {"indicators": output}


# =============================================================================
# 任务书组装节点
# =============================================================================


def _run_assembly_sync(state: BriefGenerationState) -> str:
    """同步执行任务书组装。"""
    inp = state.get("input", {})

    # 收集各模块输出
    sections: Dict[str, Any] = {}
    for module_key in [
        "design",
        "indicators",
        "central_hub",
        "exhibition",
        "special_theater",
        "science_education",
        "public_service",
        "business_research",
    ]:
        output: ModuleOutput = state.get(module_key, {})
        section_key = SECTION_KEY_MAP.get(module_key, module_key)

        if output.get("error"):
            sections[section_key] = f"生成失败: {output['error']}"
        elif output.get("response"):
            parsed = _parse_json_response(output["response"])
            sections[section_key] = parsed or "(暂无内容)"
        else:
            sections[section_key] = "(暂无内容)"

    assembler = BriefAssemblyPipeline()
    result = assembler.generate_brief(
        project_name=inp.get("project_name", ""),
        project_features=inp.get("project_features", ""),
        sections=sections,
    )
    return result.get("response", "")


async def assembly_node(state: BriefGenerationState) -> Dict[str, Any]:
    """异步任务书组装节点。"""
    logger.info("📝 开始执行: 任务书组装")
    inp = state.get("input", {})

    # 检查是否为 dry_run 模式
    if inp.get("dry_run", False):
        logger.info("📝 Dry-run 模式，跳过组装")
        return {"assembled_brief": "(dry-run 模式，跳过组装)"}

    markdown = await asyncio.to_thread(_run_assembly_sync, state)
    logger.info("📝 完成: 任务书组装")
    return {"assembled_brief": markdown}


# =============================================================================
# 节点注册表（便于图构建时动态引用）
# =============================================================================

NODE_REGISTRY = {
    "design": design_node,
    "indicators": indicator_node,
    "central_hub": central_hub_node,
    "exhibition": exhibition_node,
    "special_theater": special_theater_node,
    "science_education": science_education_node,
    "public_service": public_service_node,
    "business_research": business_research_node,
    "assembly": assembly_node,
}

# 第一阶段并行节点（不依赖其他模块输出）
PARALLEL_NODES = [
    "design",
    "indicators",
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
]
