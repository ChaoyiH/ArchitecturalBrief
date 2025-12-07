"""Entry point for design concept & exhibition-space generators.

支持两种执行模式：
- sequential (默认): 顺序执行各模块
- parallel: 使用 LangGraph 异步图引擎并行执行
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from config import (
    DEFAULT_BUSINESS_RESEARCH_CONFIG,
    DEFAULT_CENTRAL_HUB_CONFIG,
    DEFAULT_DESIGN_CONCEPT_CONFIG,
    DEFAULT_EXHIBITION_CONFIG,
    DEFAULT_SPECIAL_THEATER_CONFIG,
    DEFAULT_SCIENCE_EDUCATION_CONFIG,
    DEFAULT_PUBLIC_SERVICE_CONFIG,
    BusinessResearchConfig,
    CentralHubConfig,
    DesignConceptConfig,
    ExhibitionConfig,
    SpecialTheaterConfig,
    ScienceEducationConfig,
    PublicServiceConfig,
)
from pipelines.orchestration.brief_assembly_pipeline import BriefAssemblyPipeline
from pipelines.domains.operation_pipeline import OperationGenerator
from pipelines.domains.business_research_pipeline import BusinessResearchGenerator
from pipelines.domains.central_hub_pipeline import CentralHubGenerator
from pipelines.domains.design_concept_pipeline import DesignConceptGenerator
from pipelines.domains.exhibition_pipeline import ExhibitionGenerator
from pipelines.domains.public_service_pipeline import PublicServiceGenerator
from pipelines.domains.special_theater_pipeline import SpecialTheaterGenerator
from pipelines.domains.science_education_pipeline import ScienceEducationGenerator
from utils.indicator_analyzer import analyze_indicators
from langchain_core.documents import Document

# LangGraph 图引擎（延迟导入以保持向后兼容）
_graph_engine = None

logger = logging.getLogger(__name__)


def _get_graph_engine():
    """延迟加载 LangGraph 图引擎模块。"""
    global _graph_engine
    if _graph_engine is None:
        from pipelines.graph_engine import create_initial_state, run_graph
        _graph_engine = {"create_initial_state": create_initial_state, "run_graph": run_graph}
    return _graph_engine

# 尝试启用统一日志/消息捕获
try:  # pragma: no cover
    from utils.log_setup import setup as _log_setup, record_message as _record_msg

    _log_setup()
except Exception:  # noqa: BLE001
    _record_msg = None  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Task brief generator (设计理念 / 展览空间)"
    )
    parser.add_argument("--config", help="从 YAML 文件加载项目与LLM配置，例如 design_config.yaml")
    parser.add_argument("--project-name", help="项目名称或类型")
    parser.add_argument(
        "--project-features",
        help="项目关键特征或背景描述（例如选址、规模、体验重点等）",
    )
    parser.add_argument("--query", help="自定义检索查询语句（默认使用项目特征）")
    parser.add_argument("--top-k", type=int, default=None, help="检索案例数量")
    parser.add_argument("--min-area", type=float, help="按建筑面积下限过滤")
    parser.add_argument("--max-area", type=float, help="按建筑面积上限过滤")
    parser.add_argument("--category", help="按项目类别过滤")
    parser.add_argument(
        "--target-area",
        type=float,
        default=None,
        help="目标建筑面积 (m²)，用于经济技术指标分析",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建所选步骤的向量索引",
    )
    parser.add_argument(
        "--step",
        default=None,  # 改为 None，让 YAML 配置优先
        help=(
            "指定生成阶段，可选 design / central_hub / exhibition / special_theater / science_education / public_service / operation / both / all / full（全案整合），"
            "或以逗号分隔组合。如未指定，默认为 design"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出提示词与上下文，不调用模型",
    )
    parser.add_argument(
        "--show-contexts",
        action="store_true",
        help="打印检索到的上下文摘要",
    )
    parser.add_argument(
        "--mode",
        default="sequential",
        choices=["sequential", "parallel"],
        help="执行模式：sequential (顺序) 或 parallel (并行，使用 LangGraph)",
    )
    parser.add_argument(
        "--llm-provider",
        dest="llm_provider",
        help="覆盖默认的 LLM provider，例如 minimax 或 moonshot",
    )
    parser.add_argument(
        "--llm-model",
        dest="llm_model",
        help="覆盖默认的模型名称，例如 Minimax-M2、kimi-k2 等",
    )
    return parser.parse_args()


def _load_yaml_config(args: argparse.Namespace) -> Dict[str, Any]:
    """从 YAML 配置文件加载入口参数，命令行优先级最高。"""
    base: Dict[str, Any] = {
        "project_name": None,
        "project_features": None,
        "target_area": None,
        "total_area": None,  # 综合大厅模块使用
        "project_location": None,  # 综合大厅模块使用
        "user_project_info": None,
        "query": None,
        "llm_provider": None,
        "llm_model": None,
        "step": None,
        "mode": None,
        "top_k": None,
        "min_area": None,
        "max_area": None,
        "category": None,
        "rebuild_index": False,
        "dry_run": False,
        "show_contexts": False,
    }

    if getattr(args, "config", None):
        config_path = Path(args.config)
        if not config_path.is_file():
            raise FileNotFoundError(f"找不到配置文件: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        project = raw.get("project", {}) or {}
        llm = raw.get("llm", {}) or {}
        pipeline = raw.get("pipeline", {}) or {}
        execution = raw.get("execution", {}) or {}  # 支持 central_hub_config.yaml 中的 execution 块

        # 解析 step: 可能在 pipeline.step 或 execution.steps
        step_value = pipeline.get("step")
        if step_value is None:
            steps_list = execution.get("steps")
            if isinstance(steps_list, list) and steps_list:
                step_value = ",".join(steps_list)

        base.update(
            {
                "project_name": project.get("name"),
                "project_features": project.get("features"),
                "target_area": project.get("target_area"),
                "total_area": project.get("total_area"),  # 用于综合大厅
                "project_location": project.get("location"),  # 用于综合大厅
                "user_project_info": project.get("user_project_info"),
                "query": project.get("query"),
                "llm_provider": llm.get("provider"),
                "llm_model": llm.get("model"),
                "step": step_value,
                "mode": pipeline.get("mode"),
                "top_k": pipeline.get("top_k"),
                "min_area": pipeline.get("min_area"),
                "max_area": pipeline.get("max_area"),
                "category": pipeline.get("category"),
                "rebuild_index": bool(pipeline.get("rebuild_index", False) or execution.get("rebuild_index", False)),
                "dry_run": bool(pipeline.get("dry_run", False) or execution.get("dry_run", False)),
                "show_contexts": bool(pipeline.get("show_contexts", False) or execution.get("verbose", False)),
            }
        )

        # 如果 target_area 未设置但 total_area 有值，自动使用 total_area
        if base["target_area"] is None and base["total_area"] is not None:
            base["target_area"] = base["total_area"]

    # 命令行覆盖 YAML
    if getattr(args, "project_name", None):
        base["project_name"] = args.project_name
    if getattr(args, "project_features", None):
        base["project_features"] = args.project_features
    if getattr(args, "target_area", None) is not None:
        base["target_area"] = args.target_area
    # 目前不从命令行覆盖 user_project_info，保持由 YAML 提供
    if getattr(args, "query", None):
        base["query"] = args.query
    if getattr(args, "llm_provider", None):
        base["llm_provider"] = args.llm_provider
    if getattr(args, "llm_model", None):
        base["llm_model"] = args.llm_model
    if getattr(args, "step", None):
        base["step"] = args.step
    if getattr(args, "mode", None):
        base["mode"] = args.mode
    if getattr(args, "top_k", None) is not None:
        base["top_k"] = args.top_k
    if getattr(args, "min_area", None) is not None:
        base["min_area"] = args.min_area
    if getattr(args, "max_area", None) is not None:
        base["max_area"] = args.max_area
    if getattr(args, "category", None):
        base["category"] = args.category
    if getattr(args, "rebuild_index", False):
        base["rebuild_index"] = True
    if getattr(args, "dry_run", False):
        base["dry_run"] = True
    if getattr(args, "show_contexts", False):
        base["show_contexts"] = True

    if not base["project_name"] or not base["project_features"]:
        raise ValueError("项目名称 --project-name 和项目特征 --project-features 必须在命令行或 YAML 中至少提供一处")

    return base


def _build_filters(args: argparse.Namespace) -> Dict[str, object]:
    filters: Dict[str, object] = {}
    if args.min_area is not None:
        filters["min_area"] = args.min_area
    if args.max_area is not None:
        filters["max_area"] = args.max_area
    if args.category:
        filters["category"] = args.category
    return filters


def _override_llm_config(config, args: argparse.Namespace):
    overrides = {}
    if getattr(args, "llm_provider", None):
        overrides["llm_provider"] = args.llm_provider
    if getattr(args, "llm_model", None):
        overrides["llm_model"] = args.llm_model
    if not overrides:
        return config
    return replace(config, **overrides)


def _resolve_steps(step_arg: str) -> List[str]:
    if not step_arg:
        return ["design"]

    lowered = step_arg.lower()
    alias_map = {
        "design": ["design"],
        "concept": ["design"],
        "exhibition": ["exhibition"],
        "public_service": ["public_service"],
        "public-service": ["public_service"],
        "service": ["public_service"],
        "central_hub": ["central_hub"],
        "central-hub": ["central_hub"],
        "central": ["central_hub"],
        "atrium": ["central_hub"],
        "hub": ["central_hub"],
        "indicators": ["indicators"],
        "indicator": ["indicators"],
        "area": ["indicators"],
        "data": ["indicators"],
        "special_theater": ["special_theater"],
        "special-theater": ["special_theater"],
        "theater": ["special_theater"],
        "cinema": ["special_theater"],
        "full": ["full"],
        "all": ["full"],
        "science_education": ["science_education"],
        "science-education": ["science_education"],
        "education": ["science_education"],
        "science": ["science_education"],
        "business_research": ["business_research"],
        "business-research": ["business_research"],
        "backoffice": ["business_research"],
        "back_of_house": ["business_research"],
        "business": ["business_research"],
        "research": ["business_research"],
        "operation": ["operation"],
        "operations": ["operation"],
        "commercial": ["operation"],
        "both": ["design", "exhibition"],
    }

    if lowered in alias_map:
        return alias_map[lowered]

    tokens = [token.strip() for token in lowered.replace("+", ",").split(",") if token.strip()]
    resolved: List[str] = []
    valid_steps = {
        "design",
        "central_hub",
        "exhibition",
        "special_theater",
        "science_education",
        "public_service",
        "operation",
        "business_research",
        "full",
    }
    for token in tokens:
        mapped = alias_map.get(token, [token])
        for item in mapped:
            if item not in valid_steps:
                continue
            if item not in resolved:
                resolved.append(item)
    return resolved or ["design"]


EXECUTION_ORDER = [
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

STEP_TITLES = {
    "design": "设计理念建议书",
    "indicators": "经济技术指标分析报告",
    "central_hub": "综合大厅与核心空间策划书",
    "exhibition": "展览空间设计要求",
    "special_theater": "特效影院区空间设计策划书",
    "science_education": "科教活动与空间融合策划书",
    "public_service": "公共服务区空间设计策划书",
    "business_research": "业务科研与后勤策划书",
    "operation": "商业与运营体系策划书",
    "full": "建筑设计任务书",
}


def _log_request(step: str, project_name: str, project_features: str, query: Optional[str]):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg(
            "user",
            f"{step.title()} brief: {project_name}",
            meta={"features": project_features, "query": query},
        )
    except Exception:  # noqa: BLE001
        pass


def _log_response(step: str, response: str):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg("assistant", f"[{step}] {response}")
    except Exception:  # noqa: BLE001
        pass


def _print_contexts(contexts: List[Document]):
    print("📚 检索上下文摘要:")
    for idx, doc in enumerate(contexts, 1):
        meta = doc.metadata or {}
        area_meta = meta.get('total_area') or meta.get('total_area_num')
        print(
            f"[{idx}] {meta.get('project_name', meta.get('doc_name', '片段'))} | 来源: {meta.get('source_type')} | 面积: {area_meta}"
        )
        snippet = doc.page_content.strip()
        snippet = snippet[:300] + "..." if len(snippet) > 300 else snippet
        print(snippet)
        print("-" * 40)


def _execute_step(step_name: str, args: argparse.Namespace, filters: Dict[str, object]) -> Dict[str, object]:
    # 从 args 中抽取已合并的配置（在 main 中注入）
    cfg: Dict[str, Any] = getattr(args, "_merged_config", {}) or {}
    project_name = cfg.get("project_name", getattr(args, "project_name", None))
    project_features = cfg.get("project_features", getattr(args, "project_features", None))
    query = cfg.get("query", getattr(args, "query", None))
    top_k = cfg.get("top_k", getattr(args, "top_k", None))
    target_area = cfg.get("target_area", getattr(args, "target_area", None))
    user_project_info = cfg.get("user_project_info", None)

    if step_name == "design":
        design_config: DesignConceptConfig = _override_llm_config(
            DEFAULT_DESIGN_CONCEPT_CONFIG,
            args,
        )
        design_generator = DesignConceptGenerator(design_config)
        _log_request("design", project_name, project_features, query)
        # 新实现：使用 LangGraph 图执行“溯源-对标-趋势-整合”并返回 JSON
        result = design_generator.generate(
            project_name=project_name,
            project_features=project_features,
            user_project_info=user_project_info or query or project_features or project_name,
            dry_run=args.dry_run,
        )
        # 为了兼容后续打印逻辑，这里将 json_result 再序列化为字符串形式的 response
        json_result = result.get("json_result")
        response_str = json.dumps(json_result, ensure_ascii=False, indent=2) if json_result is not None else ""
        return {
            "prompt": None,
            "contexts": result.get("retrieved_sources"),  # dry_run 时才会有
            "response": response_str,
        }

    if step_name == "indicators":
        # 推导目标面积：优先使用显式传入的 target_area，其次尝试从最小/最大面积取中值
        target_area: Optional[float] = target_area
        if target_area is None:
            if args.min_area is not None and args.max_area is not None and args.max_area >= args.min_area:
                target_area = (args.min_area + args.max_area) / 2.0
            elif args.min_area is not None:
                target_area = args.min_area
            elif args.max_area is not None:
                target_area = args.max_area
            else:
                target_area = 0.0

        _log_request("indicators", project_name, project_features, query)
        result = analyze_indicators(float(target_area)) if target_area is not None else {}
        return {"response": json.dumps(result, ensure_ascii=False)}

    if step_name == "central_hub":
        hub_config: CentralHubConfig = _override_llm_config(
            DEFAULT_CENTRAL_HUB_CONFIG,
            args,
        )
        hub_generator = CentralHubGenerator(hub_config)
        _log_request("central_hub", project_name, project_features, query)
        # 新实现：使用 LangGraph 图执行"形态溯源-规模对标-趋势分析-整合"并返回 JSON
        result = hub_generator.generate(
            project_name=project_name,
            project_features=project_features,
            total_area=target_area,  # 从用户输入或 YAML 获取的规模
            project_location=cfg.get("project_location"),
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )
        # 为了兼容后续打印逻辑，返回标准格式
        json_result = result.get("json_result")
        response_str = json.dumps(json_result, ensure_ascii=False, indent=2) if json_result is not None else ""
        return {
            "prompt": None,
            "contexts": None,  # 中间结果可通过 result["raw_text"] 访问
            "response": response_str,
        }

    if step_name == "exhibition":
        exhibition_config: ExhibitionConfig = _override_llm_config(
            DEFAULT_EXHIBITION_CONFIG,
            args,
        )
        exhibition_generator = ExhibitionGenerator(exhibition_config)
        _log_request("exhibition", project_name, project_features, query)
        return exhibition_generator.generate(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            target_area=target_area,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "special_theater":
        theater_config: SpecialTheaterConfig = _override_llm_config(
            DEFAULT_SPECIAL_THEATER_CONFIG,
            args,
        )
        theater_generator = SpecialTheaterGenerator(theater_config)
        _log_request("special_theater", project_name, project_features, query)
        return theater_generator.generate(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "science_education":
        science_config: ScienceEducationConfig = _override_llm_config(
            DEFAULT_SCIENCE_EDUCATION_CONFIG,
            args,
        )
        science_generator = ScienceEducationGenerator(science_config)
        _log_request("science_education", project_name, project_features, query)
        return science_generator.generate(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "public_service":
        service_config: PublicServiceConfig = _override_llm_config(
            DEFAULT_PUBLIC_SERVICE_CONFIG,
            args,
        )
        service_generator = PublicServiceGenerator(service_config)
        _log_request("public_service", project_name, project_features, query)
        return service_generator.generate(
            project_name=project_name,
            project_features=project_features,
            query=query,
            target_area=target_area,
            top_k=top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "business_research":
        br_config: BusinessResearchConfig = _override_llm_config(
            DEFAULT_BUSINESS_RESEARCH_CONFIG,
            args,
        )
        br_generator = BusinessResearchGenerator(br_config)
        _log_request("business_research", project_name, project_features, query)
        return br_generator.generate(
            project_name=project_name,
            project_features=project_features,
            target_area=target_area,
            top_k=top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "operation":
        business_config: BusinessResearchConfig = _override_llm_config(
            DEFAULT_BUSINESS_RESEARCH_CONFIG,
            args,
        )
        business_generator = OperationGenerator(business_config)
        _log_request("operation", project_name, project_features, query)
        return business_generator.generate(
            project_name=project_name,
            project_features=project_features,
            query=query,
            top_k=top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    raise ValueError(f"未知的步骤: {step_name}")


def _parse_json_response(response_text: Optional[str]) -> Any:
    if not response_text:
        return {}
    text = response_text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("模块输出非 JSON，按原始文本返回 (前200字): %s", text[:200])
        return text


def _resolve_output_path(project_name: str) -> Path:
    sanitized = "".join((ch if ch not in '<>:"/\\|?*' else "_") for ch in project_name).strip()
    if not sanitized:
        sanitized = "项目"
    filename = f"{sanitized}_设计任务书.md"
    return Path(__file__).resolve().parent / filename


def generate_full_brief(args: argparse.Namespace, filters: Dict[str, object]) -> None:
    """顺序模式生成完整任务书。"""
    cfg: Dict[str, Any] = getattr(args, "_merged_config", {}) or {}
    project_name = cfg.get("project_name", getattr(args, "project_name", None))
    project_features = cfg.get("project_features", getattr(args, "project_features", None))

    if args.dry_run:
        print("⚠️ 全案整合（full）暂不支持 --dry-run，请移除该参数后重试。")
        return

    context_data: Dict[str, Any] = {}
    step_errors: List[Tuple[str, str]] = []

    for step_name in EXECUTION_ORDER:
        title = STEP_TITLES.get(step_name, step_name)
        print(f"➡️ 正在生成 {title}...")
        try:
            result = _execute_step(step_name, args, filters)
        except Exception as exc:  # noqa: BLE001
            logger.exception("模块 %s 生成失败", step_name)
            step_errors.append((title, str(exc)))
            context_data[SECTION_KEY_MAP.get(step_name, step_name)] = f"生成失败: {exc}"
            continue

        response_payload = result.get("response")
        parsed_payload = _parse_json_response(response_payload)
        context_data[SECTION_KEY_MAP.get(step_name, step_name)] = parsed_payload or "(暂无内容)"

    if not context_data:
        print("⚠️ 未获取到任何模块输出，无法整合任务书。")
        return

    assembler = BriefAssemblyPipeline()
    assembly = assembler.generate_brief(project_name, project_features, context_data)
    markdown = assembly.get("response")
    if not markdown:
        print("⚠️ 整合器未返回内容，请稍后重试。")
        return

    output_path = _resolve_output_path(project_name)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ 《{project_name} 建筑设计任务书》已生成 -> {output_path}")
    _log_response("full", markdown[:2000])

    if step_errors:
        print("⚠️ 以下模块生成失败，已在任务书中标注：")
        for title, err in step_errors:
            print(f"   - {title}: {err}")


async def generate_full_brief_parallel(args: argparse.Namespace, filters: Dict[str, object]) -> None:
    """
    并行模式生成完整任务书（使用 LangGraph 图引擎）。

    所有领域模块（7个）并行执行，完成后汇聚到组装节点。
    相比顺序模式，可显著减少总执行时间。
    """
    engine = _get_graph_engine()
    create_initial_state = engine["create_initial_state"]
    run_graph = engine["run_graph"]

    print("🚀 启动并行模式（LangGraph 图引擎）")
    start_time = time.time()

    # 创建初始状态（保持签名兼容，Graph 内部从 state.input 中读取 target_area）
    cfg: Dict[str, Any] = getattr(args, "_merged_config", {}) or {}
    project_name = cfg.get("project_name", getattr(args, "project_name", None))
    project_features = cfg.get("project_features", getattr(args, "project_features", None))
    query = cfg.get("query", getattr(args, "query", None))

    initial_state = create_initial_state(
        project_name=project_name,
        project_features=project_features,
        query=query,
        top_k=args.top_k,
        rebuild_index=args.rebuild_index,
        dry_run=args.dry_run,
        filters=filters if filters else None,
        llm_provider=getattr(args, "llm_provider", None),
        llm_model=getattr(args, "llm_model", None),
        target_area=cfg.get("target_area", getattr(args, "target_area", None)),
        total_area=cfg.get("total_area", None),
        project_location=cfg.get("project_location", None),
        user_project_info=cfg.get("user_project_info", None),
    )

    # 注入 target_area 到初始状态的 input 字段，供 indicators 节点使用
    try:
        if isinstance(initial_state, dict):
            input_payload = initial_state.get("input") or {}
            if not isinstance(input_payload, dict):
                input_payload = {}
            input_payload.setdefault("project_name", project_name)
            input_payload.setdefault("project_features", project_features)
            if query is not None:
                input_payload.setdefault("query", query)
            input_payload["target_area"] = cfg.get("target_area", getattr(args, "target_area", None))
            input_payload["total_area"] = cfg.get("total_area", None)
            input_payload["project_location"] = cfg.get("project_location", None)
            if cfg.get("user_project_info"):
                input_payload["user_project_info"] = cfg.get("user_project_info")
            if getattr(args, "llm_provider", None):
                input_payload["llm_provider"] = args.llm_provider
            if getattr(args, "llm_model", None):
                input_payload["llm_model"] = args.llm_model
            initial_state["input"] = input_payload
    except Exception:  # noqa: BLE001
        logger.exception("无法在初始状态中注入 target_area，将继续使用默认图配置")

    # 执行图
    print("⏳ 并行生成所有模块内容...")
    final_state = await run_graph(initial_state)
    elapsed = time.time() - start_time

    # 检查错误
    step_errors: List[Tuple[str, str]] = []
    for module_key in EXECUTION_ORDER:
        output = final_state.get(module_key, {})
        if output.get("error"):
            title = STEP_TITLES.get(module_key, module_key)
            step_errors.append((title, output["error"]))

    # 获取组装结果
    markdown = final_state.get("assembled_brief", "")
    if not markdown:
        print("⚠️ 整合器未返回内容，请稍后重试。")
        return

    # 1) 输出 Markdown
    output_path = _resolve_output_path(project_name)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ 《{project_name} 建筑设计任务书》已生成 -> {output_path}")
    print(f"⏱️ 并行执行总耗时: {elapsed:.1f} 秒")
    _log_response("full", markdown[:2000])

    # 2) 输出与 MD 同名的大 JSON，记录各模块解析后的结果
    try:
        import json
        from pipelines.graph_engine.state import MODULE_STATE_KEYS, SECTION_KEY_MAP
        from pipelines.graph_engine.nodes import _parse_json_response  # type: ignore

        modules_payload: Dict[str, Any] = {}
        for module_key in MODULE_STATE_KEYS:
            output = final_state.get(module_key, {}) or {}
            section_key = SECTION_KEY_MAP.get(module_key, module_key)

            if output.get("error"):
                modules_payload[section_key] = {"error": output.get("error")}
            elif output.get("response"):
                parsed = _parse_json_response(output.get("response"))
                modules_payload[section_key] = parsed
            else:
                modules_payload[section_key] = None

        json_path = output_path.with_suffix(".json")
        json_path.write_text(json.dumps(modules_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🧩 模块级 JSON 输出已生成 -> {json_path}")
    except Exception as exc:  # 只记录错误，不影响主流程
        logger.exception("写入模块级 JSON 输出时出错: %s", exc)

    if step_errors:
        print("⚠️ 以下模块生成失败，已在任务书中标注：")
        for title, err in step_errors:
            print(f"   - {title}: {err}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = _parse_args()
    merged = _load_yaml_config(args)
    # 将合并后的配置挂到 args 上，供后续函数统一访问
    setattr(args, "_merged_config", merged)

    # 也将 YAML 中的 llm_provider / llm_model 回填到 args，
    # 这样 _override_llm_config 能覆盖默认 minimax 配置
    if merged.get("llm_provider"):
        args.llm_provider = merged["llm_provider"]
    if merged.get("llm_model"):
        args.llm_model = merged["llm_model"]

    filters = _build_filters(args)

    requested_steps: List[str] = _resolve_steps(merged.get("step", getattr(args, "step", "design")))

    if "full" in requested_steps:
        # 全案整合模式：根据 --mode 选择执行方式
        mode = merged.get("mode", args.mode)
        if mode == "parallel":
            asyncio.run(generate_full_brief_parallel(args, filters))
        else:
            generate_full_brief(args, filters)
        return

    # 单步/多步模式：暂不支持并行，使用顺序执行
    mode = merged.get("mode", args.mode)
    if mode == "parallel":
        print("ℹ️ 并行模式仅支持 --step full，当前自动降级为顺序模式")

    steps: List[str] = [step for step in EXECUTION_ORDER if step in requested_steps]
    if not steps:
        steps = [step for step in requested_steps if step != "full"] or ["design"]

    results: Dict[str, Dict[str, object]] = {}

    for step_name in steps:
        results[step_name] = _execute_step(step_name, args, filters)

    for step_name in steps:
        result = results.get(step_name)
        if not result:
            continue

        contexts = result.get("contexts", [])
        prompt = result.get("prompt")
        response = result.get("response")
        cfg: Dict[str, Any] = merged
        project_name = cfg.get("project_name", args.project_name)
        project_features = cfg.get("project_features", args.project_features)
        title = STEP_TITLES.get(step_name, step_name)

        print("=" * 80)
        print(f"🎯 项目: {project_name} | 步骤: {title}")
        print(f"🧭 特征: {project_features}")
        print("=" * 80)

        if args.show_contexts and contexts:
            _print_contexts(contexts)

        if args.dry_run:
            print("🧱 DRY RUN (未调用模型)")
            if prompt:
                print("System Prompt:\n", prompt["system_prompt"])
                print("\nUser Prompt:\n", prompt["user_prompt"])
            continue

        if not response:
            print("⚠️ 未获得模型输出")
            continue

        print(f"🧠 {title} (JSON):\n")
        print(response)
        _log_response(step_name, str(response))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        logger.exception("设计理念生成模块运行失败: %s", exc)
        print(f"系统错误: {exc}")
        sys.exit(1)