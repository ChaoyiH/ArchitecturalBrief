"""Entry point for design concept & exhibition-space generators."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from pipelines.domains.business_research_pipeline import BusinessResearchGenerator
from pipelines.domains.central_hub_pipeline import CentralHubGenerator
from pipelines.domains.design_concept_pipeline import DesignConceptGenerator
from pipelines.domains.exhibition_pipeline import ExhibitionGenerator
from pipelines.domains.public_service_pipeline import PublicServiceGenerator
from pipelines.domains.special_theater_pipeline import SpecialTheaterGenerator
from pipelines.domains.science_education_pipeline import ScienceEducationGenerator
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

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
    parser.add_argument("--project-name", required=True, help="项目名称或类型")
    parser.add_argument(
        "--project-features",
        required=True,
        help="项目关键特征或背景描述（例如选址、规模、体验重点等）",
    )
    parser.add_argument("--query", help="自定义检索查询语句（默认使用项目特征）")
    parser.add_argument("--top-k", type=int, default=None, help="检索案例数量")
    parser.add_argument("--min-area", type=float, help="按建筑面积下限过滤")
    parser.add_argument("--max-area", type=float, help="按建筑面积上限过滤")
    parser.add_argument("--category", help="按项目类别过滤")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建所选步骤的向量索引",
    )
    parser.add_argument(
        "--step",
        default="design",
        help=(
            "指定生成阶段，可选 design / central_hub / exhibition / special_theater / science_education / public_service / business_research / both / all / full（全案整合），"
            "或以逗号分隔组合"
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
    return parser.parse_args()


def _build_filters(args: argparse.Namespace) -> Dict[str, object]:
    filters: Dict[str, object] = {}
    if args.min_area is not None:
        filters["min_area"] = args.min_area
    if args.max_area is not None:
        filters["max_area"] = args.max_area
    if args.category:
        filters["category"] = args.category
    return filters


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
        "business": ["business_research"],
        "research": ["business_research"],
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
    "central_hub",
    "exhibition",
    "special_theater",
    "science_education",
    "public_service",
    "business_research",
]

SECTION_KEY_MAP = {
    "design": "concept",
    "central_hub": "central_hub",
    "exhibition": "exhibition",
    "special_theater": "special_theater",
    "science_education": "science_education",
    "public_service": "public_service",
    "business_research": "business_research",
}

STEP_TITLES = {
    "design": "设计理念建议书",
    "central_hub": "综合大厅与核心空间策划书",
    "exhibition": "展览空间设计要求",
    "special_theater": "特效影院区空间设计策划书",
    "science_education": "科教活动与空间融合策划书",
    "public_service": "公共服务区空间设计策划书",
    "business_research": "业务科研区空间设计策划书",
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
    if step_name == "design":
        design_config: DesignConceptConfig = DEFAULT_DESIGN_CONCEPT_CONFIG
        design_generator = DesignConceptGenerator(design_config)
        design_generator.ensure_index(rebuild=args.rebuild_index)
        _log_request("design", args.project_name, args.project_features, args.query)
        return design_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            filters=filters if filters else None,
            dry_run=args.dry_run,
        )

    if step_name == "central_hub":
        hub_config: CentralHubConfig = DEFAULT_CENTRAL_HUB_CONFIG
        hub_generator = CentralHubGenerator(hub_config)
        _log_request("central_hub", args.project_name, args.project_features, args.query)
        return hub_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "exhibition":
        exhibition_config: ExhibitionConfig = DEFAULT_EXHIBITION_CONFIG
        exhibition_generator = ExhibitionGenerator(exhibition_config)
        _log_request("exhibition", args.project_name, args.project_features, args.query)
        return exhibition_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "special_theater":
        theater_config: SpecialTheaterConfig = DEFAULT_SPECIAL_THEATER_CONFIG
        theater_generator = SpecialTheaterGenerator(theater_config)
        _log_request("special_theater", args.project_name, args.project_features, args.query)
        return theater_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "science_education":
        science_config: ScienceEducationConfig = DEFAULT_SCIENCE_EDUCATION_CONFIG
        science_generator = ScienceEducationGenerator(science_config)
        _log_request("science_education", args.project_name, args.project_features, args.query)
        return science_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "public_service":
        service_config: PublicServiceConfig = DEFAULT_PUBLIC_SERVICE_CONFIG
        service_generator = PublicServiceGenerator(service_config)
        _log_request("public_service", args.project_name, args.project_features, args.query)
        return service_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
            rebuild_index=args.rebuild_index,
            dry_run=args.dry_run,
        )

    if step_name == "business_research":
        business_config: BusinessResearchConfig = DEFAULT_BUSINESS_RESEARCH_CONFIG
        business_generator = BusinessResearchGenerator(business_config)
        _log_request("business_research", args.project_name, args.project_features, args.query)
        return business_generator.generate(
            project_name=args.project_name,
            project_features=args.project_features,
            query=args.query,
            top_k=args.top_k,
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
    assembly = assembler.generate_brief(args.project_name, args.project_features, context_data)
    markdown = assembly.get("response")
    if not markdown:
        print("⚠️ 整合器未返回内容，请稍后重试。")
        return

    output_path = _resolve_output_path(args.project_name)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"✅ 《{args.project_name} 建筑设计任务书》已生成 -> {output_path}")
    _log_response("full", markdown[:2000])

    if step_errors:
        print("⚠️ 以下模块生成失败，已在任务书中标注：")
        for title, err in step_errors:
            print(f"   - {title}: {err}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = _parse_args()
    filters = _build_filters(args)

    requested_steps: List[str] = _resolve_steps(args.step)

    if "full" in requested_steps:
        generate_full_brief(args, filters)
        return

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
        title = STEP_TITLES.get(step_name, step_name)

        print("=" * 80)
        print(f"🎯 项目: {args.project_name} | 步骤: {title}")
        print(f"🧭 特征: {args.project_features}")
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