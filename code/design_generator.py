"""Entry point for the Design Concept generation module."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Optional

from config import DEFAULT_DESIGN_CONCEPT_CONFIG, DesignConceptConfig
from rag_modules.design_concept_pipeline import DesignConceptGenerator

logger = logging.getLogger(__name__)

# 尝试启用统一日志/消息捕获
try:  # pragma: no cover - best effort logging bootstrap
    from log_setup import setup as _log_setup, record_message as _record_msg

    _log_setup()
except Exception:  # noqa: BLE001
    _record_msg = None  # type: ignore[assignment]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design Concept (任务书) 生成模块入口"
    )
    parser.add_argument("--project-name", required=True, help="项目名称或类型")
    parser.add_argument(
        "--project-features",
        required=True,
        help="项目关键特征或背景描述（例如选址、规模、体验重点等）",
    )
    parser.add_argument(
        "--query",
        help="自定义检索查询语句（默认使用项目特征）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="检索案例数量（默认读取配置 top_k）",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        help="按建筑面积下限过滤（单位：与数据一致）",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        help="按建筑面积上限过滤",
    )
    parser.add_argument(
        "--category",
        help="按项目类别（如 museum、science museum 等）过滤",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="强制重建设计理念向量索引",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅输出提示词与检索内容，不调用大模型",
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


def _log_request(project_name: str, project_features: str, query: Optional[str]):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg(
            "user",
            f"Design Concept brief: {project_name}",
            meta={"features": project_features, "query": query},
        )
    except Exception:  # noqa: BLE001
        pass


def _log_response(response: str):
    if '_record_msg' not in globals() or _record_msg is None:
        return
    try:
        _record_msg("assistant", response)
    except Exception:  # noqa: BLE001
        pass


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = _parse_args()
    config: DesignConceptConfig = DEFAULT_DESIGN_CONCEPT_CONFIG

    generator = DesignConceptGenerator(config)
    generator.ensure_index(rebuild=args.rebuild_index)

    filters = _build_filters(args)
    _log_request(args.project_name, args.project_features, args.query)

    result = generator.generate(
        project_name=args.project_name,
        project_features=args.project_features,
        query=args.query,
        top_k=args.top_k,
        filters=filters if filters else None,
        dry_run=args.dry_run,
    )

    contexts = result.get("contexts", [])
    prompt = result.get("prompt")
    response = result.get("response")

    print("=" * 80)
    print(f"🎯 项目: {args.project_name}")
    print(f"🧭 特征: {args.project_features}")
    print("=" * 80)

    if args.show_contexts and contexts:
        print("📚 检索上下文摘要:")
        for idx, doc in enumerate(contexts, 1):
            meta = doc.metadata or {}
            print(f"[{idx}] {meta.get('project_name', '未知案例')} | 来源: {meta.get('source_type')} | 面积: {meta.get('total_area') or meta.get('total_area_num')}")
            snippet = doc.page_content.strip()
            snippet = snippet[:300] + "..." if len(snippet) > 300 else snippet
            print(snippet)
            print("-" * 40)

    if args.dry_run:
        print("🧱 DRY RUN (未调用模型)")
        if prompt:
            print("System Prompt:\n", prompt["system_prompt"])
            print("\nUser Prompt:\n", prompt["user_prompt"])
        return

    if not response:
        print("⚠️ 未获得模型输出")
        return

    print("🧠 设计理念建议书 (JSON):\n")
    print(response)
    _log_response(str(response))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001
        logger.exception("设计理念生成模块运行失败: %s", exc)
        print(f"系统错误: {exc}")
        sys.exit(1)