"""CLI entry for Virtual Hearing Engine (Brief Reviewer).

Usage example:
    python brief_reviewer_main.py --input_file path/to/brief.md --output_file out.json --sample 5
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Union

import yaml

from core.review_engine import ReviewEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual Hearing Engine - Brief Reviewer")
    parser.add_argument("--config", help="Path to YAML config for reviewer")
    parser.add_argument("--input_file", help="Path to the Markdown brief file")
    parser.add_argument("--output_file", help="Path to save JSON review output")
    parser.add_argument(
        "--sample",
        default=None,
        help="Number of personas to simulate (int) or 'all'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: dict = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    review_cfg = cfg.get("review", {}) if isinstance(cfg, dict) else {}
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}

    input_file = args.input_file or review_cfg.get("input_file")
    output_file = args.output_file or review_cfg.get("output_file")
    sample_val = args.sample if args.sample not in (None, "") else review_cfg.get("sample", "5")

    if not input_file:
        raise ValueError("input_file is required (via --input_file or config.review.input_file)")
    if not output_file:
        raise ValueError("output_file is required (via --output_file or config.review.output_file)")

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        md_content = f.read()

    sample_arg: Union[int, str]
    if isinstance(sample_val, str) and str(sample_val).lower() == "all":
        sample_arg = "all"
    else:
        try:
            sample_arg = int(sample_val)
        except ValueError:
            sample_arg = 5
            logger.warning("Invalid sample value '%s', defaulting to 5", sample_val)

    provider = llm_cfg.get("provider") if isinstance(llm_cfg, dict) else None
    model = llm_cfg.get("model") if isinstance(llm_cfg, dict) else None

    engine = ReviewEngine(provider=provider, model=model)
    results = engine.run(md_content, n=sample_arg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Saved review results to %s", output_path)


if __name__ == "__main__":
    main()
