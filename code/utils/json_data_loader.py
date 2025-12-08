"""JSON case data loader.

Converts loose ArchDaily/China JSON files into a structured pandas DataFrame
with cleaned numeric areas for downstream analytics and retrieval.
"""

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from config import DATA_ROOT

logger = logging.getLogger(__name__)

ARCHDAILY_DIR = DATA_ROOT / "archdaily"
CHINA_DIR = DATA_ROOT / "china"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("读取 JSON 失败 %s: %s", path, exc)
        return {}


def _get_first(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    lowered = {k.lower(): v for k, v in data.items()}
    for key in keys:
        val = data.get(key)
        if val is not None:
            return val
        val = lowered.get(key.lower())
        if val is not None:
            return val
    return None


def _parse_description(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "\n".join(str(item) for item in raw if item is not None)
    return str(raw)


def _parse_architects(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, list):
        return "; ".join(str(item) for item in raw if item is not None)
    return str(raw)


def _normalize_area(value: Any) -> Optional[float]:
    """Convert area strings like "45,000 m2" or "3000 sqm" to square meters."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    lower = text.lower()
    multiplier = 1.0

    if re.search(r"\bhectare\b", lower) or re.search(r"\bha\b", lower):
        multiplier = 10000.0
    elif "sq ft" in lower or "sqft" in lower or "ft2" in lower or "ft^2" in lower or "square foot" in lower:
        multiplier = 0.092903
    elif "km2" in lower or "km^2" in lower:
        multiplier = 1_000_000.0

    match = re.search(r"[-+]?\d[\d.,]*", lower)
    if not match:
        return None

    num_str = match.group(0).replace(",", "")
    try:
        base = float(num_str)
    except ValueError:
        return None

    return base * multiplier


def _build_record(payload: Dict[str, Any], source: Path) -> Dict[str, Any]:
    city = _get_first(payload, ["City", "city"])
    country = _get_first(payload, ["Country", "country"])
    location = "".join(
        part for part in [str(city or "").strip(), str(country or "").strip()] if part
    )

    if city and country:
        location = f"{city}, {country}"
    elif city:
        location = str(city)
    elif country:
        location = str(country)

    project_name = _get_first(
        payload,
        ["Project Title", "project_title", "project_name", "name", "title"],
    )
    architects = _parse_architects(
        _get_first(payload, ["Architects", "architects", "设计单位"])
    )
    year = _get_first(payload, ["Year", "year", "opening_date"])

    area_raw = _get_first(
        payload,
        ["Area", "area", "total_construction_area_sqm", "建筑面积", "建筑面积（㎡）"],
    )
    area_clean = _normalize_area(area_raw)

    description = _parse_description(
        _get_first(
            payload,
            ["Description", "description", "concept&appearance", "设计理念"],
        )
    )

    return {
        "project_name": project_name or "",
        "location": location,
        "architects": architects,
        "year": year or "",
        "area": area_clean,
        "area_raw": area_raw,
        "description": description,
        "source": str(source),
    }


def _load_directory(root: Path) -> List[Dict[str, Any]]:
    if not root.exists():
        logger.warning("案例目录不存在: %s", root)
        return []

    records: List[Dict[str, Any]] = []
    for json_file in sorted(root.glob("*.json")):
        payload = _read_json(json_file)
        if not payload:
            continue
        records.append(_build_record(payload, json_file))
    return records


@lru_cache(maxsize=1)
def load_case_database() -> pd.DataFrame:
    """Load ArchDaily + China case JSON into a cached DataFrame."""

    rows: List[Dict[str, Any]] = []
    for folder in (ARCHDAILY_DIR, CHINA_DIR):
        rows.extend(_load_directory(folder))

    if not rows:
        logger.warning("未找到案例数据，返回空 DataFrame")
        return pd.DataFrame(
            columns=["project_name", "location", "architects", "year", "area", "area_raw", "description", "source"]
        )

    df = pd.DataFrame(rows)
    # Ensure area column is float for numeric operations
    df["area"] = pd.to_numeric(df["area"], errors="coerce")
    return df


__all__ = ["load_case_database"]
