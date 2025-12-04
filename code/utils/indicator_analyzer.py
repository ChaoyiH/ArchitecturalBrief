"""
经济技术指标分析模块 (Indicator Analyzer)
==========================================

用于建筑策划 RAG 系统的数值推理，包含：
1. 鲁棒的面积数值提取
2. 基于《科学技术馆建设标准》的分级判定
3. 同级统计与相似案例匹配

Usage:
    from utils.indicator_analyzer import analyze_indicators
    
    result = analyze_indicators(target_area=25000)
    print(result["target_classification"])  # {'class_name': '大型馆', 'design_life': 100}
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 动态路径修复：确保可以从 utils/ 中导入上层的 config.py 等模块
# ---------------------------------------------------------------------------

current_file = Path(__file__).resolve()
# 当前文件位于 code/utils/indicator_analyzer.py → project_root = code/
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 尝试安全加载配置中的数据根目录
try:
    from config import DATA_ROOT as _DATA_ROOT
    DATA_ROOT = _DATA_ROOT
except Exception:  # noqa: BLE001 - 配置缺失时使用回退路径
    # 回退到 code/../data 目录，并给出日志警告
    DATA_ROOT = (project_root.parent / "data").resolve()

from utils.log_setup import setup as setup_logging

logger = logging.getLogger(__name__)


# =============================================================================
# 0. 数字法规库与行业参考 (Dual-Track Knowledge Base)
# =============================================================================

DIGITAL_STANDARD: Dict[str, Any] = {
    "source": "《科学技术馆建设标准》(建标 101-2007)",
    "classification": {
        "extra_large": {"name": "特大型馆", "min_area": 30000.0, "max_area": None, "design_life": 100},
        "large": {"name": "大型馆", "min_area": 15000.0, "max_area": 30000.0, "design_life": 100},
        "medium": {"name": "中型馆", "min_area": 8000.0, "max_area": 15000.0, "design_life": 50},
        "small": {"name": "小型馆", "min_area": 0.0, "max_area": 8000.0, "design_life": 50},
    },
    "function_ratios": {
        # 百分比区间，均为占总建筑面积的比例
        "特大型馆": {
            "exhibition_education": (55.0, 60.0),  # 展览教育用房
            "public_service": (15.0, 20.0),        # 公共服务用房
            "business_research": (10.0, 15.0),     # 业务研究用房
            "management": (10.0, 15.0),            # 管理保障用房
        },
        "大型馆": {
            "exhibition_education": (60.0, 65.0),
            "public_service": (10.0, 15.0),
            "business_research": (10.0, 15.0),
            "management": (10.0, 15.0),
        },
        "中型馆": {
            "exhibition_education": (65.0, 70.0),
            "public_service": (5.0, 10.0),
            "business_research": (5.0, 10.0),
            "management": (15.0, 20.0),
        },
        "小型馆": {
            "exhibition_education": (65.0, 75.0),
            "public_service": (5.0, 10.0),
            "business_research": (5.0, 10.0),
            "management": (10.0, 20.0),
        },
    },
    "technical_parameters": {
        "exhibit_density_sqm_per_piece": (15.0, 30.0),  # m²/件
        "instantaneous_people_per_sqm": (0.2, 0.25),    # 人/m²（基于展厅面积）
        "permanent_exhibition_min_sqm": 3000.0,
        "recommended_min_total_area": 5000.0,
    },
}

INDUSTRY_BENCHMARKS: Dict[str, Tuple[float, float]] = {
    "floor_area_ratio": (0.4, 3.2),   # 容积率经验区间
    "building_density": (15.0, 55.0), # 建筑密度 %
    "green_ratio": (10.0, 76.0),      # 绿地率 %
}


# =============================================================================
# 1. 鲁棒的数值提取 (Robust Parsing)
# =============================================================================

def parse_area_value(value: Any) -> Optional[float]:
    """
    从各种格式的输入中提取面积数值（单位：平方米）。
    
    支持的输入格式：
    - int / float: 直接返回
    - str: "20,000 sqm", "1.5 hectares", "3500m²", "21367 m²", "100600" 等
    
    核心约束：
    - 若包含 "m²" 或 "m2"，严禁将单位中的 "2" 识别为数值的一部分
    - 使用正则表达式提取第一个连续的数值序列（支持千分位逗号和小数点）
    - 若无法提取有效数值，返回 None
    
    Args:
        value: 任意类型的输入值
        
    Returns:
        float: 面积数值（平方米），或 None（无法解析时）
    """
    if value is None:
        return None
    
    # 直接处理数值类型
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    
    if not isinstance(value, str):
        return None
    
    # 清理字符串
    text = value.strip()
    if not text:
        return None
    
    # 预处理：移除单位标识符，避免 "m²" 或 "m2" 中的 "2" 被误识别
    # 先检测并记录是否有公顷单位（需要转换）
    is_hectares = bool(re.search(r'\bhectares?\b', text, re.IGNORECASE))
    
    # 移除常见单位后缀（在提取数值前）
    # 注意：
    # - m² 中的 ² 是 Unicode U+00B2 (SUPERSCRIPT TWO)
    # - ㎡ 是 Unicode U+33A1 (SQUARE M SQUARED)
    # - 必须在提取数值前移除，避免误识别
    text_cleaned = re.sub(
        r'\s*m[\u00b2\u00B2]|\s*m2\b|\s*sqm\b|\s*sq\.?\s*m\b|\s*square\s*met(?:er|re)s?|\s*平方米|\s*㎡',
        ' ',
        text,
        flags=re.IGNORECASE
    )
    text_cleaned = re.sub(
        r'\s*(hectares?|公顷|ha)\s*',
        ' ',
        text_cleaned,
        flags=re.IGNORECASE
    )
    
    # 移除可能残留的上标字符（如单独的 ²）
    text_cleaned = re.sub(r'[\u00b2\u00B2]', '', text_cleaned)
    
    # 提取第一个连续的数值序列
    # 支持：千分位逗号 (20,000)、小数点 (3.5)
    # 模式：可选负号 + 整数部分（可含千分位逗号）+ 可选小数部分
    # 注意：必须先匹配 \d+ 再匹配千分位格式，否则 "21367" 会只匹配到 "213"
    pattern = r'-?(?:\d+|\d{1,3}(?:,\d{3})*)(?:\.\d+)?'
    
    match = re.search(pattern, text_cleaned)
    if not match:
        return None
    
    # 解析匹配到的数值字符串
    num_str = match.group()
    # 移除千分位逗号
    num_str = num_str.replace(',', '')
    
    try:
        result = float(num_str)
    except ValueError:
        return None
    
    # 如果原始输入是公顷，转换为平方米
    if is_hectares:
        result *= 10000
    
    return result if result > 0 else None


def get_size_class(area_sqm: float) -> Dict[str, Any]:
    """
    根据面积判定科技馆等级（依据《科学技术馆建设标准》）。
    
    分级标准（依据《科学技术馆建设标准》）：
    - 特大型馆：面积 > 30,000 m² (设计使用年限: 100年)
    - 大型馆：15,000 m² < 面积 ≤ 30,000 m² (设计使用年限: 100年)
    - 中型馆：8,000 m² < 面积 ≤ 15,000 m² (设计使用年限: 50年)
    - 小型馆：面积 ≤ 8,000 m² (设计使用年限: 50年)
    
    Args:
        area_sqm: 建筑面积（平方米）
        
    Returns:
        dict: {
            "class_name": str,      # 等级名称
            "design_life": int,     # 设计使用年限（年）
            "area_range": str,      # 面积范围描述
        }
    """
    if area_sqm is None or area_sqm <= 0:
        return {
            "class_name": "未知",
            "design_life": None,
            "area_range": "无效面积",
        }
    
    classification = DIGITAL_STANDARD["classification"]

    if area_sqm > classification["extra_large"]["min_area"]:
        info = classification["extra_large"]
        return {
            "class_name": info["name"],
            "design_life": info["design_life"],
            "area_range": f"> {int(info['min_area']):,} m²",
        }

    if classification["large"]["min_area"] < area_sqm <= classification["large"]["max_area"]:
        info = classification["large"]
        return {
            "class_name": info["name"],
            "design_life": info["design_life"],
            "area_range": "15,000 ~ 30,000 m²",
        }

    if classification["medium"]["min_area"] < area_sqm <= classification["medium"]["max_area"]:
        info = classification["medium"]
        return {
            "class_name": info["name"],
            "design_life": info["design_life"],
            "area_range": "8,000 ~ 15,000 m²",
        }

    info = classification["small"]
    return {
        "class_name": info["name"],
        "design_life": info["design_life"],
        "area_range": "≤ 8,000 m²",
    }


# =============================================================================
# 3. 数据加载与案例结构
# =============================================================================

@dataclass
class BuildingCase:
    """建筑案例数据结构"""
    name: str
    area: Optional[float]  # 建筑面积（平方米）
    year: Optional[int]
    filepath: str
    source: str  # "archdaily", "china", "world"
    height: Optional[float] = None  # 建筑总高度（米）
    size_class: Optional[str] = None  # 等级名称
    
    def __post_init__(self):
        """初始化后自动计算等级"""
        if self.area is not None:
            self.size_class = get_size_class(self.area)["class_name"]


def _extract_year(data: Dict[str, Any]) -> Optional[int]:
    """从 JSON 数据中提取年份"""
    year_fields = ["Year", "year", "opening_date", "建成年份"]
    
    for key in year_fields:
        if key in data:
            value = data[key]
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                # 尝试提取四位数年份
                match = re.search(r'\b(19|20)\d{2}\b', value)
                if match:
                    return int(match.group())
    return None


def _extract_area(data: Dict[str, Any]) -> Optional[float]:
    """从 JSON 数据中提取面积"""
    area_fields = [
        "Area",
        "area", 
        "total_construction_area",
        "total_construction_area_sqm",
        "gross_floor_area",
        "建筑面积",
        "总建筑面积",
    ]
    
    for key in area_fields:
        if key in data:
            area = parse_area_value(data[key])
            if area is not None:
                return area
    return None


def _extract_height(data: Dict[str, Any]) -> Optional[float]:
    """从 JSON 数据中提取建筑总高度（米）。

    支持字段名：Height, height, 建筑高度, building_height 等；
    支持单位：m, meter(s), 米；若包含 ft/feet，则自动换算为米 (1 ft = 0.3048 m)。
    若解析出的高度 < 3m 或 > 800m，则视为异常值并丢弃。
    """

    height_fields = [
        "Height",
        "height",
        "建筑高度",
        "building_height",
    ]

    raw_value: Any | None = None
    for key in height_fields:
        if key in data and data[key] is not None:
            raw_value = data[key]
            break

    if raw_value is None:
        return None

    # 统一转为字符串以便正则解析
    if isinstance(raw_value, (int, float)):
        text = str(raw_value)
    else:
        text = str(raw_value).strip()

    if not text:
        return None

    lower = text.lower()

    # 是否为英尺单位
    is_feet = "ft" in lower or "feet" in lower

    # 提取第一个连续数值（整数或小数，支持千分位）
    match = re.search(r"-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?", lower)
    if not match:
        return None

    num_str = match.group().replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return None

    if is_feet:
        value *= 0.3048

    # 过滤异常高度：过低/过高
    if value < 3 or value > 800:
        return None

    return value


def _extract_name(data: Dict[str, Any], filepath: Path) -> str:
    """从 JSON 数据中提取项目名称"""
    name_fields = ["Project Title", "name", "名称", "项目名称"]
    
    for key in name_fields:
        if key in data and data[key]:
            return str(data[key])
    
    # 使用文件名作为备选
    return filepath.stem


def _clamp_percentage_range(rng: Tuple[float, float]) -> Tuple[float, float]:
    """归一化百分比区间，确保有序且在 0~100 之间。"""

    low, high = rng
    low = max(0.0, min(100.0, low))
    high = max(0.0, min(100.0, high))
    if high < low:
        low, high = high, low
    return low, high


def _get_classification_entry(name: str) -> Optional[Dict[str, Any]]:
    for entry in DIGITAL_STANDARD["classification"].values():
        if entry["name"] == name:
            return entry
    return None


def calculate_compliance_specs(target_area: float, size_class_name: str) -> Dict[str, Any]:
    """依据《科学技术馆建设标准》推演功能配比与承载力。"""

    total_area = max(0.0, float(target_area or 0))
    ratios_all = DIGITAL_STANDARD["function_ratios"]
    tech = DIGITAL_STANDARD["technical_parameters"]
    func_ratios = ratios_all.get(size_class_name)

    size_info = _get_classification_entry(size_class_name)

    if func_ratios is None:
        return {
            "standard_source": DIGITAL_STANDARD["source"],
            "size_class_name": size_class_name,
            "design_life": size_info.get("design_life") if size_info else None,
            "total_area_sqm": round(total_area, 1),
            "function_area_ranges": {},
            "exhibition_area_estimate": None,
            "exhibit_count_estimate": None,
            "instantaneous_capacity": None,
            "warnings": [f"未找到等级 '{size_class_name}' 的功能占比配置"],
        }

    function_area_ranges: Dict[str, Dict[str, float]] = {}
    for key, pct_range in func_ratios.items():
        pct_min, pct_max = _clamp_percentage_range(pct_range)
        area_min = total_area * pct_min / 100.0
        area_max = total_area * pct_max / 100.0
        function_area_ranges[key] = {
            "percent_min": round(pct_min, 1),
            "percent_max": round(pct_max, 1),
            "area_min_sqm": round(area_min, 1),
            "area_max_sqm": round(area_max, 1),
        }

    exhibition_area_estimate: Optional[float] = None
    if "exhibition_education" in function_area_ranges:
        fr = function_area_ranges["exhibition_education"]
        exhibition_area_estimate = (fr["area_min_sqm"] + fr["area_max_sqm"]) / 2.0

    exhibit_counts: Optional[Dict[str, int]] = None
    if exhibition_area_estimate and exhibition_area_estimate > 0:
        density_min, density_max = tech["exhibit_density_sqm_per_piece"]
        if density_min > 0 and density_max > 0:
            exhibit_counts = {
                "min": int(exhibition_area_estimate / density_max),
                "max": int(exhibition_area_estimate / density_min),
            }

    instantaneous_capacity: Optional[Dict[str, int]] = None
    if exhibition_area_estimate and exhibition_area_estimate > 0:
        occ_min, occ_max = tech["instantaneous_people_per_sqm"]
        instantaneous_capacity = {
            "min": int(exhibition_area_estimate * occ_min),
            "max": int(exhibition_area_estimate * occ_max),
        }

    warnings: List[str] = []
    if total_area < tech["recommended_min_total_area"]:
        warnings.append("建筑面积低于推荐下限 5,000 m²，需谨慎论证功能配置")
    if exhibition_area_estimate is not None and exhibition_area_estimate < tech["permanent_exhibition_min_sqm"]:
        warnings.append("估算展厅面积低于常设展厅 3,000 m² 的规范要求")

    return {
        "standard_source": DIGITAL_STANDARD["source"],
        "size_class_name": size_class_name,
        "design_life": size_info.get("design_life") if size_info else None,
        "total_area_sqm": round(total_area, 1),
        "function_area_ranges": function_area_ranges,
        "exhibition_area_estimate": round(exhibition_area_estimate, 1) if exhibition_area_estimate is not None else None,
        "exhibit_count_estimate": exhibit_counts,
        "instantaneous_capacity": instantaneous_capacity,
        "warnings": warnings,
    }


def load_all_cases(
    data_dirs: Optional[List[str]] = None,
    base_path: Optional[Path] = None,
) -> List[BuildingCase]:
    """
    加载所有建筑案例数据。
    
    Args:
        data_dirs: 数据目录列表，默认为 ["archdaily", "china", "world"]
        base_path: 数据根目录，默认为 code/../data
        
    Returns:
        List[BuildingCase]: 解析成功的案例列表
    """
    if data_dirs is None:
        data_dirs = ["archdaily", "china", "world"]

    # 优先使用全局配置中的 DATA_ROOT，其次回退到相对路径
    if base_path is None:
        try:
            base_path = DATA_ROOT
        except Exception:
            base_path = Path(__file__).resolve().parents[2] / "data"
    
    cases: List[BuildingCase] = []
    
    for dir_name in data_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            logger.warning("数据目录不存在: %s", dir_path)
            continue
        
        for json_file in dir_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("读取文件失败 %s: %s", json_file, e)
                continue
            
            area = _extract_area(data)
            if area is None:
                continue  # 跳过无有效面积的案例
            height = _extract_height(data)
            
            case = BuildingCase(
                name=_extract_name(data, json_file),
                area=area,
                height=height,
                year=_extract_year(data),
                filepath=str(json_file),
                source=dir_name,
            )
            cases.append(case)
    
    logger.info("已加载 %d 个有效案例", len(cases))
    return cases


# =============================================================================
# 4. 动态分析主程序
# =============================================================================

def analyze_indicators(
    target_area: float,
    data_dirs: Optional[List[str]] = None,
    base_path: Optional[Path] = None,
    top_k_similar: int = 5,
    recent_year_threshold: int = 2018,
) -> Dict[str, Any]:
    """
    根据目标建筑面积执行经济技术指标分析。
    
    分析内容：
    1. 定级：根据《科学技术馆建设标准》判定目标馆等级
    2. 规范合规性推演：根据功能用房占比与核心技术指标计算建议面积区间、展品数量与瞬时承载量
    3. 同级统计：计算同等级案例的面积均值、最大值、最小值、近年趋势
    4. 相似案例匹配：寻找面积最接近的前 K 个案例
    
    Args:
        target_area: 目标建筑面积（平方米）
        data_dirs: 数据目录列表，默认为 ["archdaily", "china", "world"]
        base_path: 数据根目录
        top_k_similar: 返回相似案例数量
        recent_year_threshold: "近年"定义的起始年份
        
    Returns:
        dict: {
            "target_area": float,
            "target_classification": {
                "class_name": str,
                "design_life": int,
                "area_range": str,
            },
            "statistics": {
                "same_class_count": int,
                "mean_area": float,
                "max_area": float,
                "min_area": float,
                "recent_count": int,
                "recent_mean_area": float | None,
            },
            "compliance": {...},
            "similar_cases": [
                {
                    "name": str,
                    "area": float,
                    "year": int | None,
                    "source": str,
                    "filepath": str,
                    "area_diff": float,
                }
            ],
            "all_cases_count": int,
        }
    """
    # Step A: 加载数据
    all_cases = load_all_cases(data_dirs=data_dirs, base_path=base_path)
    
    # Step B: 执行分析

    # B1: 定级
    classification = get_size_class(target_area)
    target_class = classification["class_name"]

    # B2: 规范合规性推演
    compliance = calculate_compliance_specs(target_area, target_class)

    # B3: 同级统计
    same_class_cases = [c for c in all_cases if c.size_class == target_class]
    same_class_areas = [c.area for c in same_class_cases if c.area is not None]
    
    stats: Dict[str, Any] = {
        "same_class_count": len(same_class_cases),
        "mean_area": None,
        "max_area": None,
        "min_area": None,
        "recent_count": 0,
        "recent_mean_area": None,
    }
    
    if same_class_areas:
        stats["mean_area"] = round(sum(same_class_areas) / len(same_class_areas), 2)
        stats["max_area"] = max(same_class_areas)
        stats["min_area"] = min(same_class_areas)
    
    # 近年趋势分析
    recent_cases = [
        c for c in same_class_cases
        if c.year is not None and c.year >= recent_year_threshold
    ]
    recent_areas = [c.area for c in recent_cases if c.area is not None]
    stats["recent_count"] = len(recent_cases)
    if recent_areas:
        stats["recent_mean_area"] = round(sum(recent_areas) / len(recent_areas), 2)
    
    # B4: 相似案例匹配（全局）
    cases_with_diff = [
        (c, abs(c.area - target_area))
        for c in all_cases
        if c.area is not None
    ]
    cases_with_diff.sort(key=lambda x: x[1])
    
    similar_cases = []
    for case, diff in cases_with_diff[:top_k_similar]:
        similar_cases.append({
            "name": case.name,
            "area": case.area,
            "height": case.height,
            "year": case.year,
            "source": case.source,
            "filepath": case.filepath,
            "size_class": case.size_class,
            "area_diff": round(diff, 2),
        })
    
    # Step C: 返回结果
    return {
        "target_area": target_area,
        "target_classification": classification,
        "statistics": stats,
        "compliance": compliance,
        "industry_reference": INDUSTRY_BENCHMARKS,
        "similar_cases": similar_cases,
        "all_cases_count": len(all_cases),
    }


# =============================================================================
# 5. 便捷函数与命令行入口
# =============================================================================

def format_analysis_report(result: Dict[str, Any]) -> str:
    """以 Markdown 文本输出“规范 + 实证”双轨结论。"""

    lines: List[str] = []
    target_area = result.get("target_area", 0)
    classification = result.get("target_classification", {})
    compliance = result.get("compliance") or {}
    stats = result.get("statistics") or {}
    similar_cases = result.get("similar_cases") or []
    industry = result.get("industry_reference") or INDUSTRY_BENCHMARKS

    lines.append("=" * 70)
    lines.append("🔬 科技馆策划指标推演 (Evidence-Based)")
    lines.append("=" * 70)
    lines.append(f"目标建筑面积：{target_area:,.0f} m²")

    # Section 1
    lines.append("\n## 1. 定级与规范依据 (Track A · Normative)")
    standard_source = compliance.get("standard_source", DIGITAL_STANDARD["source"])
    lines.append(f"- 参考标准：{standard_source}")
    lines.append(f"- 判定等级：{classification.get('class_name', '未知')} (设计寿命 {classification.get('design_life', 'N/A')} 年)")
    lines.append(f"- 面积判定区间：{classification.get('area_range', '-')}")

    # Section 2
    lines.append("\n## 2. 功能配比推演")
    func_ranges = compliance.get("function_area_ranges") or {}
    if func_ranges:
        lines.append("| 功能区 | 占比范围 | 建议面积 (m²) |")
        lines.append("| --- | --- | --- |")
        name_map = {
            "exhibition_education": "展览教育用房",
            "science_education": "科研教学",
            "public_service": "公众服务用房",
            "business_research": "业务研究用房",
            "management": "管理保障用房",
        }
        order = [
            "exhibition_education",
            "science_education",
            "public_service",
            "business_research",
            "management",
        ]
        for key in order:
            fr = func_ranges.get(key)
            if not fr:
                continue
            pct = f"{fr['percent_min']:.1f}%–{fr['percent_max']:.1f}%"
            area = f"{fr['area_min_sqm']:,.0f}–{fr['area_max_sqm']:,.0f}"
            lines.append(f"| {name_map.get(key, key)} | {pct} | {area} |")
    else:
        lines.append("> 暂无功能配比数据，请检查目标等级配置。")

    # Section 3
    lines.append("\n## 3. 关键技术指标")
    ex_area = compliance.get("exhibition_area_estimate")
    if ex_area is not None:
        lines.append(f"- 展厅面积估算：{ex_area:,.0f} m²")
    exhibit_counts = compliance.get("exhibit_count_estimate")
    if exhibit_counts:
        lines.append(f"- 参考展品规模：{exhibit_counts['min']:,d} – {exhibit_counts['max']:,d} 件 (按 15–30 m²/件)")
    capacity = compliance.get("instantaneous_capacity")
    if capacity:
        lines.append(f"- 瞬时最大观众容量：{capacity['min']:,d} – {capacity['max']:,d} 人 (按 0.2–0.25 人/m²)")
    warnings = compliance.get("warnings") or []
    for warn in warnings:
        lines.append(f"  ⚠️ {warn}")

    # Section 4
    lines.append("\n## 4. 相似案例与参考 (Track B · Empirical)")
    lines.append("**同级案例统计**")
    lines.append(f"- 案例数量：{stats.get('same_class_count', 0)} 个")
    if stats.get("mean_area"):
        lines.append(f"- 平均面积：{stats['mean_area']:,.0f} m² (最大 {stats['max_area']:,.0f} / 最小 {stats['min_area']:,.0f})")
    if stats.get("recent_count"):
        lines.append(f"- 近年 (≥2018) 案例：{stats['recent_count']} 个，平均 {stats.get('recent_mean_area') or 0:,.0f} m²")

    if similar_cases:
        lines.append("\n**面积最接近的参考案例**")
        for idx, case in enumerate(similar_cases, 1):
            year_str = f"，{case['year']}" if case.get("year") else ""
            diff = case.get("area", 0) - target_area
            diff_str = f"{diff:+,.0f}"
            lines.append(f"{idx}. {case['name']} ({case['source']}{year_str})")
            detail = [
                f"面积 {case.get('area', 0):,.0f} m²",
                f"差异 {diff_str} m²",
                f"等级 {case.get('size_class') or '-'}",
            ]
            if case.get("height"):
                detail.append(f"高度 {case['height']:.1f} m")
            lines.append("   - " + " | ".join(detail))
    else:
        lines.append("\n> 案例库中暂无足够的相似项目。")

    lines.append("\n**行业经验值参考（建筑设计资料集）**")
    lines.append(
        f"- 容积率 (FAR)：{industry['floor_area_ratio'][0]} – {industry['floor_area_ratio'][1]}"
    )
    lines.append(
        f"- 建筑密度：{industry['building_density'][0]}% – {industry['building_density'][1]}%"
    )
    lines.append(
        f"- 绿地率：{industry['green_ratio'][0]}% – {industry['green_ratio'][1]}%"
    )

    lines.append("\n数据源：内部案例库 {} 项；标准来源：{}".format(
        result.get("all_cases_count", 0), standard_source
    ))
    lines.append("=" * 70)
    return "\n".join(lines)


def query_building_indicators(target_area: float) -> str:
    """Run building indicator analysis and return a formatted report.

    Useful for retrieving building cases and technical indicators based on area.

    This is the primary tool-style entrypoint for agents/graphs:
    - Takes a target gross floor area (m²)
    - Classifies the building size per national standards
    - Computes statistics over similar-scale reference projects
    - Returns a human-readable Markdown-like text report.
    """

    result = analyze_indicators(target_area)
    return format_analysis_report(result)


def query_building_indicators(target_area: float) -> str:
    """通过建筑面积查询国家标准等级及相似案例的工具。

    该函数作为对外暴露的 Agent / Graph 工具入口：

    - 输入目标建筑面积（单位：平方米）
    - 基于《科学技术馆建设标准》自动判定馆舍等级与设计使用年限
    - 结合内置案例库，统计同等级项目的面积分布与近年趋势
    - 返回人类可读的文本报告，便于 LLM 直接引用与总结

    Args:
        target_area: 目标建筑面积（平方米）。

    Returns:
        已格式化的经济技术指标分析报告文本。
    """

    result = analyze_indicators(target_area)
    return format_analysis_report(result)


def _configure_sys_path() -> None:
    """确保以模块方式运行时可以正确导入项目内模块。

    支持在项目根目录下执行：

        python -m code.utils.indicator_analyzer 35000
    """

    project_root = Path(__file__).resolve().parents[2]
    code_dir = project_root / "code"

    for p in (project_root, code_dir):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


def main() -> None:
    """命令行入口：在终端中快速运行指标分析。"""

    _configure_sys_path()
    setup_logging()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
        except ValueError:
            print(f"错误: 无法解析面积参数 '{sys.argv[1]}'")
            sys.exit(1)
    else:
        # 默认测试值
        target = 25000.0

    print(f"\n分析目标面积: {target:,.0f} m²\n")
    result = analyze_indicators(target)
    print(format_analysis_report(result))


if __name__ == "__main__":
    try:
        # 基本日志配置，配合 log_setup 做文件记录
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # 显示当前数据根目录，便于排查路径问题
        print(f"当前数据目录: {DATA_ROOT}")

        if len(sys.argv) > 1:
            try:
                target = float(sys.argv[1])
            except ValueError:
                print(f"错误: 无法解析面积参数 '{sys.argv[1]}'，请提供数字，例如 35000")
                sys.exit(1)
        else:
            # 默认测试面积
            target = 35000.0

        print(f"\n分析目标面积: {target:,.0f} m²\n")
        setup_logging()
        result = analyze_indicators(target)
        print(format_analysis_report(result))
    except Exception as exc:  # noqa: BLE001
        print(f"运行过程中发生错误: {exc}")
        sys.exit(1)
