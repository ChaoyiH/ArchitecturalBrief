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
# 0. 《科学技术馆建设标准》规范常量 (GB_SCIENCE_MUSEUM_STANDARD)
# =============================================================================

GB_SCIENCE_MUSEUM_STANDARD: Dict[str, Any] = {
    "classification_thresholds": {
        "extra_large": {"name": "特大型馆", "min_area": 30000.0, "max_area": None},
        "large": {"name": "大型馆", "min_area": 15000.0, "max_area": 30000.0},
        "medium": {"name": "中型馆", "min_area": 8000.0, "max_area": 15000.0},
        "small": {"name": "小型馆", "min_area": 0.0, "max_area": 8000.0},
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
    "technical_indicators": {
        # 单位：m²/件
        "exhibit_density_sqm_per_piece": (15.0, 30.0),
        # 单位：人/m²（基于展厅面积）
        "instantaneous_occupancy_person_per_sqm": (0.2, 0.25),
        # 常设展厅最小面积 (m²)
        "permanent_exhibition_min_area": 3000.0,
        # 建议的小型馆下限（用于提示）
        "recommended_min_building_area": 5000.0,
    },
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
    
    thresholds = GB_SCIENCE_MUSEUM_STANDARD["classification_thresholds"]

    if area_sqm > thresholds["extra_large"]["min_area"]:
        return {
            "class_name": thresholds["extra_large"]["name"],
            "design_life": 100,
            "area_range": "> 30,000 m²",
        }

    if thresholds["large"]["min_area"] < area_sqm <= thresholds["large"]["max_area"]:
        return {
            "class_name": thresholds["large"]["name"],
            "design_life": 100,
            "area_range": "15,000 ~ 30,000 m²",
        }

    if thresholds["medium"]["min_area"] < area_sqm <= thresholds["medium"]["max_area"]:
        return {
            "class_name": thresholds["medium"]["name"],
            "design_life": 50,
            "area_range": "8,000 ~ 15,000 m²",
        }

    return {
        "class_name": thresholds["small"]["name"],
        "design_life": 50,
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


def calculate_compliance(target_area: float, size_class_name: str) -> Dict[str, Any]:
    """根据《科学技术馆建设标准》对目标面积进行合规性推演。

    输出内容包括：
    - 各功能用房面积区间（m²）及对应百分比
    - 估算展厅面积
    - 理论展品数量区间
    - 瞬时最大观众容量区间
    - 关键提醒（如建筑面积过小等）
    """

    tech = GB_SCIENCE_MUSEUM_STANDARD["technical_indicators"]
    ratios_all = GB_SCIENCE_MUSEUM_STANDARD["function_ratios"]

    func_ratios = ratios_all.get(size_class_name)
    if func_ratios is None:
        return {
            "size_class_name": size_class_name,
            "function_area_ranges": {},
            "exhibition_area_estimate": None,
            "exhibit_count_range": None,
            "instantaneous_occupancy_range": None,
            "warnings": [f"未找到等级 '{size_class_name}' 的功能占比配置"],
        }

    total_area = max(0.0, float(target_area))

    function_area_ranges: Dict[str, Dict[str, float]] = {}
    for key, (pct_min, pct_max) in func_ratios.items():
        pct_min, pct_max = _clamp_percentage_range((pct_min, pct_max))
        area_min = total_area * pct_min / 100.0
        area_max = total_area * pct_max / 100.0
        function_area_ranges[key] = {
            "percent_min": round(pct_min, 1),
            "percent_max": round(pct_max, 1),
            "area_min_sqm": round(area_min, 1),
            "area_max_sqm": round(area_max, 1),
        }

    # 展览教育用房面积区间
    exhibit_ratio = func_ratios["exhibition_education"]
    ex_pct_min, ex_pct_max = _clamp_percentage_range(exhibit_ratio)
    ex_area_min = total_area * ex_pct_min / 100.0
    ex_area_max = total_area * ex_pct_max / 100.0
    # 采用平均值作为展厅面积的估算值
    exhibition_area_estimate = (ex_area_min + ex_area_max) / 2.0 if total_area > 0 else 0.0

    # 展品密度：m²/件 → 件数 = 面积 / 单位面积
    density_min, density_max = tech["exhibit_density_sqm_per_piece"]
    exhibit_count_min = exhibition_area_estimate / density_max if density_max > 0 else 0.0
    exhibit_count_max = exhibition_area_estimate / density_min if density_min > 0 else 0.0

    # 瞬时最大承载量：人/m² × 展厅面积
    occ_min, occ_max = tech["instantaneous_occupancy_person_per_sqm"]
    instantaneous_min = exhibition_area_estimate * occ_min
    instantaneous_max = exhibition_area_estimate * occ_max

    warnings: List[str] = []
    if total_area < tech["recommended_min_building_area"]:
        warnings.append("建筑面积不宜小于 5000 m² (小型科技馆合理下限)")
    if exhibition_area_estimate < tech["permanent_exhibition_min_area"]:
        warnings.append("估算展厅面积小于 3000 m²，可能难以满足常设展厅最小面积要求")

    return {
        "size_class_name": size_class_name,
        "total_area_sqm": round(total_area, 1),
        "function_area_ranges": function_area_ranges,
        "exhibition_area_estimate": round(exhibition_area_estimate, 1),
        "exhibit_count_range": (
            int(exhibit_count_min),
            int(exhibit_count_max),
        ),
        "instantaneous_occupancy_range": (
            int(instantaneous_min),
            int(instantaneous_max),
        ),
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
            "compliance_analysis": {...},
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
    compliance = calculate_compliance(target_area, target_class)

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
        "compliance_analysis": compliance,
        "similar_cases": similar_cases,
        "all_cases_count": len(all_cases),
    }


# =============================================================================
# 5. 便捷函数与命令行入口
# =============================================================================

def format_analysis_report(result: Dict[str, Any]) -> str:
    """将分析结果格式化为可读的文本报告"""
    lines = []
    
    # 标题
    lines.append("=" * 60)
    lines.append("📊 经济技术指标分析报告")
    lines.append("=" * 60)
    
    # 目标信息
    lines.append(f"\n📐 目标建筑面积: {result['target_area']:,.0f} m²")
    
    # 分级结果
    cls = result["target_classification"]
    lines.append(f"\n🏛️ 等级判定 (依据《科学技术馆建设标准》):")
    lines.append(f"   • 等级: {cls['class_name']}")
    lines.append(f"   • 面积范围: {cls['area_range']}")
    lines.append(f"   • 建议设计使用年限: {cls['design_life']} 年")
    
    # 合规性推演
    comp = result.get("compliance_analysis") or {}
    lines.append(f"\n📋 国家标准合规性推演 (依据：建标 101-2007):")
    lines.append(
        f"   • 判定等级: {cls['class_name']} | 面积范围: {cls['area_range']}"
    )

    func_ranges = comp.get("function_area_ranges") or {}
    if func_ranges:
        lines.append("   • 功能用房面积分配建议：")
        name_map = {
            "exhibition_education": "展览教育用房",
            "public_service": "公共服务用房",
            "business_research": "业务研究用房",
            "management": "管理保障用房",
        }
        for key in ("exhibition_education", "public_service", "business_research", "management"):
            if key not in func_ranges:
                continue
            fr = func_ranges[key]
            label = name_map.get(key, key)
            pct_min = fr.get("percent_min")
            pct_max = fr.get("percent_max")
            a_min = fr.get("area_min_sqm")
            a_max = fr.get("area_max_sqm")
            lines.append(
                f"      - {label}: {pct_min:.1f}–{pct_max:.1f}% 约 {a_min:,.0f}–{a_max:,.0f} m²"
            )

    ex_area = comp.get("exhibition_area_estimate")
    exhibit_range = comp.get("exhibit_count_range") or (None, None)
    occ_range = comp.get("instantaneous_occupancy_range") or (None, None)
    if ex_area is not None:
        lines.append(
            f"   • 展厅面积估算: {ex_area:,.0f} m² (用于下述指标推演)"
        )
    if all(v is not None for v in exhibit_range):
        lines.append(
            f"   • 理论展品数量: 约 {exhibit_range[0]:,d}–{exhibit_range[1]:,d} 件"
        )
    if all(v is not None for v in occ_range):
        lines.append(
            f"   • 瞬时最大观众容量: 约 {occ_range[0]:,d}–{occ_range[1]:,d} 人"
        )

    warnings = comp.get("warnings") or []
    for w in warnings:
        lines.append(f"   ⚠️ {w}")

    # 同级统计
    stats = result["statistics"]
    lines.append(f"\n📈 同级案例统计 ({cls['class_name']}):")
    lines.append(f"   • 案例数量: {stats['same_class_count']} 个")
    if stats["mean_area"]:
        lines.append(f"   • 平均面积: {stats['mean_area']:,.0f} m²")
        lines.append(f"   • 最大面积: {stats['max_area']:,.0f} m²")
        lines.append(f"   • 最小面积: {stats['min_area']:,.0f} m²")
    if stats["recent_count"] > 0:
        lines.append(f"   • 近年 (2018+) 案例: {stats['recent_count']} 个")
        if stats["recent_mean_area"]:
            lines.append(f"   • 近年平均面积: {stats['recent_mean_area']:,.0f} m²")
    
    # 相似案例
    lines.append(f"\n🔍 最相似案例 (面积接近):")
    for i, case in enumerate(result["similar_cases"], 1):
        year_str = f", {case['year']}" if case["year"] else ""
        lines.append(
            f"   {i}. {case['name']} ({case['source']})"
        )
        lines.append(
            f"      面积: {case['area']:,.0f} m² | "
            f"差异: {'+' if case['area'] > result['target_area'] else ''}"
            f"{case['area'] - result['target_area']:,.0f} m² | "
            f"等级: {case['size_class']}"
            f"{year_str}"
        )
        if case.get("height") is not None:
            lines.append(
                f"      📐 建筑高度: {case['height']:.1f} m"
            )
    
    lines.append("\n" + "=" * 60)
    lines.append(f"数据来源: {result['all_cases_count']} 个建筑案例")
    lines.append("=" * 60)
    
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
