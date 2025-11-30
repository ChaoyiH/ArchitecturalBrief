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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


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


# =============================================================================
# 2. 内置分级标准 (Hardcoded Standard)
# =============================================================================

# 《科学技术馆建设标准》分级表
# | 种类     | 规模与面积           | 设计使用年限 |
# |----------|---------------------|-------------|
# | 特大型馆 | 大于 40,000 m²       | 宜 100 年    |
# | 大型馆   | 20,000 ~ 40,000 m²   | 宜 100 年    |
# | 中型馆   | 8,000 ~ 20,000 m²    | 50 年        |
# | 小型馆   | 小于 8,000 m²        | 50 年        |

SIZE_CLASS_THRESHOLDS = [
    # (下限, 上限, 等级名称, 设计年限)
    # 注意：上限使用 None 表示无穷大
    (40000, None, "特大型馆", 100),
    (20000, 40000, "大型馆", 100),
    (8000, 20000, "中型馆", 50),
    (0, 8000, "小型馆", 50),
]


def get_size_class(area_sqm: float) -> Dict[str, Any]:
    """
    根据面积判定科技馆等级（依据《科学技术馆建设标准》）。
    
    分级标准（硬编码）：
    - 特大型馆：面积 > 40,000 m² (设计使用年限: 100年)
    - 大型馆：20,000 m² ≤ 面积 ≤ 40,000 m² (设计使用年限: 100年)
    - 中型馆：8,000 m² ≤ 面积 < 20,000 m² (设计使用年限: 50年)
    - 小型馆：面积 < 8,000 m² (设计使用年限: 50年)
    
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
    
    # 特大型馆：> 40,000 m²
    if area_sqm > 40000:
        return {
            "class_name": "特大型馆",
            "design_life": 100,
            "area_range": "> 40,000 m²",
        }
    
    # 大型馆：20,000 ~ 40,000 m²
    if area_sqm >= 20000:
        return {
            "class_name": "大型馆",
            "design_life": 100,
            "area_range": "20,000 ~ 40,000 m²",
        }
    
    # 中型馆：8,000 ~ 20,000 m²
    if area_sqm >= 8000:
        return {
            "class_name": "中型馆",
            "design_life": 50,
            "area_range": "8,000 ~ 20,000 m²",
        }
    
    # 小型馆：< 8,000 m²
    return {
        "class_name": "小型馆",
        "design_life": 50,
        "area_range": "< 8,000 m²",
    }


# =============================================================================
# 3. 数据加载与案例结构
# =============================================================================

@dataclass
class BuildingCase:
    """建筑案例数据结构"""
    name: str
    area: Optional[float]  # 平方米
    year: Optional[int]
    filepath: str
    source: str  # "archdaily", "china", "world"
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


def _extract_name(data: Dict[str, Any], filepath: Path) -> str:
    """从 JSON 数据中提取项目名称"""
    name_fields = ["Project Title", "name", "名称", "项目名称"]
    
    for key in name_fields:
        if key in data and data[key]:
            return str(data[key])
    
    # 使用文件名作为备选
    return filepath.stem


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
    
    if base_path is None:
        # 默认路径：code/utils/indicator_analyzer.py -> code/../data
        base_path = Path(__file__).parent.parent.parent / "data"
    
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
            
            case = BuildingCase(
                name=_extract_name(data, json_file),
                area=area,
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
    2. 同级统计：计算同等级案例的面积均值、最大值、最小值、近年趋势
    3. 相似案例匹配：寻找面积最接近的前 K 个案例
    
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
    
    # B2: 同级统计
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
    
    # B3: 相似案例匹配（全局）
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
    
    lines.append("\n" + "=" * 60)
    lines.append(f"数据来源: {result['all_cases_count']} 个建筑案例")
    lines.append("=" * 60)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
        except ValueError:
            print(f"错误: 无法解析面积参数 '{sys.argv[1]}'")
            sys.exit(1)
    else:
        # 默认测试值
        target = 25000
    
    print(f"\n分析目标面积: {target:,.0f} m²\n")
    
    # 执行分析
    result = analyze_indicators(target)
    
    # 输出报告
    print(format_analysis_report(result))
