#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化中国科技馆地理分布 (高保真版)
Visualize geographic distribution of science & technology museums in China.
Uses province-level GeoJSON for a high-fidelity base map.
"""

import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path

# ============================================================
# 1. 科技馆 -> 所在地级市 映射
# ============================================================
MUSEUM_CITY_MAP = {
    "上海科技馆": "上海",
    "东莞市科学技术博物馆": "东莞",
    "中国地质博物馆": "北京",
    "中国科学技术馆": "北京",
    "中国航海博物馆": "上海",
    "中国铁道博物馆": "北京",
    "云南省科学技术馆": "昆明",
    "内蒙古科学技术馆": "呼和浩特",
    "北京科学中心": "北京",
    "南京科技馆": "南京",
    "厦门科技馆": "厦门",
    "合肥市科技馆": "合肥",
    "吉林省科学技术馆": "长春",
    "四川科技馆": "成都",
    "天津科学技术馆": "天津",
    "宁夏科技馆": "银川",
    "宁波科学探索中心": "宁波",
    "安徽省科学技术馆": "合肥",
    "山东省科学技术馆": "济南",
    "山西省科学技术馆": "太原",
    "广东科学中心": "广州",
    "广西科技馆": "南宁",
    "扬州科技馆": "扬州",
    "新疆科学技术馆": "乌鲁木齐",
    "杭州低碳科技馆": "杭州",
    "柳州科技馆": "柳州",
    "武汉科学技术馆": "武汉",
    "江苏省科学技术馆": "南京",
    "江西省科学技术馆": "南昌",
    "河北省科学技术馆": "石家庄",
    "河南省科学技术馆": "郑州",
    "泉州市科技馆": "泉州",
    "浙江省科技馆": "杭州",
    "海南省科学技术馆": "海口",
    "深圳市科学馆": "深圳",
    "深圳红立方(龙岗区科技馆)": "深圳",
    "湖北省科学技术馆": "武汉",
    "湖南省科学技术馆": "长沙",
    "甘肃科技馆": "兰州",
    "福建省科学技术馆": "福州",
    "绍兴科技馆": "绍兴",
    "绵阳科技馆": "绵阳",
    "自贡恐龙博物馆": "自贡",
    "苏州科技馆": "苏州",
    "贵州科技馆": "贵阳",
    "辽宁省科学技术馆": "沈阳",
    "重庆科技馆": "重庆",
    "陕西省科学技术馆": "西安",
    "青岛市科技馆": "青岛",
    "青海省科学技术馆": "西宁",
    "黑龙江省科学技术馆": "哈尔滨",
    # JSON name 字段与文件名不一致的别名
    "中国杭州低碳科技馆": "杭州",
    "深圳·红立方（龙岗区科技馆）": "深圳",
    "苏州科技馆（含工业展览馆）": "苏州",
    "陕西科学技术馆": "西安",
    "青岛科技馆": "青岛",
}

# ============================================================
# 2. 地级市经纬度 (WGS84)
# ============================================================
CITY_COORDS = {
    "北京": (116.407, 39.904), "上海": (121.473, 31.230),
    "天津": (117.190, 39.125), "重庆": (106.551, 29.563),
    "广州": (113.264, 23.129), "深圳": (114.058, 22.543),
    "东莞": (113.746, 23.043), "南京": (118.797, 32.060),
    "苏州": (120.585, 31.299), "扬州": (119.413, 32.394),
    "杭州": (120.153, 30.287), "宁波": (121.550, 29.868),
    "绍兴": (120.580, 30.030), "合肥": (117.227, 31.820),
    "福州": (119.296, 26.074), "厦门": (118.089, 24.479),
    "泉州": (118.675, 24.874), "南昌": (115.858, 28.682),
    "济南": (117.000, 36.675), "青岛": (120.383, 36.067),
    "武汉": (114.305, 30.593), "长沙": (112.938, 28.228),
    "成都": (104.066, 30.572), "绵阳": (104.679, 31.467),
    "自贡": (104.773, 29.352), "昆明": (102.832, 25.040),
    "贵阳": (106.630, 26.647), "西安": (108.940, 34.341),
    "兰州": (103.834, 36.061), "银川": (106.278, 38.487),
    "西宁": (101.778, 36.617), "乌鲁木齐": (87.617, 43.793),
    "呼和浩特": (111.751, 40.842), "太原": (112.549, 37.870),
    "石家庄": (114.515, 38.042), "郑州": (113.665, 34.757),
    "沈阳": (123.432, 41.805), "长春": (125.324, 43.886),
    "哈尔滨": (126.534, 45.803), "南宁": (108.320, 22.824),
    "柳州": (109.412, 24.314), "海口": (110.350, 20.017),
}


# ============================================================
# 3. GeoJSON 渲染
# ============================================================

def load_geojson(filepath):
    """加载 GeoJSON 文件。"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def geojson_to_patches(geojson_data, facecolor='#F0F0F0', edgecolor='#C0C0C0',
                       linewidth=0.4):
    """将 GeoJSON FeatureCollection 转换为 matplotlib patches。"""
    patches = []
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        geom_type = geom.get('type', '')
        if geom_type == 'Polygon':
            for ring in geom['coordinates']:
                coords = np.array(ring)
                if len(coords) > 2:
                    poly = MplPolygon(coords, closed=True)
                    patches.append(poly)
        elif geom_type == 'MultiPolygon':
            for polygon in geom['coordinates']:
                for ring in polygon:
                    coords = np.array(ring)
                    if len(coords) > 2:
                        poly = MplPolygon(coords, closed=True)
                        patches.append(poly)
    return patches, facecolor, edgecolor, linewidth


def draw_china_geojson(ax, geojson_path):
    """使用 GeoJSON 绘制高保真中国地图。"""
    geojson_data = load_geojson(geojson_path)

    # 绘制省级行政区
    patches, fc, ec, lw = geojson_to_patches(
        geojson_data,
        facecolor='#EAEEF2',
        edgecolor='#B8C4D0',
        linewidth=0.35
    )
    collection = PatchCollection(
        patches, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2
    )
    ax.add_collection(collection)

    # 只保留九段线示意，移除缩略省级轮廓

    # 九段线和南海诸岛已经包含在底图中，无需重复绘制。


def load_museum_data(data_dir):
    """从 JSON 文件加载科技馆数据。"""
    museums = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name", Path(fpath).stem)
        area_str = data.get("total_construction_area_sqm", "0")
        try:
            area = float(area_str) if area_str else 0
        except ValueError:
            area = 0
        city = MUSEUM_CITY_MAP.get(name, None)
        if city and city in CITY_COORDS:
            lon, lat = CITY_COORDS[city]
            museums.append({
                "name": name, "area": area, "city": city,
                "lon": lon, "lat": lat,
            })
        else:
            print(f"⚠️  未找到城市坐标: {name} -> {city}")
    return museums


def setup_chinese_font():
    """配置中文字体。"""
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for fp in font_candidates:
        if os.path.isfile(fp):
            chinese_font = FontProperties(fname=fp)
            from matplotlib.font_manager import fontManager
            fontManager.addfont(fp)
            font_name = chinese_font.get_name()
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            print(f"🔤 使用字体: {font_name} ({fp})")
            return chinese_font
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti TC', 'SimHei', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False
    return None


def plot_museums(museums, output_path, geojson_path):
    """在高保真中国底图上绘制科技馆散点图。"""

    chinese_font = setup_chinese_font()
    fp = {"fontproperties": chinese_font} if chinese_font else {}

    # ---------- 面积分级 ----------
    areas = [m["area"] for m in museums]
    max_area = max(areas) if areas else 1

    def size_for_area(area):
        if area <= 0:
            return 50
        return 50 + 300 * (area / max_area)

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(1, 1, figsize=(14, 11), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # 高保真底图
    draw_china_geojson(ax, geojson_path)

    # 颜色分级
    colors, sizes = [], []
    for m in museums:
        sizes.append(size_for_area(m["area"]))
        if m["area"] >= 50000:
            colors.append('#D94F4F')   # 大型
        elif m["area"] >= 20000:
            colors.append('#E8983E')   # 中型
        else:
            colors.append('#4A90D9')   # 小型

    lons = [m["lon"] for m in museums]
    lats = [m["lat"] for m in museums]

    # 散点
    ax.scatter(
        lons, lats,
        s=sizes, c=colors, alpha=0.8,
        edgecolors='white', linewidths=0.8,
        zorder=10
    )

    # ---------- 标注名称 ----------
    placed = {}
    for m in museums:
        city = m["city"]
        idx = placed.get(city, 0)
        placed[city] = idx + 1

        # 偏移表
        offsets = {
            "上海":   (1.0, -1.0 - idx * 0.8),
            "北京":   (0.8, 0.5 + idx * 0.8),
            "南京":   (-3.5, 0.5 + idx * 0.8),
            "合肥":   (-3.5, -0.5 - idx * 0.8),
            "杭州":   (-3.5, -0.5 - idx * 0.8),
            "深圳":   (1.5, -0.8 - idx * 0.8),
            "广州":   (-3.5, 0.5),
            "武汉":   (-3.5, 0.3 + idx * 0.8),
            "苏州":   (1.5, 0.5 + idx * 0.8),
            "扬州":   (1.5, 0.5),
            "绍兴":   (1.5, 0.8),
            "宁波":   (1.5, -0.3),
            "福州":   (1.5, 0.5),
            "厦门":   (1.5, -0.5),
            "泉州":   (1.5, 0.3),
            "长沙":   (-3.0, -0.5),
            "南昌":   (0.8, -0.8),
            "郑州":   (-3.5, -0.3),
            "石家庄": (-3.5, 0.3),
            "太原":   (-3.0, -0.5),
            "济南":   (-3.5, 0.5),
            "青岛":   (1.0, 0.5),
            "东莞":   (-3.5, -0.3),
            "成都":   (-3.0, 0.5),
            "绵阳":   (0.8, 0.5),
            "自贡":   (0.8, -0.5),
            "天津":   (1.0, 0.0),
            "沈阳":   (1.0, -0.5),
            "长春":   (1.0, -0.5),
            "哈尔滨": (0.8, -0.5),
            "重庆":   (-3.0, -0.3),
            "贵阳":   (-2.5, -0.5),
            "昆明":   (-3.0, 0.0),
            "海口":   (0.8, -0.5),
            "柳州":   (0.8, 0.5),
            "南宁":   (-3.0, 0.3),
        }
        dx, dy = offsets.get(city, (0.5, 0.5 + idx * 0.5))

        ax.annotate(
            m["name"],
            xy=(m["lon"], m["lat"]),
            xytext=(m["lon"] + dx, m["lat"] + dy),
            fontsize=5, color='#333333',
            arrowprops=dict(arrowstyle='-', color='#AAAAAA', linewidth=0.3),
            zorder=12, **fp,
        )

    # ---------- 图例 ----------
    legend_elements = [
        mpatches.Patch(facecolor='#D94F4F', edgecolor='white', label='大型 (≥50,000 m²)'),
        mpatches.Patch(facecolor='#E8983E', edgecolor='white', label='中型 (20,000–50,000 m²)'),
        mpatches.Patch(facecolor='#4A90D9', edgecolor='white', label='小型 (<20,000 m²)'),
    ]
    legend = ax.legend(
        handles=legend_elements, loc='lower left',
        fontsize=8, framealpha=0.95, edgecolor='#DDDDDD',
        prop=chinese_font if chinese_font else None,
        title="建筑面积分级",
        title_fontproperties=chinese_font if chinese_font else None,
        borderpad=0.8,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0)

    # ---------- 隐藏坐标轴 ----------
    ax.set_xlim(72, 136)
    ax.set_ylim(2, 55)
    ax.set_aspect(1.2)
    ax.axis('off')

    # 数据说明
    ax.text(
        0.98, 0.02,
        "注：位置以各馆所在地级行政区中心坐标标示",
        transform=ax.transAxes, fontsize=5.5, color='#AAAAAA',
        ha='right', va='bottom', **fp,
    )

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 地图已保存至: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    data_dir = os.path.join(project_root, "data", "china")
    output_dir = os.path.join(project_root, "code", "output")
    geojson_path = os.path.join(script_dir, "geodata", "china_provinces.json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "china_museums_map.png")

    print(f"📂 数据目录: {data_dir}")
    print(f"🗺  GeoJSON: {geojson_path}")

    museums = load_museum_data(data_dir)
    print(f"📊 共加载 {len(museums)} 座科技馆数据")

    cities = set(m["city"] for m in museums)
    print(f"📍 分布在 {len(cities)} 个城市")

    from collections import Counter
    city_count = Counter(m["city"] for m in museums)
    multi = {c: n for c, n in city_count.items() if n > 1}
    if multi:
        print(f"🏛  同城多馆: {multi}")

    plot_museums(museums, output_path, geojson_path)


if __name__ == "__main__":
    main()
