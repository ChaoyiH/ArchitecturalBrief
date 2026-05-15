#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化世界科技馆地理分布
Visualize geographic distribution of science & technology museums worldwide.
Reads from both data/world and data/archdaily directories.
"""

import json
import os
import glob
import re
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
# 1. 世界城市经纬度坐标 (WGS84)
# ============================================================
CITY_COORDS = {
    # World data museums
    "New York":       (-74.006, 40.713),
    "Geneva":         (6.143, 46.204),
    "Los Angeles":    (-118.244, 34.052),
    "Sao Paulo":      (-46.636, -23.548),
    "Paris":          (2.352, 48.857),
    "Warsaw":         (21.012, 52.230),
    "Barcelona":      (2.177, 41.389),
    "Munich":         (11.576, 48.137),
    "Copenhagen":     (12.568, 55.676),
    "San Francisco":  (-122.419, 37.775),
    "Glasgow":        (-4.251, 55.864),
    "Chicago":        (-87.630, 41.878),
    "Gwacheon":       (126.988, 37.428),
    "Vantaa":         (25.039, 60.293),
    "Houston":        (-95.370, 29.760),
    "Merritt Island": (-80.681, 28.395),
    "Jersey City":    (-74.044, 40.721),
    "Haifa":          (34.989, 32.794),
    "Tokyo":          (139.692, 35.690),
    "Montreal":       (-73.568, 45.502),
    "Santiago":       (-70.669, -33.449),
    "Milan":          (9.190, 45.464),
    "Valencia":       (-0.376, 39.470),
    "Boston":         (-71.058, 42.360),
    "Dubai":          (55.296, 25.277),
    "Amsterdam":      (4.900, 52.370),
    "Daejeon":        (127.385, 36.351),
    "Pathum Thani":   (100.534, 14.020),
    "Leicester":      (-1.133, 52.637),
    "Portland":       (-122.676, 45.523),
    "Toronto":        (-79.383, 43.653),
    "Seattle":        (-122.332, 47.606),
    "Kuala Lumpur":   (101.687, 3.139),
    "Wolfsburg":      (10.787, 52.423),
    "Canberra":       (149.129, -35.281),
    "Singapore":      (103.820, 1.352),
    "Kolkata":        (88.364, 22.573),
    "London":         (-0.128, 51.508),
    "Vancouver":      (-123.121, 49.283),
    "Melbourne":      (144.963, -37.814),
    "Washington D.C.":(-77.037, 38.907),
    "Lucerne":        (8.310, 47.050),
    "Sinsheim":       (8.879, 49.253),
    "Stockholm":      (18.069, 59.329),
    "Philadelphia":   (-75.164, 39.953),
    "Dearborn":       (-83.177, 42.322),
    "Bremen":         (8.801, 53.080),

    # Archdaily data museums
    "Douai":              (3.080, 50.372),
    "Mexico":             (-99.133, 19.432),
    "Ciudad de México":   (-99.133, 19.432),
    "Shangrao":           (117.943, 28.455),
    "Shen Zhen Shi":      (114.058, 22.543),
    "Shenzhen":           (114.058, 22.543),
    "Tulln an der Donau": (15.883, 48.333),
    "Seoul":              (126.978, 37.567),
    "Panzhihua":          (101.718, 26.582),
    "Zhengzhou":          (113.665, 34.757),
    "Hemet":              (-116.972, 33.748),
    "Monterrey":          (-100.316, 25.687),
    "United States":      (-98.580, 39.833),   # center US (fallback)
    "Linz":               (14.290, 48.307),
    "Modena":             (10.925, 44.647),
    "Dallas":             (-96.797, 32.777),
    "Taiwan":             (121.000, 23.700),
    "Immenstaad am Bodensee": (9.371, 47.665),
    "Trento":             (11.122, 46.069),
    "Porsgrunn":          (9.655, 59.138),
    "Beijing":            (116.407, 39.904),
    "Panama":             (-79.534, 8.952),
    "Aarhus":             (10.204, 56.157),
    "Wembley":            (-119.139, 55.758),   # Wembley, Alberta
    "Nanjing":            (118.797, 32.060),
    "Sofia":              (23.322, 42.697),
    "Kerkrade":           (6.065, 50.865),
    "Rio de Janeiro":     (-43.173, -22.907),
    "Lisboa":             (-9.140, 38.737),
    "Stuttgart":          (9.181, 48.776),
    "Tel Aviv-Yafo":      (34.782, 32.084),
    "Miami":              (-80.192, 25.762),
    "Ankara":             (32.854, 39.920),
    "Binhai":             (117.700, 39.000),
    "Huzhou":             (120.088, 30.894),
    "Xiamen":             (118.089, 24.479),
    "Yibin":              (104.644, 28.752),
    "Shanghai":           (121.473, 31.230),
    "Luzern":             (8.310, 47.050),
    "Gangseo-gu":         (126.851, 37.551),
    "Montréal":           (-73.568, 45.502),
    "Xinyang":            (114.075, 32.124),
    "Sausalito":          (-122.485, 37.859),
}


def setup_chinese_font():
    """配置中文字体。"""
    font_candidates = [
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for fp_path in font_candidates:
        if os.path.isfile(fp_path):
            chinese_font = FontProperties(fname=fp_path)
            from matplotlib.font_manager import fontManager
            fontManager.addfont(fp_path)
            font_name = chinese_font.get_name()
            plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams.get('font.sans-serif', [])
            plt.rcParams['axes.unicode_minus'] = False
            return chinese_font
    return None


def parse_area(area_str):
    """解析面积字符串为 float (m²)。"""
    if not area_str or area_str == 'N/A':
        return 0
    area_str = str(area_str).strip()
    # Handle "sq ft" conversion
    if 'sq ft' in area_str.lower() or 'sqft' in area_str.lower():
        nums = re.findall(r'[\d.]+', area_str.replace(',', ''))
        if nums:
            return float(nums[0]) * 0.0929  # sq ft -> m²
    # Handle various "m²" / "sqm" formats
    nums = re.findall(r'[\d.]+', area_str.replace(',', ''))
    if nums:
        return float(nums[0])
    return 0


def load_world_museums(data_dir):
    """加载 data/world 目录的博物馆数据。"""
    museums = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("name", Path(fpath).stem)
        area = parse_area(data.get("total_construction_area", "0"))

        # 城市名: "Museum Name, City"
        parts = name.rsplit(", ", 1)
        if len(parts) == 2:
            city = parts[1].strip()
        else:
            city = Path(fpath).stem.rsplit(", ", 1)[-1].strip()

        if city in CITY_COORDS:
            lon, lat = CITY_COORDS[city]
            museums.append({
                "name": name, "area": area, "city": city,
                "lon": lon, "lat": lat, "source": "world",
            })
        else:
            print(f"⚠️  [world] 未找到城市坐标: {name} -> {city}")
    return museums


def load_archdaily_museums(data_dir):
    """加载 data/archdaily 目录的博物馆数据。"""
    museums = []
    for fpath in sorted(glob.glob(os.path.join(data_dir, "*.json"))):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("Project Title", Path(fpath).stem)
        area = parse_area(data.get("Area", "0"))
        city = data.get("City", "N/A")
        country = data.get("Country", "N/A")

        if city == "N/A" or city is None:
            print(f"⚠️  [archdaily] 无城市信息: {name}")
            continue

        if city in CITY_COORDS:
            lon, lat = CITY_COORDS[city]
            museums.append({
                "name": name, "area": area, "city": city,
                "country": country,
                "lon": lon, "lat": lat, "source": "archdaily",
            })
        else:
            print(f"⚠️  [archdaily] 未找到城市坐标: {name} -> {city}, {country}")
    return museums


def draw_world_map(ax, geojson_path):
    """使用 GeoJSON 绘制世界地图底图。"""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)

    patches = []
    for feature in geojson_data.get('features', []):
        geom = feature.get('geometry', {})
        if not geom:
            continue
        geom_type = geom.get('type', '')
        
        def add_polygon(ring):
            coords = np.array(ring)
            if len(coords) > 2:
                # Add a check to avoid polygons crossing the antimeridian (e.g. Russia, Antarctica, Fiji)
                if np.max(coords[:, 0]) - np.min(coords[:, 0]) > 300:
                    diffs = np.abs(np.diff(coords[:, 0]))
                    if np.any(diffs > 180):
                        return
                patches.append(MplPolygon(coords, closed=True))

        if geom_type == 'Polygon':
            for ring in geom['coordinates']:
                add_polygon(ring)
        elif geom_type == 'MultiPolygon':
            for polygon in geom['coordinates']:
                for ring in polygon:
                    add_polygon(ring)

    collection = PatchCollection(
        patches,
        facecolor='#EAEEF2',
        edgecolor='#C8CED6',
        linewidth=0.25,
        zorder=2
    )
    ax.add_collection(collection)


def plot_world_museums(museums, output_path, geojson_path):
    """在世界底图上绘制科技馆散点图。"""

    chinese_font = setup_chinese_font()
    fp = {"fontproperties": chinese_font} if chinese_font else {}

    # 面积分级
    areas = [m["area"] for m in museums if m["area"] > 0]
    max_area = max(areas) if areas else 1

    def size_for_area(area):
        if area <= 0:
            return 25
        return 25 + 160 * (area / max_area)

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(22, 11), dpi=200)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    draw_world_map(ax, geojson_path)

    # 按来源分色
    colors, sizes = [], []
    for m in museums:
        sizes.append(size_for_area(m["area"]))
        if m["source"] == "world":
            colors.append('#D94F4F')
        else:
            colors.append('#4A90D9')

    lons = [m["lon"] for m in museums]
    lats = [m["lat"] for m in museums]

    ax.scatter(
        lons, lats,
        s=sizes, c=colors, alpha=0.8,
        edgecolors='white', linewidths=0.5,
        zorder=10
    )

    # ---------- 智能标注 ----------
    # 按面积降序排列，优先标注大馆
    sorted_museums = sorted(museums, key=lambda m: m["area"], reverse=True)

    # 空间去重：已标注的文本位置
    label_boxes = []  # [(x, y, width_est, height_est)]
    MIN_DIST = 4.0    # 最小标注间距（经度单位）

    def can_place(tx, ty, label_len):
        """检查标注是否与已有标注重叠。"""
        w_est = label_len * 0.4  # 粗估文字宽度
        h_est = 1.2
        for bx, by, bw, bh in label_boxes:
            if (abs(tx - bx) < (w_est + bw) * 0.5 + 0.5 and
                    abs(ty - by) < (h_est + bh) * 0.5 + 0.3):
                return False
        return True

    labeled_count = 0
    for m in sorted_museums:
        # 生成简短标签
        label = m["name"]
        if len(label) > 40:
            label = label[:37] + "..."

        lon, lat = m["lon"], m["lat"]

        # 按地理区域确定偏移方向
        if lon < -100:       # 北美西海岸
            candidates = [(-3.0, 1.5), (-3.0, -1.5), (2.5, 1.5)]
        elif lon < -60:      # 北美东部 + 南美
            candidates = [(2.5, 1.5), (2.5, -1.5), (-3.0, 1.5), (2.5, 0)]
        elif lon < 0:        # 欧洲西部
            candidates = [(-3.0, 1.2), (2.5, 1.2), (-3.0, -1.2), (2.5, -1.2)]
        elif lon < 40:       # 欧洲东部 / 中东
            candidates = [(2.5, 1.2), (-3.0, 1.2), (2.5, -1.2), (-3.0, -1.2)]
        elif lon < 105:      # 南亚 / 中亚
            candidates = [(2.5, 1.5), (2.5, -1.5), (-3.0, 1.5)]
        else:                # 东亚 / 澳洲
            candidates = [(3.0, -1.5), (3.0, 1.5), (-3.0, -1.5), (3.0, 0)]

        placed = False
        for dx, dy in candidates:
            tx, ty = lon + dx, lat + dy
            if can_place(tx, ty, len(label)):
                ax.annotate(
                    label,
                    xy=(lon, lat),
                    xytext=(tx, ty),
                    fontsize=3.5, color='#444444',
                    arrowprops=dict(arrowstyle='-', color='#BBBBBB', linewidth=0.2),
                    zorder=12,
                )
                w_est = len(label) * 0.4
                label_boxes.append((tx, ty, w_est, 1.2))
                labeled_count += 1
                placed = True
                break

        # 如果所有候选位置都冲突，跳过该标注
        if not placed:
            pass  # 点仍然可见，只是没有文字标注

    print(f"🏷  标注了 {labeled_count}/{len(museums)} 座博物馆名称")

    # 图例
    legend_elements = [
        mpatches.Patch(facecolor='#D94F4F', edgecolor='white',
                       label='知名科技馆 (data/world)'),
        mpatches.Patch(facecolor='#4A90D9', edgecolor='white',
                       label='ArchDaily建筑项目 (data/archdaily)'),
    ]
    legend = ax.legend(
        handles=legend_elements, loc='lower left',
        fontsize=7, framealpha=0.95, edgecolor='#DDDDDD',
        prop=chinese_font if chinese_font else None,
        borderpad=0.8,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(0)

    # 隐藏坐标轴
    ax.set_xlim(-170, 180)
    ax.set_ylim(-55, 72)
    ax.set_aspect(1.3)
    ax.axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path, dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"✅ 世界地图已保存至: {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    world_dir = os.path.join(project_root, "data", "world")
    archdaily_dir = os.path.join(project_root, "data", "archdaily")
    output_dir = os.path.join(project_root, "code", "output")
    geojson_path = os.path.join(script_dir, "geodata", "world_countries.json")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "world_museums_map.png")

    print(f"📂 World 数据: {world_dir}")
    print(f"📂 ArchDaily 数据: {archdaily_dir}")
    print(f"🗺  GeoJSON: {geojson_path}")

    # 加载两个数据源
    world_museums = load_world_museums(world_dir)
    print(f"📊 World: {len(world_museums)} 座")

    archdaily_museums = load_archdaily_museums(archdaily_dir)
    print(f"📊 ArchDaily: {len(archdaily_museums)} 座")

    all_museums = world_museums + archdaily_museums
    print(f"📊 合计: {len(all_museums)} 座科技馆/博物馆")

    from collections import Counter
    countries = Counter(m.get("country", m["city"]) for m in all_museums)
    print(f"🌍 涵盖城市: {len(set(m['city'] for m in all_museums))} 个")

    plot_world_museums(all_museums, output_path, geojson_path)


if __name__ == "__main__":
    main()
