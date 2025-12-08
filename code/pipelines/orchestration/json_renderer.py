"""Smart JSON to Markdown renderer with customization strategies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class JSONRenderer:
    """Render structured JSON into Markdown using strategy-based formatting."""

    # Heuristic to detect if a string already contains Markdown-like markers
    MARKDOWN_HINT_RE = re.compile(r"[#|\*-]|\d+\.\s", re.MULTILINE)

    def render(
        self,
        data: Any,
        level: int = 3,
        key: Optional[str] = None,
        section_number: Optional[int] = None,
    ) -> str:
        if data is None:
            return "_(暂无数据)_\n"

        # Pass-through for strings
        if isinstance(data, str):
            return data.strip() + "\n" if data.strip() else "_(暂无数据)_\n"

        # Simple scalars
        if isinstance(data, (int, float)):
            return f"{data}\n"
        if isinstance(data, bool):
            return f"{'是' if data else '否'}\n"

        # Lists
        if isinstance(data, list):
            if not data:
                return "_(无列表数据)_\n"

            # Specific list customizations by parent key
            if key in {
                "benchmarking_cases",
                "trend_list",
            }:
                return self._render_cards(data, level)

            # Typology / suitability tables
            if key in {"types", "recommendations"}:
                return self._render_auto_table(data)

            # Auto-table for flat dict lists
            if self._is_simple_dict_list(data):
                return self._render_auto_table(data)

            # Fallback bullet list
            return "\n".join([f"- {self.render(item, level, key).strip()}" for item in data]) + "\n"

        # Dicts
        if isinstance(data, dict):
            if not data:
                return "_(暂无数据)_\n"

            # Technical indicators specialized renderer (triggered by parent key)
            if key and str(key).lower() in {"technical_indicators", "indicators"}:
                return self._render_technical_indicators(data)

            # Design concept specialized renderer (deductive layout)
            if key and str(key).lower() == "concept":
                return self._render_design_concept(data, level)

            # Exhibition specialized renderer (strategy-first layout)
            if key and str(key).lower() == "exhibition":
                return self._render_exhibition_system(data, level)

            # Science education specialized renderer
            if key and str(key).lower() == "science_education":
                return self._render_science_education(data, level)

            # Business research / back-of-house specialized renderer
            if key and str(key).lower() == "business_research":
                return self._render_business_research(data, level)

            # Special theater specialized renderer
            if key and str(key).lower() == "special_theater":
                return self._render_special_theater(data, level, section_number)

            # Central hub specialized renderer
            if key and str(key).lower() == "central_hub":
                return self._render_central_hub(data, level)

            # Operation specialized renderer (structured JSON -> Markdown)
            if key and str(key).lower() == "operation":
                return self._render_operation(data, level, section_number)

            # Key-specific tables
            if key == "function_area_ranges":
                return self._render_function_area_table(data)

            if key in {
                "standard_mandates",
                "similar_cases",
                "mandatory_rooms",
                "spatial_integration_matrix",
                "commercial_planning",
            }:
                return self._render_auto_table(data if isinstance(data, list) else data)

            # Suitability / typology nested handling
            if key in {"typology_data", "suitability_data"}:
                lines: List[str] = []
                for k, v in data.items():
                    lines.append(self._heading(k, level))
                    lines.append(self.render(v, level + 1, k))
                return "\n".join(lines)

            lines: List[str] = []
            for k, v in data.items():
                lowered = str(k).lower()

                if lowered in {"technical_indicators", "indicators"}:
                    lines.append(self._heading(k, level))
                    lines.append(self._render_technical_indicators(v))
                    continue

                # Inline key-specific handlers
                if lowered == "function_area_ranges":
                    lines.append(self._heading(k, level))
                    lines.append(self._render_function_area_table(v))
                    continue
                if lowered in {
                    "standard_mandates",
                    "similar_cases",
                    "mandatory_rooms",
                    "spatial_integration_matrix",
                    "commercial_planning",
                }:
                    lines.append(self._heading(k, level))
                    lines.append(self._render_auto_table(v))
                    continue
                if lowered in {"benchmarking_cases", "trend_list"}:
                    lines.append(self._heading(k, level))
                    lines.append(self._render_cards(v, level + 1))
                    continue
                if lowered in {"types", "recommendations"}:
                    lines.append(self._heading(k, level))
                    lines.append(self._render_auto_table(v))
                    continue

                lines.append(self._heading(k, level))
                lines.append(self.render(v, level + 1, lowered))
            return "\n".join(lines)

        return f"{data}\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _heading(self, key: str, level: int) -> str:
        return f"{'#' * min(level, 6)} {self._format_key(key)}\n"

    @staticmethod
    def _format_key(key: str) -> str:
        return str(key).replace("_", " ").title()

    @staticmethod
    def _is_simple_dict_list(items: Sequence[Any]) -> bool:
        if not items or not all(isinstance(i, dict) for i in items):
            return False
        keys = list(items[0].keys())
        if not keys:
            return False
        for item in items:
            if set(item.keys()) != set(keys):
                return False
            if any(isinstance(v, (list, dict)) for v in item.values()):
                return False
        return True

    def _render_auto_table(self, items: Sequence[Any]) -> str:
        if not items:
            return "_(无表格数据)_\n"
        if not isinstance(items, list):
            return self.render(items)
        if not self._is_simple_dict_list(items):
            # fallback to bullets
            return "\n".join([f"- {self.render(i).strip()}" for i in items]) + "\n"

        headers = list(items[0].keys())
        header_row = "| " + " | ".join(self._format_key(h) for h in headers) + " |"
        divider = "| " + " | ".join(["---"] * len(headers)) + " |"
        rows = []
        for item in items:
            row = "| " + " | ".join(self._stringify(item.get(h, "")) for h in headers) + " |"
            rows.append(row)
        return "\n".join([header_row, divider, *rows]) + "\n"

    def _render_function_area_table(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict) or not data:
            return "_(无表格数据)_\n"
        headers = ["功能区", "占比范围", "建议面积"]
        lines = ["| " + " | ".join(headers) + " |", "| --- | --- | --- |"]
        for name, payload in data.items():
            if not isinstance(payload, dict):
                lines.append(f"| {self._stringify(name)} |  |  |")
                continue
            pct = self._range_str(payload.get("percent_min"), payload.get("percent_max"), suffix="%")
            area = self._range_str(payload.get("area_min_sqm"), payload.get("area_max_sqm"), suffix=" ㎡")
            lines.append(f"| {self._stringify(name)} | {pct} | {area} |")
        return "\n".join(lines) + "\n"

    def _render_cards(self, items: Sequence[Any], level: int) -> str:
        if not items:
            return "_(暂无数据)_\n"
        lines: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                lines.append(f"- {self._stringify(item)}")
                continue
            title = item.get("case_name") or item.get("trend_name") or item.get("feature_name") or item.get("name") or "项"
            lines.append(f"{'#' * min(level, 6)} {title}\n")
            # metadata/details bullets
            for k, v in item.items():
                if k in {"case_name", "trend_name", "feature_name", "name"}:
                    continue
                if k in {"evidence_quote", "evidence_snippet"}:
                    continue
                lines.append(f"- **{self._format_key(k)}**: {self._stringify(v)}")
            # evidence
            evidence = item.get("evidence_quote") or item.get("evidence_snippet")
            if evidence:
                lines.append(f"> {self._stringify(evidence)}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return "; ".join(str(v) for v in value)
        return str(value)

    @staticmethod
    def _range_str(min_v: Any, max_v: Any, suffix: str = "") -> str:
        if min_v is None and max_v is None:
            return ""
        if min_v is None:
            return f"≤ {max_v}{suffix}"
        if max_v is None:
            return f"≥ {min_v}{suffix}"
        return f"{min_v}{suffix} ~ {max_v}{suffix}"

    # ------------------------------------------------------------------
    # Specialized renderers
    # ------------------------------------------------------------------
    def _render_technical_indicators(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return self.render(data)

        target_area = data.get("target_area")
        target_cls = data.get("target_classification") or {}
        compliance = data.get("compliance") or {}
        industry = data.get("industry_reference") or {}
        similar = data.get("similar_cases") or []

        standard_basis = compliance.get("standard_source") or "《科学技术馆建设标准》"
        function_basis = compliance.get("function_basis") or standard_basis
        planning_basis = industry.get("basis") or "同类项目规划控制指标参考"
        cases_basis = industry.get("similar_basis") or "同类案例数据库"

        def fmt(val: Any, suffix: str = "") -> str:
            if val is None or val == "":
                return "-"
            if isinstance(val, (int, float)):
                return f"{val}{suffix}" if not suffix else f"{val}{suffix}"
            return str(val)

        def fmt_range(min_v: Any, max_v: Any, suffix: str = "") -> str:
            if min_v is None and max_v is None:
                return "-"
            if min_v is None:
                return f"≤ {max_v}{suffix}"
            if max_v is None:
                return f"≥ {min_v}{suffix}"
            return f"{min_v}{suffix} ~ {max_v}{suffix}"

        def fmt_area(val: Any) -> str:
            if val is None:
                return "-"
            try:
                return f"{float(val):.2f}"
            except Exception:
                return str(val)

        # Table 1 --------------------------------------------------
        inst = compliance.get("instantaneous_capacity") or {}
        inst_range = fmt_range(inst.get("min"), inst.get("max"))
        core_rows = [
            ("总建筑面积", fmt(target_area, " ㎡"), fmt(compliance.get("total_area_sqm"), " ㎡"), ""),
            ("建设规模分级", fmt(target_cls.get("class_name")), fmt(compliance.get("size_class_name")), ""),
            ("设计使用年限", fmt(target_cls.get("design_life")), fmt(compliance.get("design_life")), ""),
            ("瞬时最大容量", inst_range, "依据总面积估算", ""),
        ]
        tbl1 = ["| 指标项 | 设定目标 | 规范分级/依据 | 备注 |", "| --- | --- | --- | --- |"]
        tbl1 += [f"| {a} | {b} | {c} | {d or ''} |" for a, b, c, d in core_rows]
        # Table 2 --------------------------------------------------
        func_ranges = compliance.get("function_area_ranges") or {}
        exhibit_est = compliance.get("exhibit_count_estimate") or {}
        exhibition_area = compliance.get("exhibition_area_estimate")
        zone_map = {
            "exhibition_education": "展览教育区",
            "public_service": "公众服务区",
            "business_research": "业务研究区",
            "management": "管理保障区",
        }
        tbl2 = ["| 功能分区 | 规范占比范围 | 建议面积范围 | 备注/详细指标 |", "| --- | --- | --- | --- |"]
        for key, payload in func_ranges.items():
            zone_name = zone_map.get(key, str(key))
            if not isinstance(payload, dict):
                tbl2.append(f"| {zone_name} | - | - | - |")
                continue
            pct = fmt_range(payload.get("percent_min"), payload.get("percent_max"), "%")
            area = fmt_range(payload.get("area_min_sqm"), payload.get("area_max_sqm"), " ㎡")
            note = ""
            if key == "exhibition_education":
                exhibits = fmt_range(exhibit_est.get("min"), exhibit_est.get("max"), " 件")
                note_parts = []
                if exhibits != "-":
                    note_parts.append(f"展品数估算: {exhibits}")
                if exhibition_area is not None:
                    note_parts.append(f"展览面积估算: {exhibition_area} ㎡")
                note = "; ".join(note_parts)
            tbl2.append(f"| {zone_name} | {pct} | {area} | {note} |")

        # Table 3 --------------------------------------------------
        far = industry.get("floor_area_ratio") or []
        density = industry.get("building_density") or []
        green = industry.get("green_ratio") or []
        tbl3 = ["| 指标类型 | 本项目参考区间 |", "| --- | --- |"]
        tbl3.append(
            f"| 容积率 (FAR) | {fmt_range(far[0] if len(far)>0 else None, far[1] if len(far)>1 else None)} |"
        )
        tbl3.append(
            f"| 建筑密度 | {fmt_range(density[0] if len(density)>0 else None, density[1] if len(density)>1 else None)} |"
        )
        tbl3.append(
            f"| 绿地率 | {fmt_range(green[0] if len(green)>0 else None, green[1] if len(green)>1 else None)} |"
        )

        # Table 4 --------------------------------------------------
        tbl4 = ["| 案例名称 | 建筑面积 | 建设年份 | 地区 |", "| --- | --- | --- | --- |"]
        for case in similar:
            if not isinstance(case, dict):
                continue
            name = case.get("name") or "-"
            area = fmt_area(case.get("area"))
            year = fmt(case.get("year"))
            source = fmt(case.get("source"))
            tbl4.append(f"| {name} | {area} | {year} | {source} |")

        parts = [
            f"**表1 核心建设指标**（依据：{standard_basis}）",
            "\n".join(tbl1),
            f"**表2 功能分区与展陈规模**（依据：{function_basis}）",
            "\n".join(tbl2),
            f"**表3 规划控制指标**（依据：{planning_basis}）",
            "\n".join(tbl3),
            f"**表4 对标案例**（依据：{cases_basis}）",
            "\n".join(tbl4) if len(tbl4) > 2 else "_(暂无对标案例)_",
        ]
        return "\n\n".join(parts) + "\n"

    def _render_design_concept(self, data: Dict[str, Any], level: int) -> str:
        """Render the design concept section with a deductive layout in Chinese."""
        if not isinstance(data, dict):
            return self.render(data, level)

        # Extract blocks with safe defaults
        concept_clusters = data.get("concept_clusters") or []
        concept_summary = data.get("concept_summary") or data.get("summary_statement")
        sources = data.get("concept_sources") or {}
        theoretical_basis = sources.get("theoretical_basis")
        standard_mandates = sources.get("standard_mandates") or []

        trends = data.get("design_trends") or {}
        trend_list = trends.get("trend_list") or []
        innovative_suggestions = trends.get("innovative_suggestions") or []

        benchmarking_cases = data.get("benchmarking_cases") or []

        # Heading levels
        h = "#" * min(level, 6)
        h_sub = "#" * min(level + 1, 6)

        lines: List[str] = []

        # 1) 概念聚类/理念来源
        if concept_summary:
            lines.append(f"{h} 理念总述\n\n> {concept_summary.strip()}")
        elif theoretical_basis:
            lines.append(f"{h} 理念来源\n\n> {theoretical_basis.strip()}")

        if concept_clusters:
            cluster_header = "| 聚类名称 | 核心哲学 | 代表案例 | 证据摘录 |"
            cluster_divider = "| --- | --- | --- | --- |"
            cluster_rows: List[str] = []
            for item in concept_clusters:
                if not isinstance(item, dict):
                    continue
                name = item.get("cluster_name", "-")
                philosophy = str(item.get("core_philosophy", "-")).replace("\n", "<br>")
                examples = item.get("examples") or []
                ex_str = "<br>".join(str(e) for e in examples) if examples else "-"
                evidence = item.get("evidence_snippet", "-")
                cluster_rows.append(f"| {name} | {philosophy} | {ex_str} | {evidence} |")
            if cluster_rows:
                lines.append(f"{h} 概念聚类\n\n" + "\n".join([cluster_header, cluster_divider, *cluster_rows]))
        elif standard_mandates:
            table_lines: List[str] = []
            header = "| 来源 | 关注条款 | 内容 |"
            divider = "| --- | --- | --- |"
            for item in standard_mandates:
                if not isinstance(item, dict):
                    continue
                src = str(item.get("source", "-")).replace("参考案例：", "", 1)
                clause = item.get("clause", "-")
                content = item.get("content", "-")
                table_lines.append(f"| {src} | {clause} | {content} |")
            if table_lines:
                lines.append(f"{h} 相关案例\n\n" + "\n".join([header, divider, *table_lines]))

        # 3) 设计趋势（表 + 列表）
        if trend_list:
            trend_rows: List[str] = []
            t_header = "| 趋势名称 | 描述 | 代表案例 |"
            t_divider = "| --- | --- | --- |"
            for item in trend_list:
                if not isinstance(item, dict):
                    continue
                name = item.get("trend_name", "-")
                desc = str(item.get("description", "-")).replace("\n", "<br>")
                evidence_cases = item.get("evidence_cases") or []
                ev_str = "<br>".join(str(e) for e in evidence_cases) if evidence_cases else "-"
                trend_rows.append(f"| {name} | {desc} | {ev_str} |")
            if trend_rows:
                lines.append(f"{h} 设计趋势\n\n" + "\n".join([t_header, t_divider, *trend_rows]))

        if innovative_suggestions:
            bullet_block = "\n".join(f"- {sug}" for sug in innovative_suggestions)
            lines.append(f"{h_sub} 创新建议 (Innovative Suggestions)\n\n{bullet_block}")

        # 4) 对标案例（高密度表格）
        if benchmarking_cases:
            b_rows: List[str] = []
            b_header = "| 案例信息 | 相似性逻辑 | 设计特征 | 核心证据/隐喻 |"
            b_divider = "| --- | --- | --- | --- |"
            for case in benchmarking_cases:
                if not isinstance(case, dict):
                    continue
                name = case.get("case_name", "-")
                location = (case.get("metadata") or {}).get("location")
                case_info = f"{name}{' / ' + location if location else ''}"
                similarity = case.get("similarity_logic", "-")

                details = case.get("design_details") or {}
                form_logic = details.get("form_logic") or []
                materiality = details.get("materiality") or []
                spatial = details.get("spatial_features") or []

                feature_parts: List[str] = []
                if form_logic:
                    feature_parts.append(f"**形体**: {'; '.join(str(x) for x in form_logic)}")
                if materiality:
                    feature_parts.append(f"**材质**: {'; '.join(str(x) for x in materiality)}")
                if spatial:
                    feature_parts.append(f"**空间**: {'; '.join(str(x) for x in spatial)}")
                features = "<br>".join(feature_parts) if feature_parts else "-"

                evidence = case.get("evidence_snippet", "-")
                b_rows.append(f"| {case_info} | {similarity} | {features} | {evidence} |")

            if b_rows:
                lines.append(f"{h} 对标案例\n\n" + "\n".join([b_header, b_divider, *b_rows]))

        if not lines:
            return "_(暂无数据)_\n"

        return "\n\n".join(lines) + "\n"

    def _render_exhibition_system(self, data: Dict[str, Any], level: int) -> str:
        """Render the exhibition section with a strategy-first layout."""
        if not isinstance(data, dict):
            return self.render(data, level)

        h = "#" * min(level, 6)
        h_sub = "#" * min(level + 1, 6)

        lines: List[str] = []

        trend_data = data.get("trend_analysis_data") or {}
        innovative_features = trend_data.get("innovative_features") or []

        # 1) 核心策略（顶部）
        if innovative_features:
            lines.append(f"{h} 核心策略：黄河文脉与科技融合 (Project Strategy)")
            lines.append("\n".join(f"> {feat}" for feat in innovative_features))

        # 2) 核心规划数据
        core = data.get("core_planning_data") or {}
        standard = core.get("standard_compliance") or {}
        baseline = core.get("baseline_themes") or []

        if standard or baseline:
            lines.append(f"{h} 核心规划数据")

            key_bullets: List[str] = []
            mandatory = standard.get("mandatory_zones")
            spatial = standard.get("spatial_requirements")
            if mandatory:
                key_bullets.append(f"- **必备功能分区**: {mandatory}")
            if spatial:
                key_bullets.append(f"- **空间要求**: {spatial}")
            if key_bullets:
                lines.append("\n".join(key_bullets))

            if baseline:
                b_header = "| 功能单元 | 空间定义 | 依据 |"
                b_divider = "| --- | --- | --- |"
                b_rows: List[str] = []
                for item in baseline:
                    if not isinstance(item, dict):
                        continue
                    theme = item.get("theme_name", "-")
                    desc = item.get("description", "-")
                    ref = item.get("reference", "-")
                    # 清理“片段X | ”前缀
                    ref = re.sub(r"^片段\d+\s*\|\s*", "", str(ref))
                    b_rows.append(f"| {theme} | {desc} | {ref} |")
                if b_rows:
                    lines.append("\n".join([b_header, b_divider, *b_rows]))

        # 3) 趋势（结构化列表）
        global_trends = trend_data.get("global_trends") or []
        if global_trends:
            lines.append(f"{h} 趋势")
            for trend in global_trends:
                if not isinstance(trend, dict):
                    continue
                name = trend.get("trend_name", "-")
                desc = trend.get("description", "-")
                lines.append(f"{h_sub} {name}")
                if desc:
                    lines.append(desc)
                evidence_cases = trend.get("evidence_cases") or []
                if evidence_cases:
                    ev_lines: List[str] = []
                    for case in evidence_cases:
                        if isinstance(case, dict):
                            c_name = case.get("case_name", "-")
                            snippet = case.get("snippet", "-")
                            ev_lines.append(f"- {c_name}: {snippet}")
                        else:
                            ev_lines.append(f"- {case}")
                    lines.append("\n".join(ev_lines))

        if not lines:
            return "_(暂无数据)_\n"

        return "\n\n".join(lines) + "\n"

    def _render_operation(self, data: Dict[str, Any], level: int, section_number: Optional[int]) -> str:
        """Render operation module from structured JSON into Markdown."""
        if not isinstance(data, dict):
            return self.render(data, level)

        base = section_number or 5
        h = "#" * min(level, 6)
        lines: List[str] = []

        # 5.1 全球运营模式案例循证
        lines.append(f"{h} {base}.1 全球运营模式案例循证 (Evidence of Operational Models)")
        evidence_list = data.get("global_evidence") or []
        if evidence_list:
            lines.append("> 本节内容基于数据库案例检索生成，仅陈述事实。")
            for item in evidence_list:
                if not isinstance(item, dict):
                    lines.append(f"- {self._stringify(item)}")
                    continue
                category = item.get("category", "-")
                desc = item.get("description")
                cases = item.get("cases") or []
                bullet = f"- **{category}**"
                if desc:
                    bullet += f": {self._stringify(desc)}"
                lines.append(bullet)
                if cases:
                    lines.append("\n".join(f"  - {self._stringify(c)}" for c in cases))
        else:
            lines.append("_(暂无案例数据)_")

        # 5.2 商业空间落位策略（表格）
        lines.append(f"\n{h} {base}.2 商业空间落位策略 (Spatial Integration Strategy)")
        strategy_list = data.get("spatial_strategy") or []
        if strategy_list:
            header = "| 业态类型 | 空间耦合建议 | 规范/案例依据 |"
            divider = "| --- | --- | --- |"
            rows: List[str] = []
            for item in strategy_list:
                if not isinstance(item, dict):
                    continue
                t = item.get("type", "-")
                proposal = item.get("proposal", "-")
                ref = item.get("reference", "-")
                rows.append(f"| {self._stringify(t)} | {self._stringify(proposal)} | {self._stringify(ref)} |")
            if rows:
                lines.append("\n".join([header, divider, *rows]))
            else:
                lines.append("_(暂无落位策略)_")
        else:
            lines.append("_(暂无落位策略)_")

        # 5.3 济南项目定制化建议
        lines.append(f"\n{h} {base}.3 济南项目定制化建议 (Tailored Recommendations)")
        tailored = data.get("tailored_recommendations") or []
        if tailored:
            for item in tailored:
                if not isinstance(item, dict):
                    lines.append(f"- {self._stringify(item)}")
                    continue
                topic = item.get("topic", "主题")
                strategy = item.get("strategy")
                rationale = item.get("rationale")
                lines.append(f"- **{self._stringify(topic)}**")
                if strategy:
                    lines.append(f"  - *策略*: {self._stringify(strategy)}")
                if rationale:
                    lines.append(f"  - *依据*: {self._stringify(rationale)}")
        else:
            lines.append("_(暂无定制化建议)_")

        # 5.4 运营支持空间技术要求
        lines.append(f"\n{h} {base}.4 运营支持空间技术要求 (Technical Requirements)")
        tech_reqs = data.get("technical_requirements") or []
        if tech_reqs:
            for req in tech_reqs:
                lines.append(f"- {self._stringify(req)}")
        else:
            lines.append("_(暂无技术要求)_")

        return "\n".join(lines) + "\n"

    def _render_science_education(self, data: Dict[str, Any], level: int) -> str:
        """Render science education with hard/soft separation."""
        if not isinstance(data, dict):
            return self.render(data, level)

        h = "#" * min(level, 6)
        lines: List[str] = []

        activity = data.get("activity_programming") or {}
        trends = activity.get("trends") or []

        # Hardcoded normative table for 特大型馆
        lines.append(f"{h} 规范占比与功能基准（特大型馆）")
        norm_table = [
            "| 功能分区 | 规范占比范围 | 建议内容/备注 |",
            "| --- | --- | --- |",
            "| 展览教育用房 | 55% ~ 60% | 核心功能区。含常设/临时展厅、科普实验室、特效影院等。 |",
            "| 公众服务用房 | 15% ~ 20% | 含门厅、餐饮、文创商店、休息区等。 |",
            "| 业务研究用房 | 10% ~ 15% | 含行政办公、科研办公室、展品维修车间。 |",
            "| 管理保障用房 | 10% ~ 15% | 含安保监控、设备机房、总务仓库。 |",
        ]
        lines.append("\n".join(norm_table))
        lines.append("*数据来源：《科学技术馆建设标准》(建标 101-2007) - 特大型馆指标*")

        # Activity programming as compact definition-style list
        if trends:
            lines.append(f"{h} 活动策划")
            header = "| 活动名称 | 空间需求 | 参考案例 |"
            divider = "| --- | --- | --- |"
            rows: List[str] = []
            for t in trends:
                if not isinstance(t, dict):
                    continue
                name = t.get("activity_name", "-")
                need = t.get("spatial_need", "-")
                ref = t.get("reference_case", "-")
                rows.append(f"| {name} | {need} | {ref} |")
            if rows:
                lines.append("\n".join([header, divider, *rows]))

        if not lines:
            return "_(暂无数据)_\n"
        return "\n\n".join(lines) + "\n"

    def _render_business_research(self, data: Dict[str, Any], level: int) -> str:
        """Render back-of-house with program focus."""
        if not isinstance(data, dict):
            return self.render(data, level)

        h = "#" * min(level, 6)
        lines: List[str] = []

        normative = data.get("normative_analysis") or {}
        highlights = data.get("empirical_highlights") or []
        program = data.get("space_program_suggestion") or {}

        # Normative summary
        target_level = normative.get("target_level")
        area_range = normative.get("suggested_area_range")
        mandatory = normative.get("mandatory_sub_functions") or []
        if target_level or area_range or mandatory:
            lines.append(f"{h} 规范与功能要点")
            if target_level:
                lines.append(f"- **目标等级**: {target_level}")
            if area_range:
                lines.append(f"- **建议面积**: {area_range}")
            if mandatory:
                lines.append(f"- **必要功能**: {', '.join(str(m) for m in mandatory)}")

        # Empirical highlights table
        if highlights:
            lines.append(f"{h} 亮点案例")
            header = "| 创新特征 | 案例来源 | 描述 |"
            divider = "| --- | --- | --- |"
            rows = []
            for item in highlights:
                if not isinstance(item, dict):
                    continue
                feat = item.get("feature_name", "-")
                source = item.get("case_source", "-")
                desc = item.get("description", "-")
                rows.append(f"| {feat} | {source} | {desc} |")
            if rows:
                lines.append("\n".join([header, divider, *rows]))

        # Space program suggestion
        if program:
            lines.append(f"{h} 空间规划建议")
            core_zones = program.get("core_zones") or []
            if core_zones:
                lines.append("- **核心区块**:")
                lines.append("\n".join(f"  - {z}" for z in core_zones))
            adjacency = program.get("adjacency_requirements")
            if adjacency:
                lines.append(f"**关键邻接**: {adjacency}")

        if not lines:
            return "_(暂无数据)_\n"
        return "\n\n".join(lines) + "\n"

    def _render_special_theater(self, data: Dict[str, Any], level: int, section_number: Optional[int]) -> str:
        """Render special theater with inductive flow: typology -> trends -> recommendations."""
        if not isinstance(data, dict):
            return self.render(data, level)

        base = section_number or 9
        h = "#" * min(level, 6)
        lines: List[str] = []

        suitability = data.get("suitability_data") or {}
        typology = data.get("typology_data") or {}
        trend_data = data.get("trend_data") or {}

        # 1) Typology first
        types = typology.get("types") or []
        lines.append(f"{h} {base}.1 影厅类型谱系 (Theater Typologies)")
        if types:
            header = "| 类型名称 | 定义与特征 | 典型配置 |"
            divider = "| --- | --- | --- |"
            rows = []
            for t in types:
                if not isinstance(t, dict):
                    continue
                name = t.get("name", "-")
                desc = t.get("description", "-")
                evidence = t.get("evidence") or t.get("configuration") or "-"
                rows.append(f"| {self._stringify(name)} | {self._stringify(desc)} | {self._stringify(evidence)} |")
            lines.append("\n".join([header, divider, *rows]))
        else:
            lines.append("_(暂无类型数据)_")

        # 2) Trends second
        trend_list = trend_data.get("trend_list") or []
        strategies = trend_data.get("actionable_strategies") or []
        lines.append(f"\n{h} {base}.2 技术演进趋势 (Technological Trends)")
        if trend_list:
            for tr in trend_list:
                if not isinstance(tr, dict):
                    lines.append(f"- {self._stringify(tr)}")
                    continue
                name = tr.get("trend_name", "-")
                desc = tr.get("description", "-")
                lines.append(f"- **{self._stringify(name)}**: {self._stringify(desc)}")
        else:
            lines.append("_(暂无趋势数据)_")
        if strategies:
            lines.append("  - **应对策略 (Strategies):**")
            for s in strategies:
                lines.append(f"    - {self._stringify(s)}")

        # 3) Suitability recommendations third
        recs = suitability.get("recommendations") or []
        lines.append(f"\n{h} {base}.3 选型与空间建议 (Selection & Spatial Recommendations)")
        if recs:
            header = "| 推荐类型 | 建议容量 | 核心功能 | 选型逻辑 |"
            divider = "| --- | --- | --- | --- |"
            rows = []
            for r in recs:
                if not isinstance(r, dict):
                    continue
                t = r.get("theater_type", "-")
                cap = r.get("recommended_capacity", "-")
                func = r.get("core_function", "-")
                rationale = r.get("rationale", "-")
                rows.append(f"| {self._stringify(t)} | {self._stringify(cap)} | {self._stringify(func)} | {self._stringify(rationale)} |")
            lines.append("\n".join([header, divider, *rows]))
        else:
            lines.append("_(暂无选型建议)_")

        spatial_reqs = suitability.get("spatial_requirements") or []
        if spatial_reqs:
            reqs_text = "; ".join(self._stringify(r) for r in spatial_reqs)
            lines.append(f"> **技术要求**: {reqs_text}")

        return "\n".join(lines) + "\n"

    def _render_central_hub(self, data: Dict[str, Any], level: int) -> str:
        """Render central hub with visual anatomy focus."""
        if not isinstance(data, dict):
            return self.render(data, level)

        h = "#" * min(level, 6)
        h_sub = "#" * min(level + 1, 6)
        lines: List[str] = []

        trends = data.get("design_trends") or {}
        panorama = data.get("morphology_panorama") or {}
        cases = data.get("benchmarking_cases") or []

        # Strategic proposal
        proposal = trends.get("spatial_strategy_proposal")
        if proposal:
            lines.append(f"{h} 空间策略")
            lines.append(f"> {proposal}")

        # Morphology panorama
        archetypes = panorama.get("archetypes") or []
        if archetypes:
            lines.append(f"{h} 形态谱系")
            for a in archetypes:
                if not isinstance(a, dict):
                    continue
                name = a.get("name", "-")
                desc = a.get("spatial_diagram_desc", "-")
                lines.append(f"- **{name}**: {desc}")

        # Benchmarking cases table
        if cases:
            lines.append(f"{h} 对标案例")
            header = "| 案例名称 | 基础数据 | 空间解剖 | 流线整合 |"
            divider = "| --- | --- | --- | --- |"
            rows = []
            for case in cases:
                if not isinstance(case, dict):
                    continue
                name = case.get("case_name", "-")
                basic = case.get("basic_data") or {}
                base_strs = []
                if basic.get("total_area"):
                    base_strs.append(f"面积: {basic.get('total_area')}")
                if basic.get("atrium_height"):
                    base_strs.append(f"中庭高: {basic.get('atrium_height')}")
                base_cell = "<br>".join(base_strs) if base_strs else "-"

                anatomy = case.get("spatial_anatomy") or {}
                anatomy_parts: List[str] = []
                for k, v in anatomy.items():
                    label = self._format_key(k)
                    if isinstance(v, list):
                        anatomy_parts.append(f"**{label}**: {'; '.join(str(x) for x in v)}")
                    else:
                        anatomy_parts.append(f"**{label}**: {v}")
                anatomy_cell = "<br>".join(anatomy_parts) if anatomy_parts else "-"

                circulation = case.get("circulation_integration", "-")
                rows.append(f"| {name} | {base_cell} | {anatomy_cell} | {circulation} |")
            if rows:
                lines.append("\n".join([header, divider, *rows]))

        if not lines:
            return "_(暂无数据)_\n"
        return "\n\n".join(lines) + "\n"


class JsonBriefRenderer:
    """Utility to load module JSON and render Markdown using JSONRenderer."""

    def __init__(self) -> None:
        from .brief_assembly_pipeline import BriefAssemblyPipeline  # local import to avoid cycle

        self._assembler = BriefAssemblyPipeline()

    @staticmethod
    def load_sections(json_path: str | Path) -> Dict[str, Any]:
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"找不到 JSON 文件: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("JSON 内容必须是对象（dict）")
        return data

    def render(self, project_name: str, project_features: str, json_path: str | Path) -> Dict[str, Any]:
        sections = self.load_sections(json_path)
        result = self._assembler.generate_brief(project_name, project_features, sections)
        markdown = result.get("response", "")
        return {"response": markdown, "sections": sections}


__all__ = ["JSONRenderer", "JsonBriefRenderer"]
