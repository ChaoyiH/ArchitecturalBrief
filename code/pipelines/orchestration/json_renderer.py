"""Render Markdown task brief from an assembled JSON file.

This module isolates JSON parsing/loading so rendering can be run as a standalone step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .brief_assembly_pipeline import BriefAssemblyPipeline


class JsonBriefRenderer:
    """Utility to load module JSON and render Markdown."""

    def __init__(self) -> None:
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


__all__ = ["JsonBriefRenderer"]
