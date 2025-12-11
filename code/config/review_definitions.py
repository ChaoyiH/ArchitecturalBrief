"""Definitions for scales and lifecycle phases used by the Virtual Hearing Engine."""

from __future__ import annotations

from typing import Dict

SCALE_DEFINITIONS: Dict[str, str] = {
    "Urban": "建筑作为城市实体与外部环境的交互界面",
    "Venue": "建筑内部的系统逻辑、各功能区之间的连接关系与全局性支撑系统",
    "Unit": "具体功能空间的特性",
}

LIFECYCLE_DEFINITIONS: Dict[str, str] = {
    "Construction": "从设计到竣工的物理实现过程",
    "Operation": "开馆后的稳定使用阶段",
    "Renewal": "应对未来需求变化的演进阶段",
}

__all__ = ["SCALE_DEFINITIONS", "LIFECYCLE_DEFINITIONS"]
