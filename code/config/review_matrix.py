"""Persona matrix for Virtual Hearing Engine.

Defines stakeholder/scale/lifecycle enums and a sparse matrix of valid
(review_scope) combinations.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


class Stakeholder(str, Enum):
    GOVT = "政府代表"
    DIRECTOR = "科技馆馆长"
    CURATOR = "策展总监"
    OPERATOR = "商业/物业经营者"
    TEACHER = "带队老师"
    STUDENT_PRI = "小学生"
    STUDENT_SEC = "中学生"
    PARENT = "带娃家长"
    TOURIST = "外地游客"
    CITIZEN = "本地市民"


class Scale(str, Enum):
    URBAN = "城市与外部级"
    VENUE = "场馆与体系级"
    UNIT = "单元与设备级"


class Lifecycle(str, Enum):
    CONSTRUCTION = "建设交付期"
    OPERATION = "常态运营期"
    RENEWAL = "更新迭代期"


def _full_scales() -> List[Scale]:
    return [Scale.URBAN, Scale.VENUE, Scale.UNIT]


def _full_phases() -> List[Lifecycle]:
    return [Lifecycle.CONSTRUCTION, Lifecycle.OPERATION, Lifecycle.RENEWAL]


def _op_renewal() -> List[Lifecycle]:
    return [Lifecycle.OPERATION, Lifecycle.RENEWAL]


def get_persona_matrix() -> Dict[Stakeholder, List[Tuple[Scale, Lifecycle]]]:
    """Return sparse persona matrix mapping stakeholder to allowed (scale, phase)."""

    matrix: Dict[Stakeholder, List[Tuple[Scale, Lifecycle]]] = {}

    # 1) Strategic: full connection
    strategic_scales = _full_scales()
    strategic_phases = _full_phases()
    matrix[Stakeholder.GOVT] = [(s, p) for s in strategic_scales for p in strategic_phases]
    matrix[Stakeholder.DIRECTOR] = [(s, p) for s in strategic_scales for p in strategic_phases]

    # 2) Curator: all scales x operation/renewal
    matrix[Stakeholder.CURATOR] = [(s, p) for s in _full_scales() for p in _op_renewal()]

    # 3) Operator: venue/unit x operation/renewal
    matrix[Stakeholder.OPERATOR] = [
        (s, p)
        for s in [Scale.VENUE, Scale.UNIT]
        for p in _op_renewal()
    ]

    # 4) Groups (Teacher, Parent): all scales x operation
    matrix[Stakeholder.TEACHER] = [(s, Lifecycle.OPERATION) for s in _full_scales()]
    matrix[Stakeholder.PARENT] = [(s, Lifecycle.OPERATION) for s in _full_scales()]

    # 5) Students (Pri, Sec): venue/unit x operation
    matrix[Stakeholder.STUDENT_PRI] = [
        (s, Lifecycle.OPERATION) for s in [Scale.VENUE, Scale.UNIT]
    ]
    matrix[Stakeholder.STUDENT_SEC] = [
        (s, Lifecycle.OPERATION) for s in [Scale.VENUE, Scale.UNIT]
    ]

    # 6) Tourist: all scales x operation
    matrix[Stakeholder.TOURIST] = [(s, Lifecycle.OPERATION) for s in _full_scales()]

    # 7) Citizen: full connection
    matrix[Stakeholder.CITIZEN] = [(s, p) for s in _full_scales() for p in _full_phases()]

    return matrix


__all__ = [
    "Stakeholder",
    "Scale",
    "Lifecycle",
    "get_persona_matrix",
]
