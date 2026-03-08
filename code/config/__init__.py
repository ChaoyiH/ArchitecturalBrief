"""Compatibility exports for project configuration symbols.

This package shares the name ``config`` with the legacy module
``code/config.py``. Importing ``config`` resolves to this package, so we
re-export symbols from the legacy module to keep existing imports working.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_legacy_config_module():
    legacy_path = Path(__file__).resolve().parent.parent / "config.py"
    spec = spec_from_file_location("_legacy_config_module", legacy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load legacy config module: {legacy_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_legacy = _load_legacy_config_module()

# Re-export legacy config symbols that are used across the project.
CODE_DIR = _legacy.CODE_DIR
PROJECT_ROOT = _legacy.PROJECT_ROOT
DATA_ROOT = _legacy.DATA_ROOT

RAGConfig = _legacy.RAGConfig
ExhibitionConfig = _legacy.ExhibitionConfig
PublicServiceConfig = _legacy.PublicServiceConfig
CentralHubConfig = _legacy.CentralHubConfig
SpecialTheaterConfig = _legacy.SpecialTheaterConfig
ScienceEducationConfig = _legacy.ScienceEducationConfig
BusinessResearchConfig = _legacy.BusinessResearchConfig
DesignConceptConfig = _legacy.DesignConceptConfig

DEFAULT_CONFIG = _legacy.DEFAULT_CONFIG
DEFAULT_EXHIBITION_CONFIG = _legacy.DEFAULT_EXHIBITION_CONFIG
DEFAULT_PUBLIC_SERVICE_CONFIG = _legacy.DEFAULT_PUBLIC_SERVICE_CONFIG
DEFAULT_CENTRAL_HUB_CONFIG = _legacy.DEFAULT_CENTRAL_HUB_CONFIG
DEFAULT_SPECIAL_THEATER_CONFIG = _legacy.DEFAULT_SPECIAL_THEATER_CONFIG
DEFAULT_SCIENCE_EDUCATION_CONFIG = _legacy.DEFAULT_SCIENCE_EDUCATION_CONFIG
DEFAULT_BUSINESS_RESEARCH_CONFIG = _legacy.DEFAULT_BUSINESS_RESEARCH_CONFIG
DEFAULT_DESIGN_CONCEPT_CONFIG = _legacy.DEFAULT_DESIGN_CONCEPT_CONFIG

__all__ = [
    "CODE_DIR",
    "PROJECT_ROOT",
    "DATA_ROOT",
    "RAGConfig",
    "ExhibitionConfig",
    "PublicServiceConfig",
    "CentralHubConfig",
    "SpecialTheaterConfig",
    "ScienceEducationConfig",
    "BusinessResearchConfig",
    "DesignConceptConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_EXHIBITION_CONFIG",
    "DEFAULT_PUBLIC_SERVICE_CONFIG",
    "DEFAULT_CENTRAL_HUB_CONFIG",
    "DEFAULT_SPECIAL_THEATER_CONFIG",
    "DEFAULT_SCIENCE_EDUCATION_CONFIG",
    "DEFAULT_BUSINESS_RESEARCH_CONFIG",
    "DEFAULT_DESIGN_CONCEPT_CONFIG",
]
