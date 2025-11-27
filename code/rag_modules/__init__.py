"""Backward-compatible re-exports from new module locations.

DEPRECATED: Import from core/, pipelines/, or utils/ directly.
This module is kept for backward compatibility only.
"""

# Core infrastructure
from core.index_construction import IndexConstructionModule
from core.retrieval_optimization import RetrievalOptimizationModule
from core.generation_integration import GenerationIntegrationModule

# Utils
from utils.data_preparation import DataPreparationModule

# Domain pipelines
from pipelines.domains.design_concept_pipeline import (
    DesignConceptGenerator,
    DesignConceptETL,
    DesignConceptPromptBuilder,
    DesignConceptVectorStore,
)
from pipelines.domains.exhibition_pipeline import (
    ExhibitionGenerator,
    ExhibitionPromptBuilder,
    ExhibitionVectorStore,
)
from pipelines.domains.public_service_pipeline import (
    PublicServiceGenerator,
    PublicServicePromptBuilder,
    PublicServiceVectorStore,
)
from pipelines.domains.central_hub_pipeline import (
    CentralHubGenerator,
    CentralHubPromptBuilder,
    CentralHubVectorStore,
)
from pipelines.domains.special_theater_pipeline import (
    SpecialTheaterGenerator,
    SpecialTheaterPromptBuilder,
    SpecialTheaterVectorStore,
)
from pipelines.domains.science_education_pipeline import (
    ScienceEducationGenerator,
    ScienceEducationPromptBuilder,
    ScienceEducationVectorStore,
)
from pipelines.domains.business_research_pipeline import (
    BusinessResearchGenerator,
    BusinessResearchPromptBuilder,
    BusinessResearchVectorStore,
)

# Orchestration
from pipelines.orchestration.brief_assembly_pipeline import BriefAssemblyPipeline

__all__ = [
    'DataPreparationModule',
    'IndexConstructionModule', 
    'RetrievalOptimizationModule',
    'GenerationIntegrationModule',
    'DesignConceptGenerator',
    'DesignConceptETL',
    'DesignConceptPromptBuilder',
    'DesignConceptVectorStore',
    'ExhibitionGenerator',
    'ExhibitionPromptBuilder',
    'ExhibitionVectorStore',
    'PublicServiceGenerator',
    'PublicServicePromptBuilder',
    'PublicServiceVectorStore',
    'CentralHubGenerator',
    'CentralHubPromptBuilder',
    'CentralHubVectorStore',
    'SpecialTheaterGenerator',
    'SpecialTheaterPromptBuilder',
    'SpecialTheaterVectorStore',
    'ScienceEducationGenerator',
    'ScienceEducationPromptBuilder',
    'ScienceEducationVectorStore',
    'BriefAssemblyPipeline',
    'BusinessResearchGenerator',
    'BusinessResearchPromptBuilder',
    'BusinessResearchVectorStore'
]

__version__ = "1.0.0"
