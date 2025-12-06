"""Compatibility shim: use pipelines.domains.operation_pipeline.OperationGenerator."""

from pipelines.domains.operation_pipeline import OperationGenerator as BusinessResearchGenerator

__all__ = ["BusinessResearchGenerator"]
