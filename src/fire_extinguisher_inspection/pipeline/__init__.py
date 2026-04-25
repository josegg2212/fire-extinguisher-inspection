"""Pipeline de inspección visual de extintores."""

from fire_extinguisher_inspection.pipeline.inspection_pipeline import InspectionPipeline
from fire_extinguisher_inspection.pipeline.result_schema import DetectionResult, InspectionResult

__all__ = ["DetectionResult", "InspectionPipeline", "InspectionResult"]
