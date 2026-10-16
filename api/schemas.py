"""Pydantic v2 schemas for the Industrial Edge Defect Inspection & SPC API."""

from typing import List
from pydantic import BaseModel, Field


class InspectionBox(BaseModel):
    box_id: int
    y_min: int
    x_min: int
    y_max: int
    x_max: int
    area_pixels: int
    aspect_ratio: float
    circularity: float
    defect_type: str


class InspectionResponse(BaseModel):
    has_defect: bool
    defect_severity: str = Field(..., description="NORMAL, MINOR, CRITICAL")
    anomaly_score: float
    fft_spectral_energy: float
    defect_count: int
    primary_defect_type: str
    bounding_boxes: List[InspectionBox]
    latency_ms: float


class SyntheticInspectionRequest(BaseModel):
    defect_type: str = Field(default="scratch", description="scratch, pit, crack, normal")
    seed: int = Field(default=42, ge=0)


class SPCStatusResponse(BaseModel):
    samples_in_window: int
    process_in_control: bool
    first_pass_yield_pct: float
    current_mean_score: float
    center_line: float
    ucl_3sigma: float
    lcl_3sigma: float
    violated_rules: List[str]


class BenchmarkResponse(BaseModel):
    frames_evaluated: int
    avg_latency_ms: float
    throughput_fps: float
    p95_latency_ms: float
    hardware_acceleration: str


class HealthResponse(BaseModel):
    status: str
    engine_calibrated: bool
    frame_dimensions: str
