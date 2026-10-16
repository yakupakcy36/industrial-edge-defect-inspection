"""FastAPI application for Industrial Edge Defect Inspection & Statistical Process Control."""

import time
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_VISION_CONFIG
from src.inspector import EdgeDefectInspector
from src.generator import generate_brushed_metal_surface, inject_surface_defect
from api.schemas import (
    InspectionBox,
    InspectionResponse,
    SyntheticInspectionRequest,
    SPCStatusResponse,
    BenchmarkResponse,
    HealthResponse,
)

inspector: Optional[EdgeDefectInspector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global inspector
    inspector = EdgeDefectInspector(config=DEFAULT_VISION_CONFIG)
    yield


app = FastAPI(
    title="Industrial Edge Defect Inspection, 2D FFT & SPC API",
    description=(
        "Ultra-low latency Automated Optical Inspection (AOI) service combining SSIM, Sobel gradient "
        "energy, 2D FFT high-pass spectral residual filtering, morphological circularity analysis, "
        "and Shewhart / Western Electric Statistical Process Control (SPC) monitoring."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        engine_calibrated=(inspector is not None),
        frame_dimensions=f"{DEFAULT_VISION_CONFIG.image_width}x{DEFAULT_VISION_CONFIG.image_height}",
    )


@app.post("/inspect/synthetic", response_model=InspectionResponse, tags=["Inspection"])
def inspect_synthetic(req: SyntheticInspectionRequest):
    if inspector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inspector not initialized")

    base = generate_brushed_metal_surface(
        width=DEFAULT_VISION_CONFIG.image_width,
        height=DEFAULT_VISION_CONFIG.image_height,
        seed=req.seed,
    )
    surface, _ = inject_surface_defect(base, defect_type=req.defect_type, seed=req.seed)
    res = inspector.inspect(surface)

    boxes = [InspectionBox(**b.__dict__) for b in res.bounding_boxes]
    return InspectionResponse(
        has_defect=res.has_defect,
        defect_severity=res.defect_severity,
        anomaly_score=res.anomaly_score,
        fft_spectral_energy=res.fft_spectral_energy,
        defect_count=res.defect_count,
        primary_defect_type=res.primary_defect_type,
        bounding_boxes=boxes,
        latency_ms=res.latency_ms,
    )


@app.get("/spc/status", response_model=SPCStatusResponse, tags=["Statistical Process Control"])
def get_spc_status():
    """Return Shewhart 3-sigma control limits, first-pass yield, and Western Electric alerts."""
    if inspector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inspector not initialized")
    return SPCStatusResponse(**inspector.spc_monitor.evaluate_spc())


@app.get("/benchmark", response_model=BenchmarkResponse, tags=["Performance"])
def run_latency_benchmark():
    if inspector is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inspector not initialized")

    frame = generate_brushed_metal_surface(
        width=DEFAULT_VISION_CONFIG.image_width,
        height=DEFAULT_VISION_CONFIG.image_height,
        seed=10,
    )

    latencies = []
    inspector.inspect(frame)

    for _ in range(30):
        t0 = time.perf_counter()
        inspector.inspect(frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / max(1e-3, avg_lat)

    return BenchmarkResponse(
        frames_evaluated=len(latencies),
        avg_latency_ms=round(avg_lat, 2),
        throughput_fps=round(fps, 1),
        p95_latency_ms=round(p95_lat, 2),
        hardware_acceleration="CPU Vectorized (NumPy FFT + SciPy Ndimage)",
    )
