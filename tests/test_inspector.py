"""Test suite for edge defect inspector, 2D FFT spectral energy, and SPC control charts."""

import pytest
from src.config import DEFAULT_VISION_CONFIG
from src.generator import generate_brushed_metal_surface, inject_surface_defect
from src.inspector import EdgeDefectInspector


@pytest.fixture
def inspector():
    return EdgeDefectInspector(config=DEFAULT_VISION_CONFIG)


def test_clean_surface_no_defects(inspector):
    clean = generate_brushed_metal_surface(width=256, height=256, seed=42)
    res = inspector.inspect(clean)
    assert res.has_defect is False
    assert res.defect_severity == "NORMAL"
    assert res.defect_count == 0


def test_scratch_defect_detected(inspector):
    clean = generate_brushed_metal_surface(width=256, height=256, seed=42)
    defective, _ = inject_surface_defect(clean, defect_type="scratch", seed=42)

    res = inspector.inspect(defective)
    assert res.has_defect is True
    assert res.fft_spectral_energy > 0.0
    assert res.primary_defect_type == "Scratch"
    assert res.bounding_boxes[0].aspect_ratio >= 2.0


def test_pit_defect_detected_with_circularity(inspector):
    clean = generate_brushed_metal_surface(width=256, height=256, seed=42)
    defective, _ = inject_surface_defect(clean, defect_type="pit", seed=42)

    res = inspector.inspect(defective)
    assert res.has_defect is True
    assert res.primary_defect_type == "Pit/Pinhole"
    assert res.bounding_boxes[0].circularity > 0.3


def test_spc_western_electric_out_of_control(inspector):
    clean = generate_brushed_metal_surface(width=256, height=256, seed=42)
    defective, _ = inject_surface_defect(clean, defect_type="crack", seed=42)
    for _ in range(3):
        inspector.inspect(defective)

    spc = inspector.spc_monitor.evaluate_spc()
    assert spc["samples_in_window"] >= 3
    assert spc["process_in_control"] is False
    assert len(spc["violated_rules"]) > 0


def test_latency_is_under_50ms(inspector):
    clean = generate_brushed_metal_surface(width=256, height=256, seed=42)
    res = inspector.inspect(clean)
    assert res.latency_ms < 50.0
