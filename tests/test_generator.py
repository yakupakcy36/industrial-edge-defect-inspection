"""Test suite for synthetic surface generator and defect injection."""

import numpy as np
import pytest
from src.generator import generate_brushed_metal_surface, inject_surface_defect


def test_surface_dimensions_and_dtype():
    """Verify generated surface has exact shape and uint8 dtype."""
    surface = generate_brushed_metal_surface(width=256, height=256, seed=42)
    assert surface.shape == (256, 256)
    assert surface.dtype == np.uint8
    assert 0 <= surface.min() <= surface.max() <= 255


def test_defect_injections():
    """Verify scratch, pit, and crack injections introduce pixel modifications."""
    base = generate_brushed_metal_surface(width=256, height=256, seed=42)

    for defect_type in ["scratch", "pit", "crack"]:
        defective, label = inject_surface_defect(base, defect_type=defect_type, seed=42)
        assert defective.shape == base.shape
        # Defective surface should differ from base
        assert not np.array_equal(base, defective)
