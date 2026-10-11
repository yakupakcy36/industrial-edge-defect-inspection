"""Synthetic industrial surface texture and defect image generator."""

import numpy as np
from scipy import ndimage
from typing import Tuple


def generate_brushed_metal_surface(
    width: int = 256,
    height: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Generates nominal brushed metallic surface texture with directional grain."""
    rng = np.random.default_rng(seed)
    
    # Base metallic gray illumination
    base = rng.normal(loc=128.0, scale=12.0, size=(height, width))
    
    # Anisotropic directional grain (scratched rolling lines along x-axis)
    grain = rng.normal(loc=0.0, scale=8.0, size=(height, 1))
    grain_expanded = np.repeat(grain, width, axis=1)
    
    # Smooth slightly to simulate industrial cold-rolled texture
    surface = ndimage.gaussian_filter(base + grain_expanded, sigma=(1.0, 3.0))
    return np.clip(surface, 0.0, 255.0).astype(np.uint8)


def inject_surface_defect(
    surface: np.ndarray,
    defect_type: str = "scratch",
    seed: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Injects realistic manufacturing defects onto the metallic surface.
    Supported types: 'scratch', 'pit', 'crack', 'normal'
    """
    rng = np.random.default_rng(seed)
    img = surface.copy().astype(float)
    h, w = img.shape

    if defect_type.lower() == "scratch":
        # Draw linear scratch with characteristic high aspect ratio
        y_start = rng.integers(low=30, high=h - 40)
        x_start = rng.integers(low=30, high=w - 100)
        length = rng.integers(low=60, high=90)
        
        for i in range(length):
            py = int(y_start + i * 0.12)
            px = int(x_start + i)
            if 0 <= py < h and 0 <= px < w:
                img[py, px] = max(0.0, img[py, px] - 90.0)
                if py + 1 < h:
                    img[py + 1, px] = max(0.0, img[py + 1, px] - 60.0)

    elif defect_type.lower() == "pit":
        # Circular pinhole indentation
        cy = rng.integers(low=50, high=h - 50)
        cx = rng.integers(low=50, high=w - 50)
        radius = rng.integers(low=6, high=14)
        
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        mask = dist_from_center <= radius
        img[mask] = np.clip(img[mask] - 110.0, 0.0, 255.0)

    elif defect_type.lower() == "crack":
        # Jagged branching fracture line
        cy = rng.integers(low=40, high=h - 80)
        cx = rng.integers(low=40, high=w - 80)
        curr_y, curr_x = cy, cx
        for _ in range(60):
            curr_y += rng.choice([-1, 0, 1, 2])
            curr_x += rng.choice([1, 2])
            if 0 <= curr_y < h and 0 <= curr_x < w:
                img[curr_y, curr_x] = max(0.0, img[curr_y, curr_x] - 120.0)

    output = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return output, defect_type.capitalize()
