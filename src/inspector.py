"""Industrial edge computer vision defect inspection with 2D FFT Spectral Residual & SSIM Fusion."""

import math
import time
from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

from src.config import DEFAULT_VISION_CONFIG, VisionConfig
from src.generator import generate_brushed_metal_surface
from src.spc import SPCProcessMonitor


@dataclass
class DefectBoundingBox:
    box_id: int
    y_min: int
    x_min: int
    y_max: int
    x_max: int
    area_pixels: int
    aspect_ratio: float
    circularity: float
    defect_type: str


@dataclass
class InspectionResult:
    has_defect: bool
    defect_severity: str  # NORMAL, MINOR, CRITICAL
    anomaly_score: float  # [0.0 - 1.0]
    fft_spectral_energy: float
    defect_count: int
    primary_defect_type: str
    bounding_boxes: List[DefectBoundingBox]
    latency_ms: float


class EdgeDefectInspector:
    """
    Ultra-low latency edge vision inspector fusing:
    1. Structural Similarity (SSIM) residual mapping
    2. Multi-scale Sobel edge gradient energy
    3. 2D Fast Fourier Transform (FFT) High-Pass periodic grain suppression
    4. Connected-component geometric & circularity classification
    """

    def __init__(self, config: VisionConfig = DEFAULT_VISION_CONFIG):
        self.config = config
        self.reference_surface = generate_brushed_metal_surface(
            width=config.image_width,
            height=config.image_height,
            seed=42,
        )
        self.spc_monitor = SPCProcessMonitor()

    def calibrate(self, reference_image: np.ndarray):
        if reference_image.shape != (self.config.image_height, self.config.image_width):
            raise ValueError(f"Reference shape must be {(self.config.image_height, self.config.image_width)}")
        self.reference_surface = reference_image.copy()

    @staticmethod
    def _compute_fft_spectral_residual(gray: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Compute 2D FFT difference map and attenuate low-frequency background illumination
        to isolate high-frequency non-periodic surface anomalies.
        """
        diff = gray.astype(float) - ref.astype(float)
        f_transform = np.fft.fft2(diff)
        f_shift = np.fft.fftshift(f_transform)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Suppress DC & ultra-low frequency illumination component (center 5x5)
        f_shift[cy - 2 : cy + 3, cx - 2 : cx + 3] *= 0.1

        reconstructed = np.abs(np.fft.ifft2(np.fft.ifftshift(f_shift)))
        max_val = float(np.max(reconstructed))
        if max_val < 1e-5:
            return np.zeros_like(reconstructed)
        return np.clip(reconstructed / max(60.0, max_val), 0.0, 1.0)

    def inspect(self, image: np.ndarray) -> InspectionResult:
        start = time.perf_counter()

        if image.ndim == 3:
            gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        # 1. Structural Similarity Anomaly Map
        _, ssim_map = ssim(
            self.reference_surface,
            gray,
            win_size=self.config.ssim_win_size,
            full=True,
            data_range=255,
        )
        ssim_diff = 1.0 - ssim_map

        # 2. Gradient Energy (Sobel operators)
        sx = ndimage.sobel(gray.astype(float), axis=1)
        sy = ndimage.sobel(gray.astype(float), axis=0)
        grad_mag = np.hypot(sx, sy)
        grad_norm = np.clip(grad_mag / (np.max(grad_mag) + 1e-5), 0.0, 1.0)

        # 3. 2D FFT High-Pass Spectral Residual Map
        fft_map = self._compute_fft_spectral_residual(gray, self.reference_surface)
        fft_energy = float(np.percentile(fft_map, 99.0))

        # 4. Tri-Channel Fused Anomaly Intensity Map
        fused_anomaly = 0.55 * ssim_diff + 0.30 * grad_norm + 0.15 * fft_map
        global_anomaly_score = float(np.percentile(fused_anomaly, 99.0))

        # 5. Morphological Defect Segmentation
        mask = fused_anomaly >= self.config.anomaly_threshold
        cleaned_mask = ndimage.binary_opening(mask, structure=np.ones((2, 2)))
        cleaned_mask = ndimage.binary_closing(cleaned_mask, structure=np.ones((3, 3)))

        # 6. Connected Component & Shape Descriptor Analysis
        labeled_mask, num_features = ndimage.label(cleaned_mask)
        bounding_boxes: List[DefectBoundingBox] = []

        primary_defect = "Normal"
        max_area = 0

        for feat_idx in range(1, num_features + 1):
            comp_mask = labeled_mask == feat_idx
            coords = np.argwhere(comp_mask)
            area = len(coords)

            if area < self.config.min_defect_area_pixels:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            height = max(1, y_max - y_min)
            width = max(1, x_max - x_min)
            aspect_ratio = round(float(max(height, width) / max(1, min(height, width))), 2)

            # Estimate perimeter via binary erosion boundary pixels
            eroded = ndimage.binary_erosion(comp_mask)
            boundary_pixels = max(4, int(np.sum(comp_mask ^ eroded)))
            circularity = round(
                float(min(1.0, (4.0 * math.pi * area) / (boundary_pixels ** 2))), 3
            )

            # Classify defect geometry
            if aspect_ratio >= 2.0:
                dtype = "Scratch"
            elif aspect_ratio <= 1.6 and area <= 1200:
                dtype = "Pit/Pinhole"
            else:
                dtype = "Crack / Surface Void"

            if area > max_area:
                max_area = area
                primary_defect = dtype

            bounding_boxes.append(DefectBoundingBox(
                box_id=feat_idx,
                y_min=int(y_min),
                x_min=int(x_min),
                y_max=int(y_max),
                x_max=int(x_max),
                area_pixels=int(area),
                aspect_ratio=aspect_ratio,
                circularity=circularity,
                defect_type=dtype,
            ))

        has_defect = len(bounding_boxes) > 0
        if not has_defect:
            severity = "NORMAL"
            primary_defect = "None"
        elif max_area >= self.config.critical_defect_area_pixels or primary_defect == "Crack / Surface Void":
            severity = "CRITICAL"
        else:
            severity = "MINOR"

        self.spc_monitor.record_frame(global_anomaly_score, has_defect)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return InspectionResult(
            has_defect=has_defect,
            defect_severity=severity,
            anomaly_score=round(global_anomaly_score, 4),
            fft_spectral_energy=round(fft_energy, 4),
            defect_count=len(bounding_boxes),
            primary_defect_type=primary_defect,
            bounding_boxes=bounding_boxes,
            latency_ms=round(elapsed_ms, 2),
        )
