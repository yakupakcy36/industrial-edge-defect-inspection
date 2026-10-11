"""Configuration settings for Industrial Edge Defect Inspection Engine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionConfig:
    """Master configuration for edge computer vision defect detection."""
    image_width: int = 256
    image_height: int = 256
    anomaly_threshold: float = 0.35  # SSIM distance threshold
    min_defect_area_pixels: int = 15
    critical_defect_area_pixels: int = 150
    blur_kernel_size: int = 3
    ssim_win_size: int = 7


DEFAULT_VISION_CONFIG = VisionConfig()
