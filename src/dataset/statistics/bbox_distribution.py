from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


class BBoxDistribution:
    """
    Compute statistical properties of bounding boxes.

    Includes:
        - width/height/area mean, median, std, min/max, quartiles
        - aspect ratio statistics
        - spatial distribution (x_center, y_center)
        - safe handling of invalid values (n < 2, negative area)

    Assumes COCO format bbox:
        [x_min, y_min, width, height]
    """

    def compute(
        self,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not annotations:
            return {"count": 0}

        widths = []
        heights = []
        areas = []
        aspect_ratios = []
        x_centers = []
        y_centers = []

        # -----------------------------------------------------------
        # Extract width, height, area, ratio, and spatial center
        # -----------------------------------------------------------
        for ann in annotations:
            x_min, y_min, w, h = ann["bbox"]

            w = float(w)
            h = float(h)
            x_min = float(x_min)
            y_min = float(y_min)

            # Compute center coordinates
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0

            # Avoid division by zero for aspect ratio
            if h <= 0:
                h = 1e-6

            widths.append(w)
            heights.append(h)
            areas.append(max(w * h, 0.0))
            aspect_ratios.append(w / h)
            x_centers.append(x_center)
            y_centers.append(y_center)

        widths = np.asarray(widths, dtype=float)
        heights = np.asarray(heights, dtype=float)
        areas = np.asarray(areas, dtype=float)
        aspect_ratios = np.asarray(aspect_ratios, dtype=float)
        x_centers = np.asarray(x_centers, dtype=float)
        y_centers = np.asarray(y_centers, dtype=float)

        # -----------------------------------------------------------
        # Safe std function
        # -----------------------------------------------------------
        def safe_std(arr: np.ndarray) -> float:
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        # -----------------------------------------------------------
        # Helper: quantiles
        # -----------------------------------------------------------
        def q(arr, percentile):
            return float(np.percentile(arr, percentile))

        return {
            "count": len(annotations),

            # -------------------------------------------------------
            # Width statistics
            # -------------------------------------------------------
            "width_mean": float(widths.mean()),
            "width_median": float(np.median(widths)),
            "width_std": safe_std(widths),
            "width_min": float(widths.min()),
            "width_max": float(widths.max()),
            "width_q1": q(widths, 25),
            "width_q3": q(widths, 75),

            # -------------------------------------------------------
            # Height statistics
            # -------------------------------------------------------
            "height_mean": float(heights.mean()),
            "height_median": float(np.median(heights)),
            "height_std": safe_std(heights),
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_q1": q(heights, 25),
            "height_q3": q(heights, 75),

            # -------------------------------------------------------
            # Area statistics
            # -------------------------------------------------------
            "area_mean": float(areas.mean()),
            "area_median": float(np.median(areas)),
            "area_std": safe_std(areas),
            "area_min": float(areas.min()),
            "area_max": float(areas.max()),
            "area_q1": q(areas, 25),
            "area_q3": q(areas, 75),

            # -------------------------------------------------------
            # Aspect ratio statistics (w / h)
            # -------------------------------------------------------
            "ratio_mean": float(aspect_ratios.mean()),
            "ratio_median": float(np.median(aspect_ratios)),
            "ratio_std": safe_std(aspect_ratios),
            "ratio_min": float(aspect_ratios.min()),
            "ratio_max": float(aspect_ratios.max()),
            "ratio_q1": q(aspect_ratios, 25),
            "ratio_q3": q(aspect_ratios, 75),

            # -------------------------------------------------------
            # Spatial statistics
            # -------------------------------------------------------
            "x_center_mean": float(x_centers.mean()),
            "y_center_mean": float(y_centers.mean()),

            # -------------------------------------------------------
            # Raw values for visualization layer
            # -------------------------------------------------------
            "widths": widths.tolist(),
            "heights": heights.tolist(),
            "areas": areas.tolist(),
            "aspect_ratios": aspect_ratios.tolist(),
            "x_centers": x_centers.tolist(),
            "y_centers": y_centers.tolist(),
            "area_log": np.log1p(areas).tolist(),
        }
