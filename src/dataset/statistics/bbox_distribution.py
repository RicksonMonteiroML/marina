from __future__ import annotations
from typing import Dict, Any, List
import numpy as np


class BBoxDistribution:
    """
    Compute statistical properties of bounding boxes.

    Includes:
        - width/height/area mean, median, std, min/max, quartiles
        - aspect ratio statistics
        - spatial distribution (absolute and normalized centers)
        - safe handling of invalid values

    Assumes COCO format bbox:
        [x_min, y_min, width, height]
    """

    def compute(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not annotations:
            return {"count": 0}

        # -----------------------------------------------------------
        # Build image_id → (width, height) map
        # -----------------------------------------------------------
        image_sizes = {
            img["id"]: (float(img["width"]), float(img["height"]))
            for img in images
        }

        widths = []
        heights = []

        areas = []
        areas_norm = []

        aspect_ratios = []
        category_ids = []

        x_centers = []
        y_centers = []

        x_centers_norm = []
        y_centers_norm = []

        widths_norm = []
        heights_norm = []

        # -----------------------------------------------------------
        # Extract bbox information
        # -----------------------------------------------------------
        for ann in annotations:
            x_min, y_min, w, h = ann["bbox"]

            w = float(w)
            h = float(h)
            x_min = float(x_min)
            y_min = float(y_min)

            img_w, img_h = image_sizes.get(ann["image_id"], (1.0, 1.0))

            # Compute centers (absolute)
            x_center = x_min + w / 2.0
            y_center = y_min + h / 2.0

            # Normalize spatial values
            x_center_n = x_center / img_w
            y_center_n = y_center / img_h
            w_n = w / img_w
            h_n = h / img_h

            # Avoid division by zero
            if h <= 0:
                h = 1e-6

            widths.append(w)
            heights.append(h)
            areas.append(max(w * h, 0.0))
            areas_norm.append(max(w_n * h_n, 0.0))
            aspect_ratios.append(w / h)
            category_ids.append(ann["category_id"])
            x_centers.append(x_center)
            y_centers.append(y_center)

            x_centers_norm.append(x_center_n)
            y_centers_norm.append(y_center_n)

            widths_norm.append(w_n)
            heights_norm.append(h_n)

        # Convert to numpy
        widths = np.asarray(widths, dtype=float)
        heights = np.asarray(heights, dtype=float)
        areas = np.asarray(areas, dtype=float)
        areas_norm = np.asarray(areas_norm, dtype=float)
        aspect_ratios = np.asarray(aspect_ratios, dtype=float)

        x_centers = np.asarray(x_centers, dtype=float)
        y_centers = np.asarray(y_centers, dtype=float)

        x_centers_norm = np.asarray(x_centers_norm, dtype=float)
        y_centers_norm = np.asarray(y_centers_norm, dtype=float)

        widths_norm = np.asarray(widths_norm, dtype=float)
        heights_norm = np.asarray(heights_norm, dtype=float)

        # -----------------------------------------------------------
        # Safe std
        # -----------------------------------------------------------
        def safe_std(arr: np.ndarray) -> float:
            return float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        def q(arr, percentile):
            return float(np.percentile(arr, percentile))

        return {
            "count": len(annotations),

            # ---------------- Width ----------------
            "width_mean": float(widths.mean()),
            "width_median": float(np.median(widths)),
            "width_std": safe_std(widths),
            "width_min": float(widths.min()),
            "width_max": float(widths.max()),
            "width_q1": q(widths, 25),
            "width_q3": q(widths, 75),

            # ---------------- Height ----------------
            "height_mean": float(heights.mean()),
            "height_median": float(np.median(heights)),
            "height_std": safe_std(heights),
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_q1": q(heights, 25),
            "height_q3": q(heights, 75),

            # ---------------- Area ----------------
            "area_mean": float(areas.mean()),
            "area_median": float(np.median(areas)),
            "area_std": safe_std(areas),
            "area_min": float(areas.min()),
            "area_max": float(areas.max()),
            "area_q1": q(areas, 25),
            "area_q3": q(areas, 75),

            # ---------------- Area (normalized) ----------------
            "area_norm_mean": float(areas_norm.mean()),
            "area_norm_median": float(np.median(areas_norm)),
            "area_norm_std": safe_std(areas_norm),
            "area_norm_min": float(areas_norm.min()),
            "area_norm_max": float(areas_norm.max()),
            "area_norm_q1": q(areas_norm, 25),
            "area_norm_q3": q(areas_norm, 75),

            # ---------------- Aspect Ratio ----------------
            "ratio_mean": float(aspect_ratios.mean()),
            "ratio_median": float(np.median(aspect_ratios)),
            "ratio_std": safe_std(aspect_ratios),
            "ratio_min": float(aspect_ratios.min()),
            "ratio_max": float(aspect_ratios.max()),
            "ratio_q1": q(aspect_ratios, 25),
            "ratio_q3": q(aspect_ratios, 75),

            # ---------------- Spatial (absolute) ----------------
            "x_center_mean": float(x_centers.mean()),
            "y_center_mean": float(y_centers.mean()),

            # ---------------- Raw values (absolute) ----------------
            "widths": widths.tolist(),
            "heights": heights.tolist(),
            "areas": areas.tolist(),
            "areas_norm": areas_norm.tolist(),
            "aspect_ratios": aspect_ratios.tolist(),
            "category_ids": category_ids,
            "x_centers": x_centers.tolist(),
            "y_centers": y_centers.tolist(),

            # ---------------- Raw values (normalized) ----------------
            "widths_norm": widths_norm.tolist(),
            "heights_norm": heights_norm.tolist(),
            "x_centers_norm": x_centers_norm.tolist(),
            "y_centers_norm": y_centers_norm.tolist(),

            "area_log": np.log1p(areas).tolist(),
        }
