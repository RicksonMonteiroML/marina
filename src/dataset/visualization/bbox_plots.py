from __future__ import annotations
from typing import Sequence
import matplotlib.pyplot as plt

from .base_plot import BasePlot


class BBoxPlots(BasePlot):
    """
    Visualization utilities for bounding box statistics.

    Responsibilities:
        - Histogram of bounding boxes per image
        - Histogram of bounding box areas
        - Histogram of aspect ratios
    """

    # -----------------------------------------------------------
    # BBoxes per Image
    # -----------------------------------------------------------

    def plot_bboxes_per_image(
        self,
        values: Sequence[float],
        save_path=None,
        bins: int = 30,
    ) -> None:

        self._setup_figure()

        plt.hist(values, bins=bins)
        plt.title("Bounding Boxes per Image")
        plt.xlabel("Number of Bounding Boxes")
        plt.ylabel("Frequency")

        self._finalize(save_path)

    # -----------------------------------------------------------
    # Bounding Box Area
    # -----------------------------------------------------------

    def plot_bbox_area_distribution(
        self,
        areas: Sequence[float],
        save_path=None,
        bins: int = 40,
        log_scale: bool = False,
    ) -> None:

        self._setup_figure()

        plt.hist(areas, bins=bins)

        if log_scale:
            plt.xscale("log")

        plt.title("Bounding Box Area Distribution")
        plt.xlabel("Area (pixels²)")
        plt.ylabel("Frequency")

        self._finalize(save_path)

    # -----------------------------------------------------------
    # Aspect Ratio
    # -----------------------------------------------------------

    def plot_aspect_ratio_distribution(
        self,
        ratios: Sequence[float],
        save_path=None,
        bins: int = 40,
    ) -> None:

        self._setup_figure()

        plt.hist(ratios, bins=bins)

        plt.title("Bounding Box Aspect Ratio Distribution")
        plt.xlabel("Aspect Ratio (width / height)")
        plt.ylabel("Frequency")

        self._finalize(save_path)
