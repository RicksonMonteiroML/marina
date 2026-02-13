import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from .base_plot import BasePlot
import matplotlib.patches as patches
import numpy as np
from matplotlib.colors import LogNorm


class BBoxDiagnosticsPlots(BasePlot):

    # -------------------------------------------------------
    # Pairplot (YOLO-style density)
    # -------------------------------------------------------

    def plot_pairplot(
        self,
        raw_data: Dict[str, list],
        save_path: str | None = None,
    ) -> None:

        df = pd.DataFrame(raw_data)

        sns.pairplot(
            df[["x", "y", "width", "height"]],
            kind="hist",
            diag_kind="hist",
            plot_kws={
                "bins": 50,
                "pmax": 0.6,
            }
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

    # -------------------------------------------------------
    # Spatial Heatmap
    # -------------------------------------------------------

    def plot_spatial_heatmap(
        self,
        x: list,
        y: list,
        save_path: str | None = None,
    ) -> None:

        self._setup_figure(figsize=(6, 6))

        plt.hist2d(
            x,
            y,
            bins=50,
            cmap="Blues",
            norm=LogNorm(),
        )

        plt.colorbar()
        plt.xlabel("x (normalized)")
        plt.ylabel("y (normalized)")
        plt.title("Bounding Box Spatial Distribution")

        self._finalize(save_path)

    # -------------------------------------------------------
    # Width vs Height Density
    # -------------------------------------------------------

    def plot_width_height_density(
        self,
        width: list,
        height: list,
        save_path: str | None = None,
    ) -> None:

        self._setup_figure(figsize=(6, 6))

        plt.hist2d(
            width,
            height,
            bins=50,
            cmap="Blues",
            norm=LogNorm(),
        )

        plt.colorbar()
        plt.xlabel("width (normalized)")
        plt.ylabel("height (normalized)")
        plt.title("Width vs Height Density")

        self._finalize(save_path)

    # -------------------------------------------------------
    # Centered Normalized Overlay (Centered at 0,0)
    # -------------------------------------------------------

    def plot_centered_overlay_by_class(
        self,
        widths_norm: list,
        heights_norm: list,
        class_ids: list,
        class_id_to_name: Dict[int, str],
        palette: Dict[str, tuple],
        max_samples: int = 2000,
        save_path: str | None = None,
        alpha: float = 0.6,
    ) -> None:

        widths_norm = np.array(widths_norm)
        heights_norm = np.array(heights_norm)
        class_ids = np.array(class_ids)

        N = len(widths_norm)

        # ---------------------------------------------------
        # Sampling (visual clarity)
        # ---------------------------------------------------
        if max_samples and N > max_samples:
            np.random.seed(42)
            indices = np.random.choice(N, max_samples, replace=False)

            widths_norm = widths_norm[indices]
            heights_norm = heights_norm[indices]
            class_ids = class_ids[indices]

        self._setup_figure(figsize=(6, 6))
        ax = plt.gca()

        # ---------------------------------------------------
        # Draw boxes centered at (0,0)
        # ---------------------------------------------------
        for w, h, cid in zip(widths_norm, heights_norm, class_ids):
            class_name = class_id_to_name.get(cid)
            color = palette.get(class_name, (0.5, 0.5, 0.5))

            # Centro agora é (0,0)
            x_min = -w / 2
            y_min = -h / 2

            rect = patches.Rectangle(
                (x_min, y_min),
                w,
                h,
                linewidth=0.4,
                edgecolor=color,
                facecolor="none",
                alpha=alpha,
            )

            ax.add_patch(rect)

        # ---------------------------------------------------
        # Axis configuration (centered system)
        # ---------------------------------------------------
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect("equal")

        # Linhas de referência no centro
        # ax.axhline(0, linewidth=0.6)
        # ax.axvline(0, linewidth=0.6)

        ax.set_title("Centered Normalized Bounding Boxes by Class (0,0 Centered)")
        ax.set_xlabel("x (normalized, centered)")
        ax.set_ylabel("y (normalized, centered)")

        self._finalize(save_path)
