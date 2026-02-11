import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from .base_plot import BasePlot


class BBoxDiagnosticsPlots(BasePlot):

    def plot_pairplot(
        self,
        raw_data: Dict[str, list],
        save_path: str | None = None,
    ) -> None:

        import pandas as pd

        df = pd.DataFrame(raw_data)

        sns.pairplot(
            df[["x", "y", "width", "height"]],
            diag_kind="hist",
            plot_kws={"alpha": 0.4, "s": 10},
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    # -------------------------------------------------------

    def plot_spatial_heatmap(
        self,
        x: list,
        y: list,
        save_path: str | None = None,
    ) -> None:

        self._setup_figure(figsize=(6, 6))

        plt.hist2d(x, y, bins=50)
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Bounding Box Spatial Distribution")

        self._finalize(save_path)

    # -------------------------------------------------------

    def plot_width_height_density(
        self,
        width: list,
        height: list,
        save_path: str | None = None,
    ) -> None:

        self._setup_figure(figsize=(6, 6))

        plt.hist2d(width, height, bins=50)
        plt.colorbar()
        plt.xlabel("width")
        plt.ylabel("height")
        plt.title("Width vs Height Density")

        self._finalize(save_path)
