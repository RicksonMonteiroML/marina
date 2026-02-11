import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from .base_plot import BasePlot


class CooccurrencePlots(BasePlot):

    def plot_heatmap(
        self,
        matrix: Dict[str, Dict[str, float]],
        title: str,
        cmap: str = "viridis",
        save_path: str | None = None,
        annotate: bool = True,
        fmt: str = ".2f",
    ) -> None:

        labels = list(matrix.keys())

        data = np.array([
            [matrix[row][col] for col in labels]
            for row in labels
        ])

        self._setup_figure(figsize=(10, 8))

        im = plt.imshow(data, cmap=cmap)
        plt.colorbar(im)

        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title(title)

        # -------------------------------------------------------
        # Annotate cells with values
        # -------------------------------------------------------
        if annotate:
            threshold = (data.max() + data.min()) / 2.0

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):

                    value = data[i, j]

                    # Choose text color based on background intensity
                    color = "white" if value > threshold else "black"

                    plt.text(
                        j, i,
                        format(value, fmt),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                    )

        self._finalize(save_path)
