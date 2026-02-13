from typing import Dict
import matplotlib.pyplot as plt
from .base_plot import BasePlot


class DistributionPlots(BasePlot):

    def plot_class_frequency(
        self,
        distribution: Dict[str, int],
        pallette,
        save_path: str | None = None,
        log_scale: bool = False,
        sort: bool = True,
    ) -> None:

        self._setup_figure()

        if sort:
            distribution = dict(
                sorted(
                    distribution.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )

        classes = list(distribution.keys())
        values = list(distribution.values())

        colors = [pallette.get(label) for label in classes]
        plt.bar(classes, values, color=colors)

        if log_scale:
            plt.yscale("log")

        plt.xticks(rotation=90)
        plt.ylabel("Number of Instances")
        plt.title("Class Frequency Distribution")

        self._finalize(save_path)
