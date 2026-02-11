from typing import Dict
import matplotlib.pyplot as plt
from .base_plot import BasePlot


class DistributionPlots(BasePlot):

    def plot_class_frequency(
        self,
        distribution: Dict[str, int],
        save_path: str | None = None,
        log_scale: bool = False,
    ) -> None:

        self._setup_figure()

        classes = list(distribution.keys())
        values = list(distribution.values())

        plt.bar(classes, values)

        if log_scale:
            plt.yscale("log")

        plt.xticks(rotation=90)
        plt.ylabel("Number of Instances")
        plt.title("Class Frequency Distribution")

        self._finalize(save_path)
