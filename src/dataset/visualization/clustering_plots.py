from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from .base_plot import BasePlot


class ClusteringPlots(BasePlot):

    def plot_dendrogram(
        self,
        linkage_matrix,
        labels: list[str],
        save_path: str | None = None,
    ) -> None:

        self._setup_figure(figsize=(10, 6))

        dendrogram(linkage_matrix, labels=labels)
        plt.xticks(rotation=90)
        plt.title("Hierarchical Clustering Dendrogram")

        self._finalize(save_path)
