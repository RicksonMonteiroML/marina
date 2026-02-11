from __future__ import annotations
from pathlib import Path
import json
import yaml
import numpy as np

from src.dataset.statistics.dataset_statistics import DatasetStatistics
from src.dataset.visualization.distribution_plots import DistributionPlots
from src.dataset.visualization.cooccurrence_plots import CooccurrencePlots
from src.dataset.visualization.clustering_plots import ClusteringPlots
from src.dataset.visualization.bbox_diagnostics_plots import BBoxDiagnosticsPlots
from src.dataset.visualization.bbox_plots import BBoxPlots


class VisualizationPipeline:
    """
    High-level pipeline to orchestrate dataset statistics
    and visualization generation.
    """

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()

    # -----------------------------------------------------------
    # Pipeline execution
    # -----------------------------------------------------------

    def run(self):
        print("\nStarting VisualizationPipeline...\n")

        data = self._load_dataset()
        stats = self._compute_statistics(data)
        self._generate_plots(stats)

        print("\nVisualizationPipeline completed successfully!\n")

    # -----------------------------------------------------------
    # Modular steps
    # -----------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_dataset(self) -> dict:
        dataset_path = Path(self.config["dataset_path"])

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            return json.load(f)

    def _compute_statistics(self, data: dict) -> dict:
        print("Computing dataset statistics...")
        stats = DatasetStatistics().compute(data)
        print("Statistics computed.")
        return stats

    # -----------------------------------------------------------
    # Plot generation
    # -----------------------------------------------------------

    def _generate_plots(self, stats: dict):
        print("Generating plots...")

        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_cfg = self.config.get("plots", {})

        # -------------------------------------------------------
        # 1️⃣ Class Distribution
        # -------------------------------------------------------
        if plot_cfg.get("class_distribution", True):
            DistributionPlots().plot_class_frequency(
                stats["categories"]["distribution"],
                save_path=output_dir / "class_distribution.png",
            )

        # -------------------------------------------------------
        # 2️⃣ Association Heatmap (Phi)
        # -------------------------------------------------------
        if plot_cfg.get("phi_heatmap", True):

            association = stats["cooccurrence"].get("association", {})
            metric = "phi"

            if association:

                labels = list(association.keys())

                metric_matrix = {
                    row: {
                        col: association[row][col][metric]
                        for col in labels
                    }
                    for row in labels
                }

                CooccurrencePlots().plot_heatmap(
                    metric_matrix,
                    title="Phi Coefficient Heatmap",
                    save_path=output_dir / "phi_heatmap.png",
                )

        # -------------------------------------------------------
        # 3️⃣ Dendrogram
        # -------------------------------------------------------
        if plot_cfg.get("dendrogram", True):

            clustering_data = stats["cooccurrence"].get("clustering")

            if clustering_data and "linkage_matrix" in clustering_data:

                linkage_matrix = np.array(clustering_data["linkage_matrix"])

                ClusteringPlots().plot_dendrogram(
                    linkage_matrix,
                    labels=clustering_data["labels"],
                    save_path=output_dir / "dendrogram.png",
                )

        # -------------------------------------------------------
        # 4️⃣ Basic Bounding Box Plots
        # -------------------------------------------------------
        bbox_stats = stats.get("bboxes", {})
        bbox_plotter = BBoxPlots()

        if plot_cfg.get("bbox_area_distribution", True):
            areas = bbox_stats.get("areas")
            if areas:
                bbox_plotter.plot_bbox_area_distribution(
                    areas,
                    save_path=output_dir / "bbox_area_distribution.png",
                )

        if plot_cfg.get("bbox_ratio_distribution", True):
            ratios = bbox_stats.get("aspect_ratios")
            if ratios:
                bbox_plotter.plot_aspect_ratio_distribution(
                    ratios,
                    save_path=output_dir / "bbox_aspect_ratio.png",
                )

        # -------------------------------------------------------
        # 5️⃣ Advanced Bounding Box Diagnostics
        # -------------------------------------------------------
        diagnostics = BBoxDiagnosticsPlots()

        widths = bbox_stats.get("widths")
        heights = bbox_stats.get("heights")
        
        x_centers = bbox_stats.get("x_centers")
        y_centers = bbox_stats.get("y_centers")

        x_centers_norm = bbox_stats.get("x_centers_norm")
        y_centers_norm = bbox_stats.get("y_centers_norm")

        if widths and heights and x_centers and y_centers:

            if plot_cfg.get("bbox_pairplot", True):
                diagnostics.plot_pairplot(
                    {
                        "x": x_centers,
                        "y": y_centers,
                        "width": widths,
                        "height": heights,
                    },
                    save_path=output_dir / "bbox_pairplot.png"
                )

            if plot_cfg.get("bbox_spatial_heatmap", True):
                diagnostics.plot_spatial_heatmap(
                    x_centers_norm,
                    y_centers_norm,
                    save_path=output_dir / "bbox_spatial_heatmap.png"
                )

            if plot_cfg.get("bbox_wh_density", True):
                diagnostics.plot_width_height_density(
                    widths,
                    heights,
                    save_path=output_dir / "bbox_wh_density.png"
                )

        print(f"Plots saved to {output_dir}")
