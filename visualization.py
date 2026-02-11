"""
visualize_dataset.py
Entry point for running VisualizationPipeline.
"""

import argparse
from pathlib import Path

from src.pipeline.visualization_pipeline import VisualizationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Dataset statistics and visualization orchestrator"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML visualization configuration"
    )

    args = parser.parse_args()

    pipeline = VisualizationPipeline(config_path=Path(args.config))
    pipeline.run()


if __name__ == "__main__":
    main()
