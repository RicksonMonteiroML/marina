from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime

from src.trainer import TrainerFactory


class TrainingOrchestratorMulti:
    """
    Runs training over a list of models.
    Each model runs a FULL K-Fold cross-validation.

    Example directory:
        experiments/
            retinanet/run_123/
            fasterrcnn/run_456/
            ssd/run_789/
    """

    def __init__(self, models_cfg, training_cfg, config_path, folds_root):
        self.models_cfg = models_cfg
        self.training_cfg = training_cfg
        self.config_path = config_path
        self.folds_root = Path(folds_root)

        # folds metadata
        self.folds_metadata = json.load(open(self.folds_root / "metadata.json"))

    # -------------------------------------------------------------------------
    def run(self):

        results = []

        for model_cfg in self.models_cfg:

            model_name = model_cfg["name"]
            print(f"\n==============================================")
            print(f"🔥 Starting Cross-Validation for MODEL: {model_name}")
            print(f"==============================================")

            # Load correct trainer
            trainer_class = TrainerFactory.get_trainer(model_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_root = Path("experiments") / model_name / f"run_{timestamp}"
            exp_root.mkdir(parents=True, exist_ok=True)

            model_results = []

            # iterate over each fold
            for fold_info in self.folds_metadata["folds"]:

                fold = fold_info["fold"]
                fold_dir = exp_root / f"fold_{fold:02d}"
                fold_dir.mkdir(exist_ok=True)

                train_json = fold_info["train_json"]
                val_json = fold_info["val_json"]

                print(f"\nTraining fold {fold} for {model_name}...")

                trainer = trainer_class(
                    training_cfg=self.training_cfg,
                    model_cfg=model_cfg,
                    fold_dir=fold_dir,
                    train_json=train_json,
                    val_json=val_json,
                )

                metrics = trainer.run()
                json.dump(metrics, open(fold_dir / "metrics.json", "w"), indent=2)

                model_results.append({
                    "fold": fold,
                    "metrics": metrics,
                })

            # model summary
            summary_path = exp_root / "summary.json"
            json.dump(model_results, open(summary_path, "w"), indent=2)

            print(f"✔ Summary for {model_name} saved at: {summary_path}")

            results.append({
                "model": model_name,
                "results": model_results,
            })

        return results
