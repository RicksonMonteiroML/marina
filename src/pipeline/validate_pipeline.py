from pathlib import Path
import json
import numpy as np
from typing import Dict, List


class ValidatorPipeline:
    """
    Pipeline responsável por validar um experimento realizando:
    """

    def __init__(self, experiment_path: Path, monitor: str = "map"):
        self.experiment_path = experiment_path
        self.monitor = monitor
        self.folds_dir = experiment_path / "folds"
        self.results = {}

    def run(self):
        if not self.folds_dir.exists():
            raise FileNotFoundError(f"Pasta de folds não encontrada: {self.folds_dir}")

        print(f"[INFO] Iniciando validação em: {self.experiment_path}")

        fold_metrics = self._iterate_folds()
        agg_metrics = self._aggregate_metrics(fold_metrics)

        self.results = {
            "folds": fold_metrics,
            "aggregated": agg_metrics
        }

        return self.results

    def _load_metrics(self, metrics_file: Path) -> Dict[str, List[float]]:
        with open(metrics_file, "r") as f:
            return json.load(f)

    def _iterate_folds(self) -> Dict[str, Dict[str, float]]:
        fold_results = {}
        sorted_folds = sorted(self.folds_dir.iterdir())

        for fold_id, fold_path in enumerate(sorted_folds):
            metrics_path = fold_path / "metrics.json"

            if not metrics_path.exists():
                raise FileNotFoundError(f"Métricas não encontradas em: {metrics_path}")

            metrics_dict = self._load_metrics(metrics_path)
            best_metrics = self._extract_best_epoch_metrics(metrics_dict)

            fold_name = f"fold_{fold_id}"
            fold_results[fold_name] = best_metrics

            print(f"[OK] {fold_name}: melhor época = {best_metrics['best_epoch']}")

        return fold_results

    def _extract_best_epoch_metrics(self, metrics_dict: Dict[str, List[float]]) -> Dict[str, float]:

        if self.monitor not in metrics_dict:
            raise KeyError(f"Métrica '{self.monitor}' não encontrada no JSON de métricas.")

        monitor_curve = metrics_dict[self.monitor]
        best_epoch = int(np.argmax(monitor_curve))

        best_metrics = {
            metric_name: float(values[best_epoch])
            for metric_name, values in metrics_dict.items()
        }

        best_metrics["best_epoch"] = best_epoch
        return best_metrics

    def _aggregate_metrics(self, fold_results: Dict[str, Dict[str, float]]):
        print("\n[INFO] Calculando métricas agregadas (mean/std)...")

        # Pega lista de métricas disponíveis (ex: train_loss, map, precision, recall, lr)
        metric_names = list(next(iter(fold_results.values())).keys())

        aggregated = {}

        for metric in metric_names:
            if metric == "best_epoch":
                continue  # não faz sentido agregar épocas

            values = np.array([fold_results[f][metric] for f in fold_results])

            aggregated[metric] = {
                "mean": float(values.mean()),
                "std":  float(values.std(ddof=1))
            }

            print(f" - {metric}: mean={aggregated[metric]['mean']:.4f}, std={aggregated[metric]['std']:.4f}")

        return aggregated

    def save_report(self):
        if not self.results:
            raise RuntimeError("Nenhum resultado disponível. Execute .run() antes de salvar.")

        report_dir = self.experiment_path / "results"
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / "validation_report.json"
        print(report_path)
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=4)

        print(f"[INFO] Relatório salvo em: {report_path}")
