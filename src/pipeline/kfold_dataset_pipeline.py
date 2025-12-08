from __future__ import annotations
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime

from src.stratification.iterative_stratification import IterativeStratification
from src.core.config import ConfigLoader


class KFoldDatasetPipeline:
    """
    Generates stratified K-fold COCO datasets (group-level stratification)
    preserving full metadata:
        - info
        - licenses
        - categories
        - images
        - annotations

    Includes full tracking for reproducibility.
    """

    def __init__(self, config_path: Path, k_folds: int = 5, seed: int = 42):
        self.k_folds = k_folds
        self.seed = seed
        self.config_loader = ConfigLoader(config_path=config_path)

        canonical_cfg = self.config_loader.load_canonical()["canonical"]
        self.output_dir = Path(canonical_cfg["output_dir"])
        self.canonical_path = self.output_dir / canonical_cfg["filename"]

        if not self.canonical_path.exists():
            raise RuntimeError(
                f"Canonical dataset NOT found: {self.canonical_path}"
            )

        # load canonical dataset
        self.canonical = json.load(open(self.canonical_path, "r"))

    # -------------------------------------------------------------------------
    def run(self):

        print("\n🚀 Starting K-Fold dataset generation...\n")

        # extract canonical elements
        images = self.canonical["images"]
        annotations = self.canonical["annotations"]
        info = self.canonical["info"]
        licenses = self.canonical["licenses"]
        categories = self.canonical["categories"]

        # ---------------------------------------------------------------------
        # BUILD structures for grouping and stratification
        # ---------------------------------------------------------------------
        img_to_labels = defaultdict(list)
        img_to_experiment = {}

        for ann in annotations:
            img_to_labels[ann["image_id"]].append(ann["category_id"])

        for img in images:
            img_to_experiment[img["id"]] = img.get("experiment_id", "default")

        # group images by experiment
        experiment_groups = defaultdict(list)
        for img_id, exp in img_to_experiment.items():
            experiment_groups[exp].append(img_id)

        group_list = list(experiment_groups.values())

        # print stats
        print(f"✔ Images: {len(images)}")
        print(f"✔ Annotations: {len(annotations)}")
        print(f"✔ Experiment groups: {len(group_list)}")

        # ---------------------------------------------------------------------
        # PREPARE STRATIFICATION INPUT
        # ---------------------------------------------------------------------
        X = []
        Y = []

        for group in group_list:
            labels = set()
            for img_id in group:
                labels.update(img_to_labels[img_id])
            Y.append(list(labels))

        splitter = IterativeStratification(
            n_splits=self.k_folds,
            order=1,
            random_state=self.seed,
        )

        folds = list(splitter.split(X, Y))

        # ---------------------------------------------------------------------
        # CREATE OUTPUT FOLDER
        # ---------------------------------------------------------------------
        folds_root = self.output_dir / "kfolds"
        folds_root.mkdir(parents=True, exist_ok=True)

        # GLOBAL METADATA
        global_meta = {
            "created_at": datetime.now().isoformat(),
            "canonical_source": str(self.canonical_path),
            "num_folds": self.k_folds,
            "folds": []
        }

        # ---------------------------------------------------------------------
        # GENERATE EACH FOLD
        # ---------------------------------------------------------------------
        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):

            print(f"\n📁 Creating Fold {fold_idx}/{self.k_folds}")

            fold_dir = folds_root / f"fold_{fold_idx:02d}"
            fold_dir.mkdir(exist_ok=True)

            # expand group indices → image IDs
            train_ids = self._expand_groups(group_list, train_idx)
            val_ids = self._expand_groups(group_list, val_idx)

            # build COCO JSONs
            fold_train = self._build_fold_json(
                images, annotations, train_ids,
                info, licenses, categories
            )

            fold_val = self._build_fold_json(
                images, annotations, val_ids,
                info, licenses, categories
            )

            # save JSONs
            train_path = fold_dir / "train.json"
            val_path = fold_dir / "val.json"

            json.dump(fold_train, open(train_path, "w"), indent=2)
            json.dump(fold_val, open(val_path, "w"), indent=2)

            print(f"   → train.json saved in {train_path}")
            print(f"   → val.json saved in {val_path}")

            # ------------------------------------------------------------------
            # TRACKING FILES FOR REPRODUCIBILITY
            # ------------------------------------------------------------------
            tracking = {
                "fold": fold_idx,
                "total_folds": self.k_folds,
                "seed": self.seed,
                "canonical_source": str(self.canonical_path),

                "train_image_ids": sorted(list(train_ids)),
                "val_image_ids": sorted(list(val_ids)),

                "train_experiments": sorted({
                    img_to_experiment[i] for i in train_ids
                }),
                "val_experiments": sorted({
                    img_to_experiment[i] for i in val_ids
                }),

                "categories": categories,
            }

            json.dump(
                tracking,
                open(fold_dir / "fold_config.json", "w"),
                indent=2
            )

            # simple text lists
            (open(fold_dir / "image_ids_train.txt", "w")
                .write("\n".join(map(str, train_ids))))

            (open(fold_dir / "image_ids_val.txt", "w")
                .write("\n".join(map(str, val_ids))))

            (open(fold_dir / "experiments_train.txt", "w")
                .write("\n".join(tracking["train_experiments"])))

            (open(fold_dir / "experiments_val.txt", "w")
                .write("\n".join(tracking["val_experiments"])))

            # register fold metadata
            global_meta["folds"].append({
                "fold": fold_idx,
                "train_json": str(train_path),
                "val_json": str(val_path),
                "config": str(fold_dir / "fold_config.json"),
            })

        # ---------------------------------------------------------------------
        # SAVE GLOBAL METADATA
        # ---------------------------------------------------------------------
        json.dump(global_meta, open(folds_root / "metadata.json", "w"), indent=2)

        print("\n🎉 K-Fold generation complete!")
        print(f"Global metadata: {folds_root / 'metadata.json'}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _expand_groups(self, group_list, idx_list):
        """Convert stratification group indexes into set of image IDs."""
        ids = []
        for idx in idx_list:
            ids.extend(group_list[idx])
        return set(ids)

    def _build_fold_json(self, images, annotations, keep_ids, info, licenses, categories):
        """Build COCO JSON for one fold, preserving metadata."""
        imgs = [img for img in images if img["id"] in keep_ids]
        anns = [ann for ann in annotations if ann["image_id"] in keep_ids]

        stats = {
            "num_images": len(imgs),
            "num_annotations": len(anns),
            "num_categories": len(categories)
        }

        return {
            "images": imgs,
            "annotations": anns,
            "categories": categories,
            "info": info,
            "licenses": licenses,
            "statistics": stats,
        }
