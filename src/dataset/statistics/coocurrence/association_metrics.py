from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
import math


class CooccurrenceAssociationMetrics:
    """
    Computes statistical association metrics between classes.

    Metrics:
        - Lift
        - PMI
        - Phi coefficient
        - Chi-square statistic

    Computed at image-level (binary presence).
    """

    def compute(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not images or not categories:
            return {}

        total_images = len(images)

        image_to_classes = defaultdict(set)
        for ann in annotations:
            image_to_classes[ann["image_id"]].add(ann["category_id"])

        class_ids = [c["id"] for c in categories]
        id_to_name = {c["id"]: c["name"] for c in categories}

        # Presence count
        freq = {cid: 0 for cid in class_ids}

        # Joint presence
        joint = {
            cid: {cid2: 0 for cid2 in class_ids}
            for cid in class_ids
        }

        for class_set in image_to_classes.values():

            for cid in class_set:
                freq[cid] += 1

            for ci in class_set:
                for cj in class_set:
                    joint[ci][cj] += 1

        results = {}

        for ci in class_ids:
            name_i = id_to_name[ci]
            results[name_i] = {}

            for cj in class_ids:
                name_j = id_to_name[cj]

                a = joint[ci][cj]                      # both
                b = freq[ci] - a                       # ci only
                c = freq[cj] - a                       # cj only
                d = total_images - (a + b + c)         # neither

                # Probabilities
                p_i = freq[ci] / total_images
                p_j = freq[cj] / total_images
                p_ij = a / total_images if total_images > 0 else 0.0

                # Lift
                lift = (
                    p_ij / (p_i * p_j)
                    if p_i > 0 and p_j > 0 else 0.0
                )

                # PMI
                pmi = (
                    math.log2(p_ij / (p_i * p_j))
                    if p_ij > 0 and p_i > 0 and p_j > 0 else 0.0
                )

                # Phi coefficient
                denominator = math.sqrt(
                    (a + b) * (c + d) * (a + c) * (b + d)
                )

                phi = (
                    (a * d - b * c) / denominator
                    if denominator > 0 else 0.0
                )

                # Chi-square (1 degree of freedom)
                chi_square = (
                    total_images * (a * d - b * c) ** 2
                    / ((a + b) * (c + d) * (a + c) * (b + d))
                    if denominator > 0 else 0.0
                )

                results[name_i][name_j] = {
                    "lift": lift,
                    "pmi": pmi,
                    "phi": phi,
                    "chi_square": chi_square,
                }

        return results
