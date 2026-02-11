from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict


class CooccurrenceProbabilities:
    """
    Computes image-level co-occurrence probabilities.

    Metrics:
        - P(A)
        - P(B)
        - P(A,B)
        - P(B|A)

    All probabilities are computed at image-level (presence/absence).
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

        # image_id -> set(category_id)
        image_to_classes = defaultdict(set)
        for ann in annotations:
            image_to_classes[ann["image_id"]].add(ann["category_id"])

        class_ids = [c["id"] for c in categories]
        id_to_name = {c["id"]: c["name"] for c in categories}

        # Presence count per class
        freq = {cid: 0 for cid in class_ids}

        # Joint presence count
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

            p_i = freq[ci] / total_images if total_images > 0 else 0.0

            for cj in class_ids:
                name_j = id_to_name[cj]

                p_j = freq[cj] / total_images if total_images > 0 else 0.0
                p_ij = joint[ci][cj] / total_images if total_images > 0 else 0.0

                p_j_given_i = p_ij / p_i if p_i > 0 else 0.0

                results[name_i][name_j] = {
                    "P_i": p_i,
                    "P_j": p_j,
                    "P_ij": p_ij,
                    "P_j_given_i": p_j_given_i,
                }

        return results
