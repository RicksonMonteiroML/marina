from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


class CooccurrenceClustering:
    """
    Performs hierarchical clustering on classes based on
    Phi coefficient similarity matrix.

    Distance definition:
        d(i, j) = 1 - phi(i, j)

    This preserves the sign of the association:
        - Strong positive association → small distance
        - No association → medium distance
        - Strong negative association → large distance
    """

    def compute(
        self,
        images: List[Dict[str, Any]],
        annotations: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
        linkage_method: str = "average",
    ) -> Dict[str, Any]:

        if not images or not categories:
            return {}

        total_images = len(images)

        image_to_classes = defaultdict(set)
        for ann in annotations:
            image_to_classes[ann["image_id"]].add(ann["category_id"])

        class_ids = [c["id"] for c in categories]
        id_to_name = {c["id"]: c["name"] for c in categories}

        n = len(class_ids)

        # -------------------------------------------------------
        # Frequency and joint occurrence
        # -------------------------------------------------------

        freq = {cid: 0 for cid in class_ids}
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

        # -------------------------------------------------------
        # Build Phi matrix
        # -------------------------------------------------------

        phi_matrix = np.zeros((n, n))

        for i, ci in enumerate(class_ids):
            for j, cj in enumerate(class_ids):

                a = joint[ci][cj]
                b = freq[ci] - a
                c = freq[cj] - a
                d = total_images - (a + b + c)

                denominator = np.sqrt(
                    (a + b) * (c + d) * (a + c) * (b + d)
                )

                phi = (
                    (a * d - b * c) / denominator
                    if denominator > 0 else 0.0
                )

                phi_matrix[i, j] = phi

        # -------------------------------------------------------
        # Convert similarity to distance
        # -------------------------------------------------------

        # Preserve sign of phi
        distance_matrix = 1.0 - phi_matrix

        # Ensure perfect zeros on diagonal
        np.fill_diagonal(distance_matrix, 0.0)

        # Convert to condensed form
        condensed = squareform(distance_matrix)

        # -------------------------------------------------------
        # Hierarchical clustering
        # -------------------------------------------------------

        linkage_matrix = linkage(condensed, method=linkage_method)

        return {
            "phi_matrix": {
                id_to_name[class_ids[i]]: {
                    id_to_name[class_ids[j]]: float(phi_matrix[i, j])
                    for j in range(n)
                }
                for i in range(n)
            },
            "distance_matrix": distance_matrix.tolist(),
            "linkage_matrix": linkage_matrix.tolist(),
            "labels": [id_to_name[cid] for cid in class_ids],
        }

