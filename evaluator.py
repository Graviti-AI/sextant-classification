#!/usr/bin/env python3
#
# Copyright 2022 Graviti. Licensed under MIT License.
#

from typing import Any, Dict

InputJson = Dict[str, Any]

"""
InputJson has the following format:
    {
        "CLASSIFICATION": {
                "category":           <string>  -- category of the label, in English
                "attributes": {       <object>  -- attributes of the label
                    <key>: <value>,             -- <key> denotes the name of attributes
                                                -- <value> denotes their value
                    ...
                    ...
                }
            }
    }
"""


class Evaluator:
    """This class defines some calculate operations in evaluation tasks."""

    tp = 0
    categories = {}

    def evaluate_one_data(
        self, input_ground_truths: InputJson, input_detections: InputJson
    ) -> Dict[str, Any]:
        """Evaluate one image.

        Arguments:
            input_ground_truths: Ground truth boxes in one image
                whose format is like InputJson.
            input_detections: Detected boxes in the same image
                whose format is like InputJson.

        Returns:
            A dict containing evaluation on one image and each category within it.
        """
        category_truth = input_ground_truths["CLASSIFICATION"]["category"]
        category_detections = input_detections["CLASSIFICATION"]["category"]
        tp = int(category_truth == category_detections)

        self.tp += tp
        if category_truth not in self.categories:
            self.categories[category_truth] = {"TP": tp}

        self.categories[category_truth]["TP"] += tp
        return {
            "scope": 1,
            "overall": {"TP": tp},
            "categories": {category_truth: {"TP": tp}},
        }

    def get_result(self):
        """Overall evaluation.

        Returns:
            A dict containing overall evaluation on all images and all categories.
        """
        return {
            "scope": 0,
            "overall": {"TP": self.tp, "pr": {"x": [1, 0.5], "y": [1, 0.5]}},
            "categories": self.categories,
        }
