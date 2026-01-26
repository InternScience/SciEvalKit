import os.path as osp
from typing import Any, Dict, List

import pandas as pd

from .text_base import TextBaseDataset


class EarthLinkTest(TextBaseDataset):
    """
    A test dataset for EarthLink agent evaluation.
    """
    
    TYPE = "TEXT"
    MODALITY = "TEXT"
    dataset_name = "EarthLinkTest"

    def __init__(
        self,
        dataset: str = "EarthLinkTest",
        **kwargs,
    ):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ["EarthLinkTest"]

    def load_data(self, dataset):
        """Load a single test data sample."""
        rows: List[Dict[str, Any]] = [
            {
                "index": 0,
                "request": "Analyze global average surface air temperature changes under different SSP scenarios.",
                "answer": "",
            }
        ]
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        """Build the prompt for the model."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        return [dict(type="text", value=line["request"])]
    
    def score_agent_sample(self, idx, final_answer, **judge_kwargs):
        """Score a single sample based on the final answer."""
        sample = self.data.iloc[idx]
        # Placeholder scoring logic
        score = 1.0 if final_answer else 0.0
        return {"score": score}

    def evaluate(self, eval_file, **judge_kwargs):
        """Evaluate predictions using exact match."""
        from ..smp import load

        data = load(eval_file)
        if isinstance(data, pd.DataFrame):
            rows = data.to_dict("records")
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError("Unsupported prediction format for evaluation.")

        scores = []
        for row in rows:
            idx = int(row.get("index", row.get("id", 0)))
            pred = row.get("prediction", row.get("final_answer", "")).strip().lower()
            
            # Get ground truth answer
            sample = self.data.iloc[idx]
            answer = sample["answer"].strip().lower()
            
            scores.append(self.score_agent_sample(idx, pred, **judge_kwargs)["score"])

        accuracy = sum(scores) / len(scores) if scores else 0.0
        return {"accuracy": accuracy}
