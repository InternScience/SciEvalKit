"""
Protein GO (Gene Ontology) text dataset.

Similar to human_ppi structure:
- Data: LMUDataRoot()/go_bp.tsv, go_cc.tsv, go_mf.tsv
- Columns: index, question, answer, category
- answer: semicolon-separated GO term names
- Evaluation: exact match (normalized string comparison)
"""
from __future__ import annotations

from ..text_base import TextBaseDataset
from .utils.protein_go import evaluate_protein_go
from ...smp import *


class ProteinGODataset(TextBaseDataset):
    """
    Protein GO text dataset for GO annotation prediction evaluation.

    - Data: LMUDataRoot()/go_bp.tsv, go_cc.tsv, go_mf.tsv
      Columns: index, question, answer, category
      answer: semicolon-separated GO term names
    - Evaluation: exact match accuracy, no external judge.
    """

    TYPE = "QA"

    DATASET_URL = {"go_bp": "", "go_cc": "", "go_mf": ""}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def load_data(self, dataset):
        """
        Load from LMUDataRoot: go_bp.tsv, go_cc.tsv, go_mf.tsv
        """
        data_path = osp.join(LMUDataRoot(), f"{dataset}.tsv")
        if not osp.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} does not exist.")
        return load(data_path)

    def build_prompt(self, line):
        """Prompt is the question field."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line["question"]
        return [dict(type="text", value=question)]

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Exact match evaluation: parse prediction to GO terms, compare with answer.
        """
        _ = judge_kwargs
        return evaluate_protein_go(eval_file, self.data)
