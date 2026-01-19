# scieval/dataset/frontierscience.py
import os
import json
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from ..smp import load, dump, gpt_key_set  # SciEvalKit utils
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich
from .text_base import TextBaseDataset
from .utils.judge_util import build_judge


# ----------------------------
# HF helpers
# ----------------------------
def _hf_token() -> Optional[str]:
    # Optional: allow running without huggingface-cli login
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _load_jsonl_from_hf(repo_id: str, filename: str) -> List[Dict[str, Any]]:
    """
    Download a jsonl file from a Hugging Face *dataset* repo and return list[dict].
    """
    fp = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",  # IMPORTANT: this is a dataset repo
        filename=filename,
        token=_hf_token(),  # optional
    )
    rows: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _ensure_columns(df: pd.DataFrame, *, dataset_tag: str) -> pd.DataFrame:
    """
    Map FrontierScience fields -> SciEvalKit TextBaseDataset convention.
    Require: index (str), question (str), answer (str).
    """
    # question
    if "question" not in df.columns:
        if "problem" in df.columns:
            df["question"] = df["problem"]
        elif "prompt" in df.columns:
            df["question"] = df["prompt"]
        else:
            raise KeyError(
                f"[{dataset_tag}] Missing question field. Expected one of "
                f"question/problem/prompt. Got columns={list(df.columns)}"
            )

    # answer
    if "answer" not in df.columns:
        for cand in ["final_answer", "gold", "solution", "target", "reference", "rubric"]:
            if cand in df.columns:
                df["answer"] = df[cand]
                break
    if "answer" not in df.columns:
        raise KeyError(
            f"[{dataset_tag}] Missing answer field. Expected answer or one of "
            f"final_answer/gold/solution/target/reference/rubric. Got columns={list(df.columns)}"
        )

    # index
    if "index" not in df.columns:
        df["index"] = [str(i) for i in range(len(df))]
    else:
        df["index"] = df["index"].astype(str)

    # normalize to string
    df["question"] = df["question"].astype(str)
    df["answer"] = df["answer"].astype(str)

    return df


# ----------------------------
# Deterministic matching (Olympiad)
# ----------------------------
_RE_FINAL = re.compile(r"(?:FINAL\s*ANSWER|Final\s*Answer|final\s*answer)\s*:?\s*(.*)$", re.IGNORECASE | re.MULTILINE)



def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    last = None
    for m in _RE_FINAL.finditer(text):
        last = m
    if last:
        ans = (last.group(2) or "").strip()
        if ans:
            return ans
    # fallback: last non-empty line
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _strip_latex_wrappers(s: str) -> str:
    s = s.strip()
    # remove common wrappers like \( \), \[ \], $ $
    s = re.sub(r"^\\\(|\\\)$", "", s)
    s = re.sub(r"^\\\[|\\\]$", "", s)
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    return s


def _norm_text(s: str) -> str:
    s = _strip_latex_wrappers(str(s or ""))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    # normalize unicode minus
    s = s.replace("−", "-")
    return s


def _try_parse_number(s: str) -> Optional[float]:
    """
    Extract the first float-like number from a string.
    """
    s = _norm_text(s)
    m = re.search(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _numeric_equal(a: str, b: str, *, rtol: float = 1e-3, atol: float = 1e-6) -> bool:
    fa = _try_parse_number(a)
    fb = _try_parse_number(b)
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= (atol + rtol * max(1.0, abs(fb)))


def _deterministic_olympiad_hit(pred: str, gold: str) -> Tuple[int, str]:
    """
    Return (hit, normalized_pred) for olympiad by deterministic matching.
    """
    p = _norm_text(_extract_final_answer(pred))
    g = _norm_text(gold)

    if p == g:
        return 1, p

    # numeric tolerance
    if _numeric_equal(p, g):
        return 1, p

    # relaxed: remove spaces/punct for short answers
    p2 = re.sub(r"[\s\.,;:()\[\]{}]", "", p).lower()
    g2 = re.sub(r"[\s\.,;:()\[\]{}]", "", g).lower()
    if p2 == g2:
        return 1, p

    return 0, p


def _simple_report(df: pd.DataFrame, *, metric_name: str) -> pd.DataFrame:
    """
    Return a small, consistent score table.
    """
    if len(df) == 0:
        return pd.DataFrame({"split": ["none"], metric_name: [0.0]})
    return pd.DataFrame({"split": ["none"], metric_name: [float(df[metric_name].mean())]})


# ----------------------------
# LLM judge helpers (Research + optional Olympiad fallback)
# ----------------------------
def _build_grader_from_kwargs(judge_kwargs: Dict[str, Any]):
    """
    judge_kwargs should include model=... etc.
    Return grader object or None.
    """
    model_name = judge_kwargs.get("model", "exact_matching")
    if model_name == "exact_matching":
        return None

    if not gpt_key_set():
        warnings.warn("OPENAI_API_KEY (or equivalent) is not set; fallback to non-LLM evaluation.")
        return None

    grader = build_judge(**judge_kwargs)
    if not grader.working():
        warnings.warn("Judge model is not working; fallback to non-LLM evaluation.")
        return None
    return grader


def _judge_score_0_10(grader, question: str, rubric_or_answer: str, prediction: str) -> Dict[str, Any]:
    """
    Ask judge to output an integer score 0..10.
    """
    prompt = f"""
You are a strict grader. Score the model prediction from 0 to 10 (integer only).
10 means fully correct and satisfies the rubric/ground truth.
0 means completely incorrect.

Question:
{question}

Rubric / Ground Truth:
{rubric_or_answer}

Model Prediction:
{prediction}

Rules:
- Focus on the final answer; ignore chain-of-thought.
- Follow the rubric as written.
- Output ONLY a single integer from 0 to 10.
"""
    retry = 3
    for i in range(retry):
        out = str(grader.generate(prompt, temperature=0.3 * i)).strip()
        m = re.search(r"\b([0-9]|10)\b", out)
        if m:
            return {"score": int(m.group(1)), "log": out}
    return {"score": 0, "log": "judge_fail"}


def _judge_binary_correct(grader, question: str, gold: str, prediction: str) -> Dict[str, Any]:
    """
    Ask judge to output 1(correct) or 0(incorrect).
    """
    prompt = f"""
You are a strict evaluator. Decide whether the model prediction is correct.

Question:
{question}

Ground Truth Answer:
{gold}

Model Prediction:
{prediction}

Rules:
- Focus on final answer; ignore chain-of-thought.
- Treat minor formatting differences as acceptable.
- Output ONLY:
1  (if correct)
0  (if incorrect)
"""
    retry = 3
    for i in range(retry):
        out = str(grader.generate(prompt, temperature=0.3 * i)).strip()
        m = re.search(r"\b([01])\b", out)
        if m:
            return {"hit": int(m.group(1)), "log": out}
    return {"hit": 0, "log": "judge_fail"}


def _auxeval_research(tup):
    grader, item = tup
    return _judge_score_0_10(grader, item["question"], item["answer"], item["prediction"])


def _auxeval_olympiad_judge(tup):
    grader, item = tup
    return _judge_binary_correct(grader, item["question"], item["answer"], item["prediction"])


# ----------------------------
# Datasets
# ----------------------------
class FrontierScience_Olympiad(TextBaseDataset):
    """
    FrontierScience (Olympiad split) as a TEXT dataset.
    Source: openai/frontierscience, olympiad/test.jsonl
    """
    MODALITY = "TEXT"
    TYPE = "OPEN"

    HF_REPO = "openai/frontierscience"
    HF_FILE = "olympiad/test.jsonl"

    @classmethod
    def supported_datasets(cls):
        return ["FrontierScience_Olympiad"]

    def load_data(self, dataset):
        rows = _load_jsonl_from_hf(self.HF_REPO, self.HF_FILE)
        df = pd.DataFrame(rows)
        df = _ensure_columns(df, dataset_tag="FrontierScience_Olympiad")
        return df

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Output a one-row dataframe with:
          - acc: deterministic accuracy (or judge accuracy if judge model is provided)
        """
        pred_df = load(eval_file)
        assert "index" in pred_df and "prediction" in pred_df, "eval_file must contain index and prediction"

        # fill gold answer from meta if missing
        meta = self.data
        meta_ans = {str(i): a for i, a in zip(meta["index"], meta["answer"])}
        meta_q = {str(i): q for i, q in zip(meta["index"], meta["question"])}

        pred_df["index"] = pred_df["index"].astype(str)
        if "answer" not in pred_df:
            pred_df["answer"] = [meta_ans.get(i, "") for i in pred_df["index"]]
        if "question" not in pred_df:
            pred_df["question"] = [meta_q.get(i, "") for i in pred_df["index"]]
        pred_df["prediction"] = pred_df["prediction"].astype(str)
        pred_df["answer"] = pred_df["answer"].astype(str)

        # optional judge
        nproc = int(judge_kwargs.pop("nproc", 4))
        grader = _build_grader_from_kwargs(judge_kwargs.copy())

        storage = get_intermediate_file_path(eval_file, "_result", "pkl")

        if grader is None:
            hits = []
            norm_preds = []
            for _, r in pred_df.iterrows():
                hit, npred = _deterministic_olympiad_hit(r["prediction"], r["answer"])
                hits.append(hit)
                norm_preds.append(npred)
            pred_df["hit"] = hits
            pred_df["norm_pred"] = norm_preds
            dump(pred_df, storage)
        else:
            # judge binary correctness
            tmp_file = get_intermediate_file_path(eval_file, "_judge_tmp", "pkl")
            ans_map = {} if not os.path.exists(tmp_file) else load(tmp_file)

            items = []
            keys = []
            for _, r in pred_df.iterrows():
                k = str(r["index"])
                if k in ans_map:
                    continue
                keys.append(k)
                items.append({"question": r["question"], "answer": r["answer"], "prediction": r["prediction"]})

            if len(items):
                res = track_progress_rich(
                    _auxeval_olympiad_judge,
                    [(grader, it) for it in items],
                    nproc=nproc,
                    chunksize=nproc,
                    keys=keys,
                    save=tmp_file,
                )
                for k, v in zip(keys, res):
                    ans_map[k] = v

            pred_df["hit"] = [int(ans_map[str(i)]["hit"]) for i in pred_df["index"]]
            pred_df["log"] = [str(ans_map[str(i)].get("log", "")) for i in pred_df["index"]]
            dump(pred_df, storage)

        pred_df = load(storage)
        acc = float(np.mean(pred_df["hit"])) * 100.0
        score = pd.DataFrame({"split": ["none"], "acc": [acc]})

        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        dump(score, score_file)
        return score


class FrontierScience_Research(TextBaseDataset):
    """
    FrontierScience (Research split) as a TEXT dataset.
    Source: openai/frontierscience, research/test.jsonl

    Evaluation:
      - LLM grader outputs integer 0..10 using the rubric text in `answer`
      - pass@7: score >= 7 treated as correct (per FrontierScience description)
    """
    MODALITY = "TEXT"
    TYPE = "OPEN"

    HF_REPO = "openai/frontierscience"
    HF_FILE = "research/test.jsonl"

    @classmethod
    def supported_datasets(cls):
        return ["FrontierScience_Research"]

    def load_data(self, dataset):
        rows = _load_jsonl_from_hf(self.HF_REPO, self.HF_FILE)
        df = pd.DataFrame(rows)
        df = _ensure_columns(df, dataset_tag="FrontierScience_Research")
        return df

    def evaluate(self, eval_file, **judge_kwargs):
        pred_df = load(eval_file)
        assert "index" in pred_df and "prediction" in pred_df, "eval_file must contain index and prediction"

        # fill gold rubric/answer + question from meta if missing
        meta = self.data
        meta_ans = {str(i): a for i, a in zip(meta["index"], meta["answer"])}
        meta_q = {str(i): q for i, q in zip(meta["index"], meta["question"])}

        pred_df["index"] = pred_df["index"].astype(str)
        if "answer" not in pred_df:
            pred_df["answer"] = [meta_ans.get(i, "") for i in pred_df["index"]]
        if "question" not in pred_df:
            pred_df["question"] = [meta_q.get(i, "") for i in pred_df["index"]]
        pred_df["prediction"] = pred_df["prediction"].astype(str)
        pred_df["answer"] = pred_df["answer"].astype(str)

        nproc = int(judge_kwargs.pop("nproc", 4))
        grader = _build_grader_from_kwargs(judge_kwargs.copy())

        if grader is None:
            warnings.warn(
                "FrontierScience_Research needs an LLM grader (0..10 rubric). "
                "Set OPENAI_API_KEY (or equivalent) and pass judge_kwargs model=... . "
                "Returning zero scores."
            )
            score = pd.DataFrame({"split": ["none"], "avg_score": [0.0], "pass@7": [0.0]})
            score_file = get_intermediate_file_path(eval_file, "_score", "csv")
            dump(score, score_file)
            return score

        storage = get_intermediate_file_path(eval_file, "_judge", "pkl")
        tmp_file = get_intermediate_file_path(eval_file, "_judge_tmp", "pkl")
        ans_map = {} if not os.path.exists(tmp_file) else load(tmp_file)

        items = []
        keys = []
        for _, r in pred_df.iterrows():
            k = str(r["index"])
            if k in ans_map:
                continue
            keys.append(k)
            items.append({"question": r["question"], "answer": r["answer"], "prediction": r["prediction"]})

        if len(items):
            res = track_progress_rich(
                _auxeval_research,
                [(grader, it) for it in items],
                nproc=nproc,
                chunksize=nproc,
                keys=keys,
                save=tmp_file,
            )
            for k, v in zip(keys, res):
                ans_map[k] = v

        pred_df["score"] = [int(ans_map[str(i)]["score"]) for i in pred_df["index"]]
        pred_df["pass@7"] = [int(s >= 7) for s in pred_df["score"]]
        pred_df["log"] = [str(ans_map[str(i)].get("log", "")) for i in pred_df["index"]]

        dump(pred_df, storage)

        pred_df = load(storage)
        avg_score = float(np.mean(pred_df["score"]))  # 0..10
        pass7 = float(np.mean(pred_df["pass@7"])) * 100.0

        score = pd.DataFrame(
            {"split": ["none"], "avg_score": [avg_score], "pass@7": [pass7]}
        )

        score_file = get_intermediate_file_path(eval_file, "_score", "csv")
        dump(score, score_file)
        return score
