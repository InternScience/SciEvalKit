"""
Protein GO evaluation utilities.

Supports parsing various model output formats (GPT, Qwen, SciReasoner, etc.).
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

from ....smp import *
from ....smp.file import get_intermediate_file_path

GO_ID_PATTERN = re.compile(r"GO:\d{7}")

_DELIMITER_MARKERS = [
    "<|im_end|>", "<im_end>", "|im_end|",
    "</thinkink>", "</thinkthink>", "<thinkink>", "<thinkthink>",
]

_ASSISTANT_BLOCK_PATTERN = re.compile(
    r"<\|im_start\|>\s*assistant\b.*?<\|im_end\|>",
    flags=re.DOTALL | re.IGNORECASE,
)

_ANSWER_INTRO_PATTERNS = [
    r"^(?:The\s+)?(?:GO\s+)?(?:annotations?|terms?)\s*(?:are|include)\s*:?\s*",
    r"^(?:Based\s+on\s+[^:]*,\s*)?(?:the\s+)?(?:annotations?|terms?)\s*(?:are|:)\s*:?\s*",
    r"^Answer\s*:?\s*",
    r"^Output\s*:?\s*",
    r"^Result\s*:?\s*",
    r"^[\(\[]?\s*(?:GO\s+)?(?:terms?|annotations?)\s*[\)\]]\s*:?\s*",
]
_ANSWER_INTRO_RE = re.compile("|".join(f"({p})" for p in _ANSWER_INTRO_PATTERNS), re.IGNORECASE)


def _extract_answer_segment(text: str) -> str:
    if not text:
        return ""
    matches = list(_ASSISTANT_BLOCK_PATTERN.finditer(text))
    if not matches:
        return text
    last = matches[-1]
    segment = text[last.start() : last.end()]
    segment = re.sub(r"<\|im_start\|>\s*assistant\b", "", segment, flags=re.IGNORECASE)
    segment = re.sub(r"<\|im_end\|>", "", segment, flags=re.IGNORECASE)
    return segment


def _strip_think_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think[^>]*>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    for marker in _DELIMITER_MARKERS:
        text = text.replace(marker, "\n")
    text = re.sub(r"\b[A-Z][a-z]+>\s*", "\n", text)
    return text


def _strip_outer_quotes(text: str) -> str:
    text = text.strip()
    for q in ['"', "'", "`"]:
        if len(text) >= 2 and text.startswith(q) and text.endswith(q):
            return text[1:-1].strip()
    return text


def _try_extract_json_array(text: str) -> List[str] | None:
    try:
        import json
        m = re.search(r'\[[\s\S]*?\]', text)
        if m:
            arr = json.loads(m.group())
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return [x.strip() for x in arr if x.strip()]
        m = re.search(r'\{[\s\S]*?\}', text)
        if m:
            obj = json.loads(m.group())
            for key in ("annotations", "terms", "go_terms", "answer"):
                if key in obj and isinstance(obj[key], list):
                    return [str(x).strip() for x in obj[key] if str(x).strip()]
    except Exception:
        pass
    return None


def clean_go_prediction(text: str) -> str:
    if not text:
        return ""
    text = _extract_answer_segment(text)
    text = _strip_think_blocks(text)
    text = re.sub(r"^[A-Za-z]+>\s*", "\n", text, flags=re.MULTILINE)
    text = re.sub(r"\bD{8,}\b", "\n", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.strip(" \n\t")
    return text


def split_go_terms(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"^[-*]\s*", "", text)
    text = re.sub(r"[\n]+[-*]\s*", "; ", text)
    text = re.sub(r"[\n]+\d+[\.\)]\s*", "; ", text)
    parts = re.split(r"[;\n]+", text)
    cleaned = []
    for p in parts:
        p = re.sub(r"^[-*]\s*", "", p)
        p = p.strip(" .\t\"'`:")
        if not p:
            continue
        if re.fullmatch(r"[A-Za-z]{1,6}", p) and p.lower() in {"vink", "rink", "sink", "link"}:
            continue
        cleaned.append(p)
    return cleaned


def extract_go_answer(text: str) -> str:
    """Extract normalized GO answer string (semicolon-separated) from model output."""
    cleaned = clean_go_prediction(text)
    if not cleaned:
        return ""
    arr = _try_extract_json_array(cleaned)
    if arr:
        return "; ".join(arr)
    cleaned = _strip_outer_quotes(cleaned)
    cleaned = _ANSWER_INTRO_RE.sub("", cleaned, count=1).strip()
    cleaned = re.sub(r"^:\s*", "", cleaned)
    terms = split_go_terms(cleaned)
    return "; ".join(terms) if terms else ""


def normalize_go_answer(text: str) -> str:
    """Normalize GO answer for exact match comparison."""
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" ;")
    return text.lower()


def evaluate_protein_go(eval_file: str, meta_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Exact match evaluation for Protein GO task.

    - Read eval_file (index, prediction)
    - Use meta_df answer as reference
    - Parse prediction to GO terms, compare with answer
    - Return DataFrame: ['Total', 'Correct', 'Accuracy']
    """
    data = load(eval_file)
    if "prediction" not in data:
        raise KeyError(f"`prediction` column not found in eval file: {eval_file}")
    if "index" not in data:
        raise KeyError(f"`index` column not found in eval file: {eval_file}")

    data = data.copy()
    meta = meta_df.copy()
    data["index"] = data["index"].astype(int)
    meta["index"] = meta["index"].astype(int)
    data = data[data["index"].isin(meta["index"])]

    idx2answer = {int(i): str(a) for i, a in zip(meta["index"], meta["answer"])}

    pred_norm = [normalize_go_answer(extract_go_answer(str(x))) for x in data["prediction"]]
    true_norm = [normalize_go_answer(str(idx2answer.get(int(i), ""))) for i in data["index"]]

    data["pred_extracted"] = [extract_go_answer(str(x)) for x in data["prediction"]]
    data["correct"] = [p == t for p, t in zip(pred_norm, true_norm)]

    total = int(len(data))
    correct = int(sum(data["correct"]))
    acc = float(correct / total) if total > 0 else 0.0

    res = pd.DataFrame({"Total": [total], "Correct": [correct], "Accuracy": [acc * 100.0]})
    score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
    dump(res, score_file)
    return res
