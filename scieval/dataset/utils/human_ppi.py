from __future__ import annotations

import re
from typing import Any

from ...smp import *
from ...smp.file import get_intermediate_file_path


def parse_human_ppi_output(output_text: str) -> int:
    """
    解析模型输出，判断是 Yes(1) / No(0)。

    逻辑基本复用自本地测试脚本 `test_human_ppi.py`：
    - 优先看开头是否是 Yes./No.
    - 再看前若干字符中是否出现 yes/no
    - 再看前 200 字符中是否包含显式 Yes/No 词
    - 如果文本里出现 interaction/binding 等肯定词，再结合否定词判断

    返回：
        1 表示预测为 Yes（存在 PPI）
        0 表示预测为 No（不存在 PPI，或无法确定）
    """
    output_text = (output_text or "").strip()
    if not output_text:
        return 0

    # 1) 直接看开头
    if re.match(r"^(Yes|yes|YES)[\s.,;:!?]", output_text):
        return 1
    if re.match(r"^(No|no|NO)[\s.,;:!?]", output_text):
        return 0

    # 2) 看前若干字符
    first_words = output_text[:50].lower()
    if first_words.startswith("yes"):
        return 1
    if first_words.startswith("no"):
        return 0

    # 3) 在前 200 字符中查找显式 Yes/No
    head = output_text[:200]
    if re.search(r"\b(No|no|NO)\b", head):
        return 0
    if re.search(r"\b(Yes|yes|YES)\b", head):
        return 1

    # 4) 检查 interaction/binding 等相关表述
    if re.search(r"\b(interaction|binding|interact|bind|will interact|can interact)\b", output_text, re.IGNORECASE):
        # 若包含明显否定词，则判为 0
        if re.search(
            r"\b(not|no|unlikely|cannot|doesn\'t|don\'t|won\'t|no interaction|unlikely to)\b",
            output_text,
            re.IGNORECASE,
        ):
            return 0
        return 1

    # 无法判断时，默认 No
    return 0


def evaluate_human_ppi_binary(eval_file: str, meta_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    对 Human PPI 任务做一个“基础测试”：

    - 读取 eval_file（通常是预测结果的 tsv/csv/pkl），要求包含：
        - `index`：样本索引（与原始 tsv 中对齐）
        - `prediction`：模型生成的自然语言回答
    - 利用 meta_df 中的 `category` 作为真实标签（0/1）
    - 使用 parse_human_ppi_output 将自然语言 prediction 解析成 0/1
    - 计算整体准确率，并返回一个只有一行的 DataFrame：
        columns: ['Total', 'Correct', 'Accuracy']
    """
    data = load(eval_file)
    if "prediction" not in data:
        raise KeyError(f"`prediction` column not found in eval file: {eval_file}")
    if "index" not in data:
        raise KeyError(f"`index` column not found in eval file: {eval_file}")

    # 确保 index 类型和 meta_df 一致（通常为 int）
    data = data.copy()
    meta = meta_df.copy()
    data["index"] = data["index"].astype(int)
    meta["index"] = meta["index"].astype(int)

    # 只保留在 meta 里出现的索引
    data = data[data["index"].isin(meta["index"])]

    # 对 prediction 做解析
    preds = [parse_human_ppi_output(str(x)) for x in data["prediction"]]
    data["pred_label"] = preds

    # 合并真实标签
    idx2label = {int(i): int(c) for i, c in zip(meta["index"], meta["category"])}
    data["true_label"] = [idx2label[int(i)] for i in data["index"]]

    data["correct"] = data["pred_label"] == data["true_label"]

    total = int(len(data))
    correct = int(data["correct"].sum())
    acc = float(correct / total) if total > 0 else 0.0

    res = pd.DataFrame(
        {
            "Total": [total],
            "Correct": [correct],
            "Accuracy": [acc * 100.0],
        }
    )

    # 同其他数据集风格：把统计结果另存一份
    score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
    dump(res, score_file)

    return res

