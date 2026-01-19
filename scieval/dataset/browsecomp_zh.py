import base64
import hashlib
import os
import os.path as osp
import re
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from openai import OpenAI

from .text_base import TextBaseDataset
from ..smp import LMUDataRoot, load_env


def _derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def _normalize_base_url(api_base: str) -> str:
    if not api_base:
        return api_base
    trimmed = api_base.rstrip("/")
    suffix = "/chat/completions"
    if trimmed.endswith(suffix):
        trimmed = trimmed[: -len(suffix)]
    return trimmed


class _LLM:
    def __init__(self, model_version: str, api_key: str, api_base: str):
        if not api_key:
            raise ValueError("API key is required for judge.")
        self.client = OpenAI(api_key=api_key, base_url=_normalize_base_url(api_base))
        self.model = model_version

    def __call__(self, query: str, system_prompt: str = "You are a helpful assistant.") -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content


class BrowseCompZH(TextBaseDataset):
    TYPE = "TEXT"
    MODALITY = "TEXT"
    dataset_name = "BrowseCompZH"

    query_prompt = """
{Question}

如果你回复的问题需要借助外部资源，请根据你自身的知识储备给出具体答案，而不是拒答后让用户自行查询。你的回复应遵循以下格式：
Explanation: {{你对最终答案的解释}}
Exact Answer: {{你简洁的最终答案}}
Confidence: {{你对答案的置信度得分在 0% 到 100% 之间}}
""".strip()

    judge_prompt = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。
""".strip()

    def __init__(
        self,
        dataset: str = "BrowseCompZH",
        split: str = "test",
        api_key: str = "",
        api_base: str = "",
        judge_model: str = "o4-mini",
        **kwargs,
    ):
        self.split = split
        if not api_key or not api_base:
            load_env()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_base = _normalize_base_url(
            api_base
            or os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("OPENAI_API_BASE", "")
        )
        self.judge_model = judge_model
        super().__init__(dataset=dataset, **kwargs)
        self.judger = None
        if self.api_key and self.api_base:
            self.judger = _LLM(self.judge_model, self.api_key, self.api_base)

    @classmethod
    def supported_datasets(cls):
        return ["BrowseCompZH"]

    def load_data(self, dataset):
        try:
            cache_dir = osp.join(LMUDataRoot(), "hf_datasets")
            ds = load_dataset("PALIN2018/BrowseComp-ZH", split=self.split, cache_dir=cache_dir)
        except Exception as err:
            raise RuntimeError(
                "Failed to load BrowseComp-ZH. Ensure 'datasets' is installed and network access is available. "
                f"Original error: {err}"
            )
        rows: List[Dict[str, Any]] = []
        for idx, sample in enumerate(ds):
            rows.append(
                {
                    "index": idx,
                    "Question": sample["Question"],
                    "Answer": sample["Answer"],
                    "canary": sample["canary"],
                }
            )
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = _decrypt(line["Question"], line["canary"])
        prompt = self.query_prompt.format(Question=question)
        return [dict(type="text", value=prompt)]

    def _get_judger(self) -> _LLM:
        if self.judger is None:
            load_env()
            resolved_key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
            resolved_base = _normalize_base_url(
                self.api_base
                or os.environ.get("OPENAI_API_BASE", "")
                or os.environ.get("OPENAI_BASE_URL", "")
            )
            self.judger = _LLM(self.judge_model, resolved_key, resolved_base)
        return self.judger

    def score_agent_sample(self, idx: int, model_response: str, **judge_kwargs) -> Dict[str, float]:
        sample = self.data.iloc[idx]
        question = _decrypt(sample["Question"], sample["canary"])
        answer = _decrypt(sample["Answer"], sample["canary"])

        prompt = self.judge_prompt.format(
            question=question, correct_answer=answer, response=model_response
        )
        judge_response = self._get_judger()(prompt)
        match = re.search(r"correct: (yes|no)", judge_response)
        final_decision = match.group(1) if match else "no"

        return {
            "accuracy": 1.0 if final_decision == "yes" else 0.0,
            "question": question,
            "answer": answer,
            "final_answer": model_response,
            "judge_decision": final_decision,
            "judge_response": judge_response,
        }

    def evaluate(self, eval_file, **judge_kwargs):
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
            pred = row.get("prediction", row.get("final_answer", ""))
            scores.append(self.score_agent_sample(idx, pred, **judge_kwargs)["accuracy"])

        accuracy = sum(scores) / len(scores) if scores else 0.0
        return {"accuracy": accuracy}
