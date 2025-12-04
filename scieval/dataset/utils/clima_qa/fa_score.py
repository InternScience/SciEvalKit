from typing import List, Optional, Tuple
from .base_metric import BaseMetric
import numpy as np
from ..judge_util import build_judge
import os

try:
    from openai import OpenAI
    import tiktoken
except Exception:
    OpenAI = None
    tiktoken = None


LABELS = ["SUPPORTS", "REFUTES"]

_LOGIT_SYS_PROMPT = f"""
You are a climate expert that annotates if a given claim either SUPPORTS or REFUTES the presented evidence.
You will be provided the following as the input:

Evidence: <evidence>
Claim: <claim>

Respond with just one word - {LABELS[0]} if the claim supports the evidence and {LABELS[1]} otherwise.
""".strip()

_USER_PROMPT = "Evidence: {}\nClaim: {}"

def _smooth_probability(p: float, T: float = 5.0) -> float:
    if p <= 0.0 or p >= 1.0:
        return float(p)
    logit = np.log(p / (1.0 - p))
    smooth_logit = logit / T
    return float(1.0 / (1.0 + np.exp(-smooth_logit)))


class FAScore(BaseMetric):
    def __init__(self,
                 model: str = "gpt-4",
                 use_openai: bool = True,
                 return_none_when_disabled: bool = False,
                 temperature_smooth_T: float = 5.0):
        self.use_openai = use_openai and (OpenAI is not None) and (tiktoken is not None)
        self.return_none_when_disabled = return_none_when_disabled
        self.temperature_smooth_T = float(temperature_smooth_T)

        self.model_name = model
        self.model = build_judge(model=self.model_name)
        if self.use_openai:
            try:
                base_url = self.model.api_base[:-17]
                self.client = OpenAI(base_url=base_url)
                encoding_model = model.split(':')[1] if model[:2] == 'ft' else model
                self.encoder = tiktoken.encoding_for_model(encoding_model)
            except Exception:
                self.use_openai = False
                self.client = None
                self.encoder = None
        else:
            self.client = None
            self.encoder = None

    def _disabled_return(self):
        return None if self.return_none_when_disabled else 0.0

    def _score_one(self, evidence: str, claim: str) -> float:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                logprobs=True,
                top_logprobs=20,
                messages=[
                    {"role": "system", "content": _LOGIT_SYS_PROMPT},
                    {"role": "user", "content": _USER_PROMPT.format(evidence, claim)},
                ],
            )
            result = response.choices[0].logprobs.content[0].top_logprobs

            label_tokens = [self.encoder.decode(self.encoder.encode(label)[:1]) for label in LABELS]
            label_logprobs = [-100.0, -100.0]
            for i in range(len(label_logprobs)):
                for res in result:
                    if res.token == label_tokens[i]:
                        label_logprobs[i] = float(res.logprob)

            ratio = np.exp(label_logprobs[0] - label_logprobs[1])  # SUPPORTS vs REFUTES
            score = float(ratio / (ratio + 1.0))
            return _smooth_probability(score, T=self.temperature_smooth_T)
        except Exception as e:
            print(e)
            return 0.0

    def get_sentence_score(self, reference_answer: str, generated_answer: str) -> float:
        if not self.use_openai:
            return self._disabled_return()
        return self._score_one(evidence=reference_answer, claim=generated_answer)

    def get_corpus_score(self, reference_answer_corpus: List[str], generated_answer_corpus: List[str]) -> float:
        if not reference_answer_corpus:
            return 0.0 if not self.return_none_when_disabled else None
        if not self.use_openai:
            return self._disabled_return()

        scores = []
        for ref, hyp in zip(reference_answer_corpus, generated_answer_corpus):
            scores.append(self._score_one(evidence=ref, claim=hyp))
        return float(np.mean(scores))
