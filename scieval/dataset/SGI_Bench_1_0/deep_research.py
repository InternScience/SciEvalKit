from typing import Any, Dict, List
import json
import copy as _copy
from datasets import load_dataset
import pandas as pd
from ..utils.judge_util import build_judge
from ...utils.mp_util import track_progress_rich
from ..text_base import TextBaseDataset
from ...smp.file import dump, load, get_intermediate_file_path
from json_repair import repair_json


PARSER_MODEL_DEFAULT = 'gpt-4.1-mini'


def extract_final_answer(answer_with_thinking: str, start_tag='<answer>', end_tag='</answer>'):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


def _build_judge_messages(prompt: str):
    return [
        {"role": "system", "value": "You are a helpful assistant.", "type": "text"},
        {"role": "user", "value": prompt, "type": "text"},
    ]


def apply_answer_parser(ques_dict: dict, judge_kwargs: dict):
    """Run a small LLM to normalize the extracted answer (mirrors upstream AnswerPaser)."""
    extracted = extract_final_answer(ques_dict['prediction'])
    if extracted is None:
        extracted = str(ques_dict['prediction'])

    # Decide example format: numeric or text
    try:
        float(ques_dict['answer'])
        example = "0.25"
    except (ValueError, TypeError):
        example = "T cell and B cell"

    type_hint = 'One letter' if isinstance(example, str) and len(example) == 1 else type(example).__name__

    paser_kwargs = _copy.deepcopy(judge_kwargs)
    paser_kwargs['model'] = PARSER_MODEL_DEFAULT
    paser_kwargs.pop('max_tokens', None)
    paser_kwargs.pop('nproc', None)

    system_prompt = (
        "You are an expert in structured data parsing. "
        "Your task is to convert text content into a standardized structured output "
        "based on a provided example data structure.\n\n"
        "### Instructions\n"
        "1. Analyze the example structure within the <example> tags.\n"
        "2. Determine output type per <type> tags.\n"
        "3. Transform the content from <input_text> to match the example format.\n"
        "4. Preserve semantics, only adjust structure.\n"
        "5. Ignore explanatory text; extract only the core final output.\n"
        "6. Output only the transformed structured content, no extra explanation.\n\n"
        f"<example>\n{example}\n</example>\n\n"
        f"<type>\n{type_hint}\n</type>"
    )
    query = f"<input_text>\n{extracted}\n</input_text>"
    messages = [
        {"role": "system", "value": system_prompt, "type": "text"},
        {"role": "user", "value": query, "type": "text"},
    ]

    try:
        paser = build_judge(**paser_kwargs)
        normalized = str(paser.generate(messages)).strip()
    except Exception:
        normalized = extracted

    return extracted, normalized


def answer_llm_judge(ques_dict: dict, judge_kwargs: dict, exact_match: int):
    """Binary 0/1 semantic judge mirroring upstream step_2_score.py answer_llm_judge."""
    if exact_match:
        return 1, "Exact match."

    prompt = (
        "You are a scientific answer evaluator. Compare the agent's answer to the "
        "reference answer for the following question.\n\n"
        f"Question: {ques_dict['question']}\n\n"
        f"Reference Answer: {ques_dict['answer']}\n\n"
        f"Agent's Answer: {ques_dict.get('model_answer', '')}\n\n"
        f"Parser-normalized Agent's Answer: {ques_dict.get('model_answer_after_llm_paser', '')}\n\n"
        "Evaluate whether the agent's answer is essentially correct. Consider:\n"
        "- For numerical answers: accept if within 5% relative error after accounting "
        "for obvious units, percentage signs, or formatting differences when the "
        "context supports the same meaning\n"
        "- For text answers: accept if the meaning is equivalent\n"
        "- Partial credit is NOT given - answer is either correct (1) or incorrect (0)\n\n"
        'Respond with a JSON object: {"judge": 0 or 1, "reason": "brief explanation"}'
    )

    try:
        judge = build_judge(**judge_kwargs)
        response = str(judge.generate(_build_judge_messages(prompt)))
        start = response.find('{')
        end = response.rfind('}') + 1
        if start == -1 or end <= start:
            return 0, f"No JSON in judge response: {response[:200]}"
        result = json.loads(repair_json(response[start:end]))
        return int(result.get("judge", 0)), result.get("reason", "")
    except Exception as e:
        return 0, f"Judge error: {e}"


def eval_model_output(ques_dict, judge_kwargs):
    # ---- 1. Answer extraction + parser normalization ----
    extracted, normalized = apply_answer_parser(ques_dict, judge_kwargs)
    ques_dict['model_answer'] = extracted
    ques_dict['model_answer_after_llm_paser'] = normalized

    # ---- 2. Exact Match (two-route) ----
    gold = str(ques_dict['answer'])
    exact_match = 1 if (
        gold == str(extracted) or gold == str(normalized or '')
    ) else 0
    ques_dict['exact_match'] = exact_match

    # ---- 3. Binary LLM Judge ----
    llm_judge_score, llm_judge_reason = answer_llm_judge(
        ques_dict, judge_kwargs, exact_match
    )
    ques_dict['llm_judge'] = llm_judge_score
    ques_dict['llm_judge_reason'] = llm_judge_reason

    # ---- 4. Step Level Acc (existing logic, key renamed to step_llm_judge) ----
    newline = '\n'
    step_prompt = f"""
You are an expert in systematically validating and evaluating LLM-generated solutions. Your task is to rigorously analyze the correctness of a provided solution by comparing it step-by-step against the reference solution, and output **only** a structured verification list—with no additional text.

## Instructions
1. Break down the given LLM solution into individual steps and evaluate each one against the corresponding reference solution steps.
2. For each step, include the following three components:
   - **solution_step**: The specific part of the LLM solution being evaluated.
   - **reason**: A clear, critical explanation of whether the step contains errors, omissions, or deviations from the reference approach. Be stringent in your assessment.
   - **judge**: Your verdict: either `"correct"` or `"incorrect"`.
3. If the final LLM answer is incorrect, you must identify at least one step in your analysis as incorrect.
4. Justify your judgments rigorously, pointing out even minor inaccuracies or logical flaws.
5. Do not attempt to answer the original question—your role is strictly to evaluate.
6. Output **only** a list of dictionaries in the exact format provided below. Do not include any other text or comments.

## Question
{ques_dict['question']}

## Reference Solution Steps
{newline.join(ques_dict['steps'])}

## Reference Answer
{ques_dict['answer']}

## LLM Solution Steps
{ques_dict['prediction']}

## LLM Answer
{extracted}

## Output Example
[
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
    {{"solution_step": "step content", "reason": "reason of the judgement", "judge": "correct or incorrect"}},
]
"""

    step_judge_list = None
    step_level_acc = 0.0
    try:
        judge = build_judge(**judge_kwargs)
        out = str(judge.generate(_build_judge_messages(step_prompt)))
        s = out.find('[')
        e = out.rfind(']') + 1
        step_judge_list = eval(repair_json(out[s:e]))
        if step_judge_list:
            correct = sum(1 for st in step_judge_list if st.get("judge") == "correct")
            step_level_acc = correct / len(step_judge_list)
    except Exception as e:
        print(e)

    ques_dict['step_llm_judge'] = step_judge_list
    ques_dict['step_level_acc'] = step_level_acc
    return ques_dict


class SGI_Bench_Deep_Research(TextBaseDataset):
    TYPE = 'QA'

    @classmethod
    def supported_datasets(cls):
        return ["SGI-DeepResearch"]

    def load_data(self, dataset):
        hf = load_dataset("InternScience/SGI-DeepResearch", split="test")

        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            rows.append(
                {
                    "index": idx,
                    "id": prob["idx"],
                    "question": prob["question"],
                    "steps": prob["steps"],
                    "answer": prob["answer"],
                    "discipline": prob["discipline"],
                    "direction": prob["direction"],
                    "type": prob["type"]
                }
            )
            idx += 1
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question'] + """
You can reason step by step before giving the final answer. The final answer should be enclosed by <answer> and </answer>.

Example:
Step 1. ...
Step 2. ...
...
<answer>1.00</answer>
"""

        msgs = [{'type': 'text', 'value': question}]
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        data = pd.DataFrame(data)

        if judge_kwargs.get('model') is None:
            judge_kwargs['model'] = 'o4-mini'
        if judge_kwargs.get('max_tokens') is None:
            judge_kwargs['max_tokens'] = None

        inp_list = []
        for item in data.to_dict(orient="records"):
            inp_list.append({
                "ques_dict": item,
                "judge_kwargs": judge_kwargs
            })
        out_list = track_progress_rich(
            func=eval_model_output,
            tasks=inp_list,
            nproc=judge_kwargs.get('nproc', 48)
        )

        n = len(out_list) if out_list else 1
        result = {
            'Exact Match':    sum(item['exact_match']    for item in out_list) / n,
            'LLM Judge':      sum(item['llm_judge']      for item in out_list) / n,
            'Step Level Acc': sum(item['step_level_acc'] for item in out_list) / n,
        }

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(out_list, score_file)
        dump(result, result_file)
        return result
