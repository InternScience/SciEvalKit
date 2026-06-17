"""
TRQA dataset for SciEvalKit
"""
from __future__ import annotations
import re
from typing import Any, Dict, Set
from .text_mcq import TextMCQDataset
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich
from scieval.dataset.utils.judge_util import build_judge_model



def _extract_choice_letters(text: str) -> Set[str]:
    """
    Extract choice letters strictly from <choice> tags if present,
    falling back to heuristics if tags are missing.
    """
    if not text:
        return set()

    text = text.strip()
    def parse_letters(s: str) -> Set[str]:
        if not s: return set()
        s_clean = re.sub(r'<[^>]+>', '', s)
        s_upper = s_clean.upper()
        letters = set(re.findall(r'[A-Z]', s_upper))
        if len(letters) > 10:
            return set()
        return letters

    choice_matches = re.findall(r'<choice>(.*?)</choice>', text, re.IGNORECASE | re.DOTALL)

    if choice_matches:
        for content in reversed(choice_matches):
            res = parse_letters(content)
            if res: return res
    latex_matches = re.findall(r'\\boxed\s*\{([^}]+)\}', text)
    if latex_matches:
        res = parse_letters(latex_matches[-1])
        if res: return res

    special_box_matches = re.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text, re.DOTALL)
    if special_box_matches:
        res = parse_letters(special_box_matches[-1])
        if res: return res

    keyword_patterns = [
        r'(?:Answer|Option|Correct choice)s?\s*[:\-]\s*([A-Z\s,]+)(?:$|\.)',
    ]
    for pattern in keyword_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            res = parse_letters(matches[-1])
            if res: return res

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1]
        clean_last = re.sub(r'<[^>]+>', '', last_line)
        clean_last = re.sub(r'[\s,\.\*`]', '', clean_last.upper())

        if clean_last.isalpha() and 0 < len(clean_last) <= 5:
            return parse_letters(clean_last)

    return set()

class TRQA(TextMCQDataset):
    """TRQA dataset for SciEvalKit - supports CSV format with Question, Options (JSON), Answer."""
    MODALITY = 'TEXT'
    TYPE = 'MCQ'
    DATASET_URL = {
        'TRQA': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/TRQA.tsv'
    }
    DATASET_MD5 = {
        'TRQA':'756bf0b1cb5e917f2ea1a0fabfd3509b'
    }

    def load_data(self, dataset):
        data = self.prepare_tsv(self.DATASET_URL[dataset],self.DATASET_MD5[dataset])
        # Standardize column names (handle case variations)
        data.columns = data.columns.str.strip()
        col_map = {}
        for col in data.columns:
            if col.lower() == 'question':
                col_map[col] = 'question'
            elif col.lower() == 'options':
                col_map[col] = 'options'
            elif col.lower() == 'answer':
                col_map[col] = 'answer'
        if col_map:
            data = data.rename(columns=col_map)

        # Ensure required columns exist
        required = {"question", "options", "answer"}
        if not required.issubset(data.columns):
            raise ValueError(f"TSV missing columns: {required - set(data.columns)}")

        # Add index if not present
        if 'index' not in data.columns:
            data['index'] = range(len(data))

        return data

    def __init__(self, dataset: str = 'TRQA', **kwargs) -> None:
        """Initialize TRQA dataset.

        Args:
            dataset: Dataset name (default: 'TRQA')
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__(dataset=dataset, **kwargs)

    # ------------- judge a single row -------------
    def _judge(self, row: pd.Series, judge_model=None) -> Dict[str, Any]:
        """Judge a single prediction against ground truth."""
        gold = str(row["answer"]).strip().upper()
        pred = str(row["prediction"]).strip()

        result = {"hit": False, "pred_norm": "", "judge_used": False, "log": ""}

        # Extract choice letters from both prediction and gold
        letters_pred = _extract_choice_letters(pred)
        letters_gold = _extract_choice_letters(gold)

        # If no letters extracted from prediction and judge model available, use judge
        if not letters_pred and judge_model is not None:
            jp = (
                f"Question: {row['question']}\n"
                f"Ground truth answer: {gold}\n"
                f"Model answer: {pred}\n"
                "Is the model answer fully correct? Reply with a single word: Yes or No."
            )
            judge_out = judge_model.generate(jp)
            result["judge_used"] = True
            result["log"] = judge_out
            result["hit"] = judge_out.strip().lower().startswith("yes")
            result["pred_norm"] = ",".join(sorted(letters_pred)) if letters_pred else ""
            return result

        # Compare choice letters
        if letters_gold:
            # Sort both sets for comparison (exact match for multiple choice)
            pred_sorted = set(sorted(letters_pred))
            gold_sorted = set(sorted(letters_gold))
            result["hit"] = (pred_sorted == gold_sorted)
        else:
            # Fallback: simple string matching
            result["hit"] = gold.lower() in pred.lower()

        result["pred_norm"] = ",".join(sorted(letters_pred)) if letters_pred else ""
        return result

    # ---------- prompt builder override ----------
    def build_prompt(self, line):
        """Build prompt from TRQA format (Question, Options JSON, Answer).

        Parses JSON Options and formats them as:
        Question: ...
        Options:
        A. ...
        B. ...
        ...
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = str(line["question"]).strip()

        # Parse Options JSON string
        options_str = str(line.get("options", ""))
        options = {}

        # pandas should handle CSV escaping, but we need to handle various formats
        try:
            # Try to parse as JSON directly (pandas should have unescaped it)
            options = json.loads(options_str)
        except (json.JSONDecodeError, ValueError):
            try:
                # Handle case where string might still have outer quotes
                cleaned = options_str.strip()
                if cleaned.startswith('"') and cleaned.endswith('"'):
                    cleaned = cleaned[1:-1]
                # Try parsing again
                options = json.loads(cleaned)
            except (json.JSONDecodeError, ValueError):
                try:
                    # Handle double-escaped quotes ("" -> ")
                    cleaned = cleaned.replace('""', '"')
                    options = json.loads(cleaned)
                except (json.JSONDecodeError, ValueError):
                    # Last resort: extract manually using regex
                    # Look for patterns like "A": "value" or '"A": "value"'
                    # This regex handles: "A": "value", "A":'value', A:"value", etc.
                    pattern = r'["\']?([A-Z])["\']?\s*:\s*["\']([^"\']+)["\']'
                    matches = re.findall(pattern, options_str)
                    options = {k: v for k, v in matches}

        # Build options prompt
        options_prompt = 'Options:\n'
        for key in sorted(options.keys()):
            options_prompt += f'{key}. {options[key]}\n'

        # Build final prompt
        prompt = f'Question: {question}\n'
        if options:
            prompt += options_prompt
            prompt += 'Please select ALL correct answers from the options above. Return only the option letters and wrapped with <choice></choice> (e.g., <choice>A, B, C</choice> or <choice>ABC</choice>).\n'

        return [dict(type='text', value=prompt)]

    # ------------- public evaluate -------------
    def evaluate(self, eval_file: str, **judge_kwargs):
        """
        Evaluate predictions against ground truth answers.

        Parameters
        ----------
        eval_file : str
            Path to pkl/json/csv/xlsx file with 'prediction' column.
        judge_kwargs : Any
            Passed to build_judge; example: {'model': 'gpt-4o-1120', 'nproc': 4}
        """
        # Resolve eval_file loading based on extension
        if eval_file.lower().endswith(('.xlsx', '.xls')):
            data = pd.read_excel(eval_file)
        elif eval_file.lower().endswith('.csv'):
            data = pd.read_csv(eval_file)
        else:
            data = load(eval_file)
            # PRED_FORMAT=json round-trips DataFrames as list[dict]; coerce back
            # so downstream .columns / column ops work uniformly.
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)

        # Standardize column names (handle case variations)
        data.columns = data.columns.str.strip()
        col_map = {}
        for col in data.columns:
            col_lower = col.lower()
            if col_lower == 'question':
                col_map[col] = 'question'
            elif col_lower == 'options':
                col_map[col] = 'options'
            elif col_lower == 'answer':
                col_map[col] = 'answer'
            elif col_lower == 'prediction':
                col_map[col] = 'prediction'
        if col_map:
            data = data.rename(columns=col_map)

        # Ensure required columns exist
        required_cols = {'prediction', 'answer'}
        if not required_cols.issubset(data.columns):
            raise ValueError(f"eval_file missing columns: {required_cols - set(data.columns)}")

        # Add index if not present
        if 'index' not in data.columns:
            data['index'] = range(len(data))

        # Ensure str types
        data['index'] = [str(x) for x in data['index']]
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        # Add question column if missing (needed for judge)
        if 'question' not in data.columns:
            # Try to get from self.data if index matches
            if hasattr(self, 'data') and 'question' in self.data.columns:
                question_map = {str(x): y for x, y in zip(self.data['index'], self.data['question'])}
                data['question'] = [question_map.get(str(idx), '') for idx in data['index']]
            else:
                data['question'] = [''] * len(data)

        storage = get_intermediate_file_path(eval_file, '_judge')
        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')

        # Ensure directories exist
        if osp.dirname(storage):
            os.makedirs(osp.dirname(storage), exist_ok=True)
        if osp.dirname(tmp_file):
            os.makedirs(osp.dirname(tmp_file), exist_ok=True)

        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            # existing partial cache
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            # optional judge model
            judge_name = judge_kwargs.pop('model', 'exact_matching')
            print(f"Judge way: {judge_name}")
            judge_model = None
            if judge_name != 'exact_matching':
                try:
                    judge_model = build_judge_model(model=judge_name, **judge_kwargs)
                    if not judge_model.working():
                        warnings.warn("Judge model unavailable; falling back to exact matching.")
                        judge_model = None
                except Exception as e:
                    warnings.warn(f"Failed to build judge model: {e}. Falling back to exact matching.")
                    judge_model = None

            # lines still requiring judgment
            lines = [data.iloc[i] for i in range(len(data)) if str(data.iloc[i]['index']) not in ans_map]
            indices = [str(x['index']) for x in lines]

            def _worker(self_ref, line):
                return self_ref._judge(line, judge_model)

            if lines:
                jobs = [(self, line) for line in lines]
                outs = track_progress_rich(_worker, jobs, nproc=nproc, chunksize=nproc)
                for idx, res in zip(indices, outs):
                    ans_map[idx] = res
                dump(ans_map, tmp_file)

            # attach results
            data['hit'] = [ans_map.get(str(x), {}).get('hit', False) for x in data['index']]
            data['log'] = [ans_map.get(str(x), {}).get('log', '') for x in data['index']]
            data['pred_norm'] = [ans_map.get(str(x), {}).get('pred_norm', '') for x in data['index']]
            dump(data, storage)

        # reload judged data
        data = load(storage)
        # PRED_FORMAT=json round-trips DataFrames as list[dict]; coerce back.
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        # simple accuracy report
        acc = np.mean(data['hit']) * 100.0
        score_df = pd.DataFrame({'TRQA-Acc(%)': [acc]})

        score_file = get_intermediate_file_path(eval_file, '_score', 'csv')
        dump(score_df, score_file)

        return score_df
