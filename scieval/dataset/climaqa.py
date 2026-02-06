from ..smp import *
from .text_base import TextBaseDataset
from typing import Dict, Any, List
from .utils.clima_qa.bleu_score import BLEUScore
from .utils.clima_qa.bert_score import BERTScore
from .utils.clima_qa.fa_score import FAScore
from .utils.clima_qa.phrase_similarity import PhraseSimilarity
from datetime import datetime
from .utils.clima_qa.llm_em_matcher import em_with_llm
from .utils.judge_util import build_judge


def get_valid_fascore_params(judge_kwargs):
    """Extract parameters accepted by FAScore constructor from judge_kwargs"""
    import inspect
    
    # Get FAScore constructor parameters
    sig = inspect.signature(FAScore.__init__)
    fascore_params = list(sig.parameters.keys())[1:]  # Skip 'self'
    
    # Extract valid parameters
    valid_params = {}
    for param_name in fascore_params:
        if param_name in judge_kwargs:
            valid_params[param_name] = judge_kwargs[param_name]
        else:
            # Use custom default for 'model' if not provided
            if param_name == 'model':
                valid_params[param_name] = 'gpt-4o-2024-11-20'
                continue
                
            # For other missing parameters, use the default from FAScore
            param = sig.parameters[param_name]
            if param.default != inspect.Parameter.empty:
                valid_params[param_name] = param.default
    
    return valid_params

class Clima_QA(TextBaseDataset):\

    judge = None

    TYPE = 'QA'

    DATASET_URL = {
        'Clima_QA': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/ClimaQA_Gold.tsv',
    }

    DATASET_MD5 = {
        'Clima_QA': 'f50e5da976533110e557747928b41014'
    }

    CLOZE_SYSTEM_PROMPT = '''
    You are an expert assistant in the domain of climate science for fill-in-the-blank question-answering.
    The question will contain a sientific statement with a single word masked with the token - <MASK>.
    You need to find the most appropriate word that can be filled in it's place based on the context around it.

    you need to ouput a single word that best fits the blank
    '''

    MCQ_SYSTEM_PROMPT = '''
    You are an expert assistant in the domain of climate science for multiple choice question-answering tasks. The question will be of the following format:
    --------------
    Question_text

    a) Option a
    b) Option b
    c) Option c
    d) Option d

    -----------

    you need to ouput a single letter that represents the correct option. Make sure to output a single letter.
    '''

    QA_SYSTEM_PROMPT = '''
    You are an expert assistant in the domain of climate science for question-answering tasks.
    Use two sentences maximum and keep the answer concise.
    '''

    @classmethod
    def supported_datasets(cls):
        return {
            'Clima_QA': cls,
        }


    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        question_type = str(line['task']).strip().lower()
        assert question_type in {'mcq', 'ffq', 'cloze'}, f'bad task: {line["task"]!r}'

        msg = []

        msg_value = ''

        if question_type == 'mcq':
            msg_value = self.MCQ_SYSTEM_PROMPT
        elif question_type == 'ffq':
            msg_value = self.QA_SYSTEM_PROMPT
        elif question_type == 'cloze':
            msg_value = self.CLOZE_SYSTEM_PROMPT

        msg_value += '\n'
        msg_value += question

        msg.append({'role':'user',"type": "text",'value':msg_value})
        return msg

    def evaluate(self, eval_file: str, **judge_kwargs):

        verbose = bool(judge_kwargs.get("verbose", False))
        model_name = judge_kwargs.get("model_name", None) or "model"
        use_fa = True if judge_kwargs.get("use_fa", False) else False

        df = load(eval_file)
        # df = pd.DataFrame(df)
        required_cols = ["index", "task", "question", "answer", "prediction", "Complexity", "Validation"]
        for c in required_cols:
            if c not in df.columns:
                raise ValueError(f"Required column '{c}' not found in eval_file: {eval_file}")

        eval_dir = os.path.dirname(os.path.abspath(eval_file))
        jsonl_path = os.path.join(eval_dir, f"{model_name}_climaqa_eval_results.jsonl")
        mcq_tsv_path = os.path.join(eval_dir, f"{model_name}_climaqa_mcq.tsv")
        cloze_tsv_path = os.path.join(eval_dir, f"{model_name}_climaqa_cloze.tsv")
        ffq_bleu_tsv_path = os.path.join(eval_dir, f"{model_name}_climaqa_ffq_bleu.tsv")
        ffq_bert_tsv_path = os.path.join(eval_dir, f"{model_name}_climaqa_ffq_bert.tsv")
        ffq_fa_tsv_path = os.path.join(eval_dir, f"{model_name}_climaqa_ffq_fa.tsv")

        bleu = BLEUScore()
        bert = BERTScore(lang="en")
        if use_fa:
            valid_kwargs = get_valid_fascore_params(judge_kwargs)
            fa = FAScore(**valid_kwargs)
        else:
            fa = None
        ps = PhraseSimilarity()

        per_sample: List[Dict[str, Any]] = []
        eval_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        eval_version = "climaqa_v1"

        for row in df.itertuples(index=False):
            task = getattr(row, "task")
            rec = {
                "index": getattr(row, "index"),
                "task": task,
                "Complexity": getattr(row, "Complexity") or "",
                "Validation": getattr(row, "Validation") or "",
                "question": getattr(row, "question"),
                "answer": getattr(row, "answer"),
                "prediction": getattr(row, "prediction"),
                "metrics": {},
                "model": model_name,
                "eval_time": eval_time,
                "eval_version": eval_version
            }

            if self.judge is None:
                judge_model_name = judge_kwargs.pop('model', 'gpt-4o-1120')
                self.judge = build_judge(model=judge_model_name, **judge_kwargs)
            if task == "mcq":
                rec["metrics"]["EM"] = em_with_llm(
                    answer=rec["answer"],
                    prediction=rec["prediction"],
                    task=rec.get("task", ""),
                    model=self.judge,
                    enable_llm_on_mismatch=True
                )

            elif task == "cloze":
                rec["metrics"]["EM"] = em_with_llm(
                    answer=rec["answer"],
                    prediction=rec["prediction"],
                    task=rec.get("task", ""),
                    model=self.judge,
                    enable_llm_on_mismatch=True
                )
                try:
                    rec["metrics"]["PS"] = float(ps.phrase_similarity(
                        blank_statement=rec["question"],
                        generated_term=rec["prediction"],
                        correct_term=rec["answer"]
                    ))
                except Exception:
                    rec["metrics"]["PS"] = 0.0

            elif task == "ffq":
                try:
                    rec["metrics"]["BLEU"] = float(bleu.get_sentence_score(rec["answer"], rec["prediction"]))
                except Exception:
                    rec["metrics"]["BLEU"] = 0.0
                try:
                    _, _, f1 = bert.get_sentence_score(rec["answer"], rec["prediction"])
                    rec["metrics"]["BERT"] = float(f1)
                except Exception:
                    rec["metrics"]["BERT"] = 0.0
                if fa is not None:
                    try:
                        rec["metrics"]["FA"] = float(fa.get_sentence_score(rec["answer"], rec["prediction"]))
                    except Exception:
                        rec["metrics"]["FA"] = 0.0

            per_sample.append(rec)

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in per_sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if verbose:
            print(f"[evaluate] per-sample results written: {jsonl_path}")

        pdf = pd.DataFrame(per_sample)

        mcq_df = pdf[pdf["task"] == "mcq"].copy()

        def _acc(sub_df):
            if len(sub_df) == 0: return 0.0
            return float(np.mean([m.get("EM", 0) for m in sub_df["metrics"]]))

        if len(mcq_df):
            mcq_base = _acc(mcq_df[mcq_df["Complexity"] == "BASE"])
            mcq_reason = _acc(mcq_df[mcq_df["Complexity"] == "REASONING"])
            mcq_hypo = _acc(mcq_df[mcq_df["Complexity"] == "HYPOTHETICAL"])
            mcq_overall = _acc(mcq_df)
            mcq_table = pd.DataFrame({
                "Model": [model_name],
                "Base": [mcq_base],
                "Reason": [mcq_reason],
                "Hypo": [mcq_hypo],
                "Overall": [mcq_overall]
            })
        else:
            mcq_table = pd.DataFrame(columns=["Model", "Base", "Reason", "Hypo", "Overall"])

        cloze_df = pdf[pdf["task"] == "cloze"].copy()
        if len(cloze_df):
            cloze_em = float(np.mean([m.get("EM", 0) for m in cloze_df["metrics"]]))
            cloze_ps = float(np.mean([m.get("PS", 0.0) for m in cloze_df["metrics"]]))
            cloze_table = pd.DataFrame({"Model": [model_name], "EM": [cloze_em], "PS": [cloze_ps]})
        else:
            cloze_table = pd.DataFrame(columns=["Model", "EM", "PS"])

        ffq_df = pdf[pdf["task"] == "ffq"].copy()

        def _ffq_subset_metrics(sub_df: pd.DataFrame):
            if len(sub_df) == 0:
                return {"BLEU": 0.0, "BERT": 0.0, "FA": 0.0}
            refs = sub_df["answer"].tolist()
            hyps = sub_df["prediction"].tolist()
            try:
                bleu_c = float(bleu.get_corpus_score(refs, hyps))
            except Exception:
                bleu_c = 0.0
            try:
                bert_c = float(bert.get_corpus_score(refs, hyps))
            except Exception:
                bert_c = 0.0
            if use_fa and ("FA" in sub_df["metrics"].iloc[0] or True):
                try:
                    fa_vals = [float(m.get("FA", 0.0)) for m in sub_df["metrics"]]
                    fa_c = float(np.mean(fa_vals)) if len(fa_vals) else 0.0
                except Exception:
                    fa_c = 0.0
            else:
                fa_c = 0.0
            return {"BLEU": bleu_c, "BERT": bert_c, "FA": fa_c}

        def _weighted_overall(parts: List[Dict[str, float]], sizes: List[int]):
            w = np.array(sizes, dtype=float)
            s = float(w.sum())
            if s == 0:
                return {"BLEU": 0.0, "BERT": 0.0, "FA": 0.0}

            def wmean(key): return float(np.sum([p[key] * w[i] for i, p in enumerate(parts)]) / s)

            return {"BLEU": wmean("BLEU"), "BERT": wmean("BERT"), "FA": wmean("FA")}

        if len(ffq_df):
            base_sub = ffq_df[ffq_df["Complexity"] == "BASE"]
            reason_sub = ffq_df[ffq_df["Complexity"] == "REASONING"]
            hypo_sub = ffq_df[ffq_df["Complexity"] == "HYPOTHETICAL"]

            base_m = _ffq_subset_metrics(base_sub)
            reason_m = _ffq_subset_metrics(reason_sub)
            hypo_m = _ffq_subset_metrics(hypo_sub)

            overall_m = _weighted_overall([base_m, reason_m, hypo_m],
                                          [len(base_sub), len(reason_sub), len(hypo_sub)])

            ffq_bleu_table = pd.DataFrame({
                "Model": [model_name],
                "Base: Bleu": [base_m["BLEU"]],
                "Reasoning: Bleu": [reason_m["BLEU"]],
                "Hypothetical: Bleu": [hypo_m["BLEU"]],
                "Overall: Bleu": [overall_m["BLEU"]],
            })
            ffq_bert_table = pd.DataFrame({
                "Model": [model_name],
                "Base: Bert": [base_m["BERT"]],
                "Reasoning: Bert": [reason_m["BERT"]],
                "Hypothetical: Bert": [hypo_m["BERT"]],
                "Overall: Bert": [overall_m["BERT"]],
            })
            ffq_fa_table = pd.DataFrame({
                "Model": [model_name],
                "Base: FA": [base_m["FA"]],
                "Reasoning: FA": [reason_m["FA"]],
                "Hypothetical: FA": [hypo_m["FA"]],
                "Overall: FA": [overall_m["FA"]],
            })
        else:
            ffq_bleu_table = pd.DataFrame(
                columns=["Model", "Base: Bleu", "Reasoning: Bleu", "Hypothetical: Bleu", "Overall: Bleu"])
            ffq_bert_table = pd.DataFrame(
                columns=["Model", "Base: Bert", "Reasoning: Bert", "Hypothetical: Bert", "Overall: Bert"])
            ffq_fa_table = pd.DataFrame(
                columns=["Model", "Base: FA", "Reasoning: FA", "Hypothetical: FA", "Overall: FA"])

        tidy_rows = []

        # MCQ -> EM
        if len(mcq_table):
            tidy_rows += [
                {"Model": model_name, "Task": "mcq", "Split": "Base", "Metric": "EM",
                 "Score": float(mcq_table["Base"].iloc[0])},
                {"Model": model_name, "Task": "mcq", "Split": "Reasoning", "Metric": "EM",
                 "Score": float(mcq_table["Reason"].iloc[0])},
                {"Model": model_name, "Task": "mcq", "Split": "Hypothetical", "Metric": "EM",
                 "Score": float(mcq_table["Hypo"].iloc[0])},
                {"Model": model_name, "Task": "mcq", "Split": "Overall", "Metric": "EM",
                 "Score": float(mcq_table["Overall"].iloc[0])},
            ]

        if len(cloze_table):
            tidy_rows += [
                {"Model": model_name, "Task": "cloze", "Split": "-", "Metric": "EM",
                 "Score": float(cloze_table["EM"].iloc[0])},
                {"Model": model_name, "Task": "cloze", "Split": "-", "Metric": "PS",
                 "Score": float(cloze_table["PS"].iloc[0])},
            ]

        if len(ffq_bleu_table):
            tidy_rows += [
                {"Model": model_name, "Task": "ffq", "Split": "Base", "Metric": "Bleu",
                 "Score": float(ffq_bleu_table["Base: Bleu"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Reasoning", "Metric": "Bleu",
                 "Score": float(ffq_bleu_table["Reasoning: Bleu"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Hypothetical", "Metric": "Bleu",
                 "Score": float(ffq_bleu_table["Hypothetical: Bleu"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Overall", "Metric": "Bleu",
                 "Score": float(ffq_bleu_table["Overall: Bleu"].iloc[0])},
            ]
        if len(ffq_bert_table):
            tidy_rows += [
                {"Model": model_name, "Task": "ffq", "Split": "Base", "Metric": "Bert",
                 "Score": float(ffq_bert_table["Base: Bert"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Reasoning", "Metric": "Bert",
                 "Score": float(ffq_bert_table["Reasoning: Bert"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Hypothetical", "Metric": "Bert",
                 "Score": float(ffq_bert_table["Hypothetical: Bert"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Overall", "Metric": "Bert",
                 "Score": float(ffq_bert_table["Overall: Bert"].iloc[0])},
            ]
        if len(ffq_fa_table):
            tidy_rows += [
                {"Model": model_name, "Task": "ffq", "Split": "Base", "Metric": "FA",
                 "Score": float(ffq_fa_table["Base: FA"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Reasoning", "Metric": "FA",
                 "Score": float(ffq_fa_table["Reasoning: FA"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Hypothetical", "Metric": "FA",
                 "Score": float(ffq_fa_table["Hypothetical: FA"].iloc[0])},
                {"Model": model_name, "Task": "ffq", "Split": "Overall", "Metric": "FA",
                 "Score": float(ffq_fa_table["Overall: FA"].iloc[0])},
            ]

        tidy_df = pd.DataFrame(tidy_rows, columns=["Model", "Task", "Split", "Metric", "Score"])

        mcq_table.to_csv(mcq_tsv_path, sep="\t", index=False)
        cloze_table.to_csv(cloze_tsv_path, sep="\t", index=False)
        ffq_bleu_table.to_csv(ffq_bleu_tsv_path, sep="\t", index=False)
        ffq_bert_table.to_csv(ffq_bert_tsv_path, sep="\t", index=False)
        ffq_fa_table.to_csv(ffq_fa_tsv_path, sep="\t", index=False)
        if verbose:
            print(f"[evaluate] reports written: "
                  f"{os.path.basename(mcq_tsv_path)}, "
                  f"{os.path.basename(cloze_tsv_path)}, "
                  f"{os.path.basename(ffq_bleu_tsv_path)}, "
                  f"{os.path.basename(ffq_bert_tsv_path)}, "
                  f"{os.path.basename(ffq_fa_tsv_path)}")

        return tidy_df


