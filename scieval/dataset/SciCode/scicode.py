from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd
from datasets import load_dataset

from scieval.smp import load

from ..text_base import TextBaseDataset
from .protocol import (
    OFFICIAL_EXCLUDED_TEST_STEPS,
    is_official_scored_step,
    make_step_id,
    official_dependency_code,
    extract_python_script,
)


class SciCode(TextBaseDataset):
    """SciCode benchmark.

    Each *sub-step* of a SciCode problem is exposed as one evaluation item.
    - `build_prompt` reproduces the official prompt construction.
    - `evaluate` writes predictions to `{prob}.{step}.py` and calls the
      original SciCode tester; files are stored under the same layout that the
      upstream scripts expect.

    Parameters
    ----------
    split : {"validation","test"}, default "test"
        Which split to load from the HF dataset `SciCode1/SciCode`.
    with_background : bool, default True
        Whether to include the optional scientific background in prompts.
    """

    TYPE = "TEXT"
    MODALITY = "TEXT"
    dataset_name = "SciCode"
    # The output of an earlier sub-step is part of every later prompt in the
    # same problem.  ``scieval.inference`` uses this marker to schedule steps
    # in dependency order instead of flattening all 288 requests at once.
    SEQUENTIAL_INFERENCE = True

    def __init__(
        self,
        split: str = "test",
        with_background: bool = True,
        dataset: str = "SciCode",
        **kwargs,
    ) -> None:
        # Save flags first; TextBaseDataset.__init__ will call self.load_data()
        self.split = split
        self.with_background = with_background
        super().__init__(dataset="SciCode", **kwargs)

        # Select the template file (match upstream gencode logic):
        # with_background=True  -> use multistep_template.txt
        # with_background=False -> use background_comment_template.txt
        tmpl_name = (
            "multistep_template.txt"
            if with_background
            else "background_comment_template.txt"
        )
        template_file = Path(__file__).resolve().parent / "eval" / tmpl_name
        if not template_file.is_file():
            raise FileNotFoundError(
                f"Cannot find prompt template: {template_file}. Please place SciCode templates under 'vlmeval/dataset/SciCode/eval/'."
            )
        self.prompt_template = template_file.read_text(encoding="utf-8")

    @classmethod
    def supported_datasets(cls):
        return ["SciCode"]

    # ---- Data loading (called by TextBaseDataset.__init__) -----------------
    def load_data(self, dataset):
        """Load and flatten the SciCode HF dataset into a DataFrame.

        Returns a DataFrame with columns: index, id, problem_id, step, tot_steps, record
        """
        try:
            hf = load_dataset("SciCode1/SciCode", split=self.split)
        except Exception as err:
            raise RuntimeError(
                "Failed to load the SciCode dataset. Ensure 'datasets' is installed and you have network access. "
                f"Original error: {err}"
            )
        rows: List[Dict[str, Any]] = []
        idx = 0
        for prob in hf:
            pid = prob["problem_id"]
            subs = prob["sub_steps"]
            total = len(subs)
            for s_idx, _ in enumerate(subs):
                step_num = s_idx + 1
                if not is_official_scored_step(self.split, pid, step_num):
                    continue
                rows.append(
                    {
                        "index": idx,
                        "id": make_step_id(pid, step_num),
                        "problem_id": pid,
                        "step": step_num,
                        "tot_steps": total,
                        "record": prob,
                    }
                )
                idx += 1
        return pd.DataFrame(rows)

    # ---- Prompt construction ----------------------------------------------
    def build_prompt(self, line: pd.Series) -> List[Dict[str, str]]:
        """Construct a prompt without generated context.

        Generic callers use this entry point.  Real SciCode inference calls
        :meth:`build_prompt_with_context` so later steps receive the model code
        generated for earlier steps, matching the upstream/AA protocol.
        """

        return self.build_prompt_with_context(line, {})

    def build_prompt_with_context(
        self,
        line: pd.Series,
        previous_predictions: Mapping[int, object],
    ) -> List[Dict[str, str]]:
        """Construct one sub-step prompt with prior model implementations.

        ``previous_predictions`` is keyed by one-based step number and contains
        raw model responses.  The three unscored dependency-only test steps are
        filled from the scientist-curated implementations shipped by SciCode.
        """

        if isinstance(line, int):
            line = self.data.iloc[line]
        record = line["record"]
        step_idx = int(line["step"]) - 1

        # Previous descriptions and generated function code.  The separator
        # and ordering deliberately mirror upstream process_problem_steps().
        prev_lines: List[str] = []
        for i in range(step_idx):
            sub = record["sub_steps"][i]
            if self.with_background:
                previous_description = (
                    f"{sub['step_description_prompt']}\n{sub.get('step_background', '')}"
                )
            else:
                previous_description = sub["step_description_prompt"]

            one_based_step = i + 1
            step_id = make_step_id(record["problem_id"], one_based_step)
            if (
                self.split == "test"
                and step_id in OFFICIAL_EXCLUDED_TEST_STEPS
            ):
                previous_code = official_dependency_code(step_id)
            else:
                previous_code = extract_python_script(
                    previous_predictions.get(one_based_step)
                )

            prev_lines.extend([previous_description, previous_code, "------"])
        problem_steps_str = "\n\n".join(prev_lines[:-1]) if prev_lines else ""

        # Next step description + header/return stub
        cur = record["sub_steps"][step_idx]
        if self.with_background:
            next_desc = (
                f"{cur['step_description_prompt']}\n{cur.get('step_background', '')}"
            )
        else:
            next_desc = cur["step_description_prompt"]
        function_code_stub = f"{cur['function_header']}\n\n{cur.get('return_line', '')}"
        next_step_str = f"{next_desc}\n\n{function_code_stub}"

        # Dependencies (list or str)
        deps = record.get("required_dependencies", "")
        # print("===== Dependencies =====\n", deps)
        if isinstance(deps, list):
            deps = "\n".join([str(x) for x in deps])
        deps = str(deps)

        prompt = self.prompt_template.format(
            problem_steps_str=problem_steps_str,
            next_step_str=next_step_str,
            dependencies=deps,
        )
        """
        # DEBUG: print every sub-step prompt by default (remove later if noisy)
        try:
            pid = str(line.get("problem_id"))
            step = int(line.get("step"))
            tot = int(line.get("tot_steps"))
            uid = str(line.get("id", f"{pid}.{step}"))
            print("\n===== [SciCode Prompt] =====")
            print(f"ID: {uid} | problem_id: {pid} | step: {step}/{tot} | with_background={self.with_background} | split={self.split}")
            print(prompt)
            print("===== [End Prompt] =====\n")
        except Exception:
            pass
        """

        return [dict(type="text", value=prompt)]

    # ---- Evaluation --------------------------------------------------------
    def evaluate(self, eval_file: str, **judge_kwargs) -> Dict[str, Any]:
        """Write predictions to files and run the original SciCode tester.

        Side effects (file outputs):
          - Generated code: eval_results/generated_code/{model}/{with|without}_background/{prob}.{step}.py
          - Logs (pass/fail/timeout cache): logs/{model}/{with|without}_background/{step_id}.txt
          - Summary metrics: eval_results/{model}_{with|without}_background.{txt,json}
        """
        # 1) Load predictions
        pred_df = load(eval_file)
        if isinstance(pred_df, (dict, list)):
            pred_df = pd.DataFrame(pred_df)
        id_col = None
        for c in ("id", "question_id"):
            if c in pred_df.columns:
                id_col = c
                break
        if id_col is None:
            raise KeyError(
                "The evaluation file must contain an 'id' or 'question_id' column."
            )
        # Build preds dict, rebuilding the unique step ID from problem_id + step
        # columns when available. The xlsx writer coerces decimal-looking strings
        # like "11.10" into floats (11.1), which collides with the genuine "11.1"
        # row and silently drops step-10 predictions for problems with >=10 steps.
        # See https://github.com/InternScience/SciEvalKit issue (SciCode 11.10/12.10/etc).
        has_pid = "problem_id" in pred_df.columns
        has_step = "step" in pred_df.columns
        preds: Dict[str, str] = {}
        for _, row in pred_df.iterrows():
            uid: str
            if has_pid and has_step and pd.notna(row["problem_id"]) and pd.notna(row["step"]):
                uid = f"{int(row['problem_id'])}.{int(row['step'])}"
            else:
                uid = str(row[id_col])
            preds[uid] = str(row["prediction"])

        # 2) Prepare paths (generated_code & logs INSIDE model folder)
        model_stub = str(
            judge_kwargs.get("eval_model_name") or judge_kwargs.get("model") or "model"
        )
        work_base = Path(judge_kwargs.get("work_dir", "outputs"))

        root_dir = work_base / model_stub  # Kimi-k2/
        code_dir = root_dir / "generated_code"  # Kimi-k2/generated_code/
        logs_dir = root_dir / "logs"  # Kimi-k2/logs/
        out_dir = root_dir  # summary files here
        bg_dir = "with_background" if self.with_background else "without_background"

        for base in (code_dir, logs_dir):
            (base / bg_dir).mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 3) Build helper maps from self.data for dependencies & grouping by problem
        #    - prob -> record (to fetch dependencies)
        prob_to_record: Dict[str, Any] = {}
        for _, row in self.data.iterrows():
            pid = str(row["problem_id"])
            if pid not in prob_to_record:
                prob_to_record[pid] = row["record"]

        def _clean(code: str) -> str:
            """Apply the same extraction used for multi-step prompt context."""

            if isinstance(code, str) and "Error:" in code:
                return ""
            return extract_python_script(code)

        # 4) Group predictions by problem and step (so we can prepend previous steps' code)
        prob_to_steps: Dict[str, Dict[int, str]] = {}
        for uid, code in preds.items():
            prob_id, step_str = uid.split(".")
            prob_to_steps.setdefault(prob_id, {})[int(step_str)] = _clean(code)

        # 5) Write files: deps + (prev steps code) + current step code
        for uid, _ in preds.items():
            prob_id, step_str = uid.split(".")
            step_num = int(step_str)
            if not is_official_scored_step(self.split, prob_id, step_num):
                continue
            record = prob_to_record.get(prob_id)

            pieces: List[str] = []

            # 0) Inject required_dependencies (safe lines only)
            raw_deps = record.get("required_dependencies", "")
            dep_lines: List[str] = []
            if isinstance(raw_deps, list):
                raw_deps = "\n".join([str(x) for x in raw_deps])
            for ln in str(raw_deps).splitlines():
                ln = ln.strip()
                # very simple heuristic: keep "import" & "from" lines, skip others
                if ln.startswith("import ") or ln.startswith("from "):
                    dep_lines.append(ln)
            if dep_lines:
                pieces.extend(dep_lines)

            # previous steps
            for s in range(1, step_num):
                prev_step_id = make_step_id(prob_id, s)
                if (
                    self.split == "test"
                    and prev_step_id in OFFICIAL_EXCLUDED_TEST_STEPS
                ):
                    # The official protocol does not ask the model to generate
                    # these three dependency-only steps.  It injects the
                    # scientist-curated implementation for downstream steps.
                    prev_code = official_dependency_code(prev_step_id)
                else:
                    prev_code = prob_to_steps.get(prob_id, {}).get(s)
                if prev_code:
                    pieces.append(prev_code)
            # current step
            cur_code = prob_to_steps.get(prob_id, {}).get(step_num, "")
            pieces.append(cur_code)

            out_file = code_dir / bg_dir / f"{prob_id}.{step_num}.py"
            out_file.write_text("\n".join(pieces), encoding="utf-8")

        # 6) Run the official tester (lazy import to avoid optional deps at import time)
        from .test_generated_code import test_code as _scicode_test_code  # type: ignore

        _scicode_test_code(
            model_name=model_stub,
            split=self.split,
            code_dir=code_dir,
            log_dir=logs_dir,
            output_dir=out_dir,
            with_background=self.with_background,
        )

        # Keep return minimal to match VLMEvalKit's expectation; metrics are in files under eval_results/.
        return {"scicode_test_invoked": True}
