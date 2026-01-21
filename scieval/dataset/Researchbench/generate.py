# -*- coding: utf-8 -*-
import time
from typing import Any
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
import requests
from scieval.smp import *
import os
import sys
import json
from ..text_base import TextBaseDataset
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import csv, json
import pandas as pd
from scieval.dataset.utils.judge_util import *

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.append(_PROJ_ROOT)

from .utils import (
    instruction_prompts,
    llm_generation,
    llm_generation_while_loop,
    get_structured_generation_from_raw_generation,
    pick_score,
    if_element_in_list_with_similarity_threshold,
    save_with_json,
)


@dataclass
class GenerateConfig:
    module_name: str
    chem_annotation_path: Optional[str] = None
    if_use_strict_survey_question: int = 1
    if_use_background_survey: int = 1

    inspiration_path: Optional[str] = None
    idx_round_of_first_step_insp_screening: int = 0
    title_abstract_collector_path: Optional[str] = None

    model_name: str = "gpt-4o-mini"
    api_type: int = 0
    temperature: float = 1.0

    num_itr_self_refine: int = 2 
    max_inspiration_steps: int = 0  
    do_recombination: bool = True
    do_self_eval: bool = True

    enable_eval: bool = False
    eval_mode: str = "hit_groundtruth_insp"
    eval_model_name: Optional[str] = None
    save_dir: str = "outputs/researchbench_generate"

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class OneCaseInput:
    bkg_question: str
    survey: str
    inspirations: List[Dict[str, str]] = field(default_factory=list)

    core_inspiration_title: Optional[str] = None
    additional_inspiration_titles: Optional[List[str]] = None

    preliminary_hypothesis: Optional[str] = None
    expert_feedbacks: Optional[str] = None
    extra_knowledge: Optional[str] = None
    hypotheses_from_other_inspirations: Optional[List[str]] = None
    core_hypothesis_from_core_inspiration: Optional[str] = None
    groundtruth_hypothesis: Optional[str] = None
    reasoning_process: Optional[str] = None
    note: Optional[str] = None


@dataclass
class OneCaseOutput:
    prompt: str
    raw_generation: str
    structured_generation: Optional[List[List[str]]] = None  # e.g., [['Hypothesis text', 'Reasoning Process text']]
    eval_summary: Optional[Dict[str, Any]] = None


class ResearchbenchGenerate(TextBaseDataset):
    NAME = 'ResearchbenchGenerate'
    dataset_name = 'ResearchbenchGenerate'
    MODALITY = 'TEXT'
    TYPE = 'GEN'

    DATASET_URL = {
        'ResearchbenchGenerate': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/ResearchbenchGenerate.tsv'
    }

    DATASET_MD5 = {
        'ResearchbenchGenerate': '5ad4204ac8e413a151902015f0acb250'
    }


    def __init__(self, dataset: str = 'ResearchbenchGenerate', ann_path: str | None = None,
                 module_name: str = 'coarse_hypothesis_generation', model_name: str = 'gpt-4o-mini', api_type: int = 0,
                 temperature: float = 1.0, save_dir: str = 'outputs/researchbench_generate', **kwargs):
        self.cfg = GenerateConfig(
            module_name=module_name,
            chem_annotation_path=None,
            inspiration_path=None,
            model_name=model_name,
            api_type=api_type,
            temperature=temperature,
            save_dir=save_dir
        )
        self.dataset = dataset
        self.dataset_name = dataset
        self.client = None
        self.prepare_tsv(self.DATASET_URL[dataset],self.DATASET_MD5[dataset])
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = self.DATASET_URL[self.dataset_name].split('/')[-1]
        data_path = osp.join(data_root, file_name)
        self.ann_path = ann_path if ann_path is not None else data_path
        os.makedirs(self.cfg.save_dir, exist_ok=True)

        self._dump_dir = os.path.join(self.cfg.save_dir, "dump_images")
        os.makedirs(self._dump_dir, exist_ok=True)

        self._cases: list[OneCaseInput] = self._load_from_tsv(self.ann_path)

        self.bkg_list = [c.bkg_question for c in self._cases]
        self.dict_bkg2survey = {c.bkg_question: c.survey for c in self._cases}
        self.dict_bkg2groundtruthHyp = {
            c.bkg_question: c.groundtruth_hypothesis
            for c in self._cases if c.groundtruth_hypothesis
        }

        self._materialize_dataframe()


    def _materialize_dataframe(self):
        rows = []
        for idx, x in enumerate(self._cases):
            q = self.build_prompt(x)
            rows.append({
                "index": idx,
                "dataset": self.dataset_name,
                "question": q,
                "image_path": None,
                "bkg_question": x.bkg_question,
                "survey": x.survey,
                "core_inspiration_title": x.core_inspiration_title,
                "additional_inspiration_titles": x.additional_inspiration_titles,
                "groundtruth_hypothesis": x.groundtruth_hypothesis,
                "reasoning_process": x.reasoning_process,
                "note": x.note
            })
        self.data = pd.DataFrame(rows)

    MODALITY = 'TEXT'

    def __post_init_minimal(self):
        if hasattr(self, '_cases_inited') and self._cases_inited:
            return
        self._cases_inited = True
        self._cases = []

        ann_path = getattr(self.cfg, 'chem_annotation_path', None) or getattr(self, 'ann_path', None)
        if self.cfg.chem_annotation_path:
            if len(self.bkg_list) == 0:
                self._load_chem_annotation()
            select_bkgs = getattr(self.cfg, 'select_bkgs', None)
            self._cases = self.build_cases_from_annotation(select_bkgs=select_bkgs)
        else:
            if ann_path is None:
                raise ValueError("No input specified: set cfg.chem_annotation_path or ann_path.")
            import csv
            with open(ann_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    self._cases.append(OneCaseInput(
                        bkg_question=(row.get('bkg_question') or '').strip(),
                        survey=(row.get('survey') or '').strip(),
                        core_inspiration_title=(row.get('core_insp') or None),
                        additional_inspiration_titles=[s.strip() for s in (row.get('add_insp') or '').split('||') if s.strip()] or None,
                        groundtruth_hypothesis=(row.get('gt_hyp') or None),
                        reasoning_process=(row.get('reasoning') or None),
                        note=(row.get('note') or None),
                    ))


    def __getitem__(self, idx: int):
        x = self._cases[idx]
        q = self.build_prompt(x)
        return {
            "question": q,
            "image_path": None,
            "dataset": self.dataset_name,
            "index": idx,
            "meta": {
                "bkg_question": x.bkg_question,
                "core_inspiration_title": x.core_inspiration_title,
                "additional_inspiration_titles": x.additional_inspiration_titles,
            }
        }
    def dump_image(self, img_path: Optional[str], index: Optional[int] = None, *args, **kwargs) -> Optional[str]:
        if not img_path:
            return None
        return img_path
    def dump_result(self, results, save_dir=None, filename='pred.jsonl'):
        import os, json
        out_dir = save_dir or self.cfg.save_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            if isinstance(results, dict):
                for idx, ans in results.items():
                    item = self[idx]
                    f.write(json.dumps({
                        "index": idx,
                        "dataset": self.dataset_name,
                        "question": item["question"],
                        "answer": ans
                    }, ensure_ascii=False) + "\n")
            elif isinstance(results, list):
                for r in results:
                    idx = r.get("index")
                    ans = r.get("answer", r.get("prediction", r))
                    item = self[idx]
                    f.write(json.dumps({
                        "index": idx,
                        "dataset": self.dataset_name,
                        "question": item["question"],
                        "answer": ans
                    }, ensure_ascii=False) + "\n")
            else:
                raise TypeError(f"Unexpected results type: {type(results)}")
        print(f"[OK] Predictions saved to: {out_path}")
        return out_path

    def _load_from_tsv(self, path: str | None) -> list[OneCaseInput]:
        if not path:
            return []
        cases: list[OneCaseInput] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for r in reader:
                bkg = (r.get('rq') or r.get('bkg_question') or '').strip()
                survey = (r.get('survey') or '').strip()

                raw_ci = r.get('core_inspirations') or '[]'
                try:
                    ci_list = json.loads(raw_ci)
                except Exception:
                    ci_list = []
                inspirations = []
                for it in ci_list:
                    t = (isinstance(it, dict) and it.get('title')) or ''
                    a = (isinstance(it, dict) and it.get('abstract')) or ''
                    t, a = (t or '').strip(), (a or '').strip()
                    if t:
                        inspirations.append({'title': t, 'abstract': a})
                core = None
                add_titles = None
                if inspirations:
                    core = f"Title: {inspirations[0]['title']}; Abstract: {inspirations[0]['abstract'] or 'N/A'}."
                    add_titles = [x['title'] for x in inspirations[1:]] or None
                cases.append(OneCaseInput(
                    bkg_question=bkg,
                    survey=survey,
                    inspirations=inspirations,
                    core_inspiration_title=core,
                    additional_inspiration_titles=add_titles,
                    groundtruth_hypothesis=(r.get('gold_hypothesis') or None),
                    reasoning_process=(r.get('reasoning_process') or None),
                    note=(r.get('note') or None),
                ))
        return cases

    def _row_to_case(self, row: Any) -> OneCaseInput:
        getv = (row.get if isinstance(row, dict)
                else (lambda k, default=None: row[k] if (hasattr(row, '__contains__') and k in row and pd.notna(row[k])) else default))

        bkg = (getv('bkg_question') or getv('rq') or '').strip()
        survey = (getv('survey') or 'Survey not provided. Please overlook the survey.').strip()
        core = getv('core_inspiration_title')
        if isinstance(core, str):
            core_insp_title = core.strip() or None
        else:
            core_insp_title = None
        add_raw = getv('additional_inspiration_titles')
        if isinstance(add_raw, str):
            additional_titles = [s.strip() for s in add_raw.split('||') if s.strip()] or None
        elif isinstance(add_raw, list):
            additional_titles = add_raw or None
        else:
            additional_titles = None

        return OneCaseInput(
            bkg_question=bkg,
            survey=survey,
            inspirations=[],
            core_inspiration_title=core_insp_title,
            additional_inspiration_titles=additional_titles,
            groundtruth_hypothesis=(getv('groundtruth_hypothesis') or getv('gold_hypothesis') or None),
            reasoning_process=(getv('reasoning_process') or None),
            note=(getv('note') or None),
        )

    def build_prompt(self, x: Any) :
        if isinstance(x, OneCaseInput):
            # return [dict(type="text", value=self._build_prompt_from_case(x))]
            return self._build_prompt_from_case(x)
        try:
            if isinstance(x, dict):
                q = x.get('question')
                if isinstance(q, str) and q.strip():
                    return [dict(type="text", value=str(q))]
                    # return q
            else:
                if 'question' in x and isinstance(x['question'], str) and x['question'].strip():
                    return [dict(type="text", value=str(x['question']))]
                    # return x['question']
        except Exception:
            pass
        try:
            tmp_case = self._row_to_case(x)
            return [dict(type="text", value=str(self._build_prompt_from_case(tmp_case)))]
            # return self._build_prompt_from_case(tmp_case)
        except Exception:
            return [dict(type="text", value="You are an expert researcher. Please propose a plausible hypothesis.\nBackground: Not provided.")]
            # return "You are an expert researcher. Please propose a plausible hypothesis.\nBackground: Not provided."
    def build_prompt_by_index(self, idx: int) -> str:
        return self._build_prompt_from_case(self._cases[idx])
    def _build_prompt_from_case(self, x: OneCaseInput) -> str:
        def _fmt_list(items: Optional[List[str]]) -> str:
            if not items:
                return "Not provided."
            return "\n".join([f"- {t}" for t in items])
        pmts = instruction_prompts(self.cfg.module_name)
        bkg = x.bkg_question.strip()
        survey = (x.survey or "Survey not provided. Please overlook the survey.").strip()
        core = x.core_inspiration_title or (
            f"Title: {x.inspirations[0]['title']}; Abstract: {x.inspirations[0]['abstract'] or 'N/A'}."
            if x.inspirations else "Not provided."
        )
        add_titles = x.additional_inspiration_titles or [it['title'] for it in x.inspirations[1:]] if x.inspirations else None
        text = ""
        text += pmts[0] + bkg
        text += pmts[1] + survey
        text += pmts[2] + core
        text += pmts[3] + _fmt_list(add_titles)
        text += pmts[4]
        return text
    def _normalize_model_for_eval(self, name: Optional[str]) -> str:
        if not name:
            return "gpt-4o-mini"
        alias = {
            "4omini": "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
            "gpt4o": "gpt-4o-2024-08-06",
            "gpt-4o": "gpt-4o-2024-08-06",
            "claude35sonnet": "claude-3-5-sonnet-20241022",
            "claude35S": "claude-3-5-sonnet-20240620",
        }
        return alias.get(name, name)

    def _chat_once(self, prompt: str, model, temperature: float = 0.2, max_retry: int = 3) -> str:
        """
        Args:
            prompt (str): The user prompt.
            model: The framework model instance (e.g., OpenAIWrapper).
            temperature (float): Sampling temperature.
            max_retry (int): (Optional) Retry count.
                             Note: Ideally, this should be set during model initialization (build_judge).
                             If BaseAPI supports runtime retry override, it will be used here.
        """
        inputs = [
            {
                "role": "system",
                "value": "You are a careful evaluator. Reply concisely."
            },
            {
                "role": "user",
                "type": "text",
                "value": prompt
            }
        ]

        try:
            response = model.generate(inputs, temperature=temperature, retry=max_retry)
            return response.strip()

        except Exception as e:
            return f"[EVAL_CALL_ERROR] {type(e).__name__}: {e}"

    def _mk_core_node(self, insp: Dict[str, str]) -> list[str]:
        title = insp.get('title', '').strip()
        abstract = insp.get('abstract', '').strip()
        reason = "" 
        return [title, reason, abstract]

    def _gen_with_optional_refine(self,
                                bkg_q: str,
                                survey: str,
                                core_node: list[str],
                                num_refine: int = 2,
                                recombination_type: int = 0,
                                this_mutation: Optional[str] = None,
                                other_mutations: Optional[list] = None,
                                ) -> tuple[str, str]:

        hyp, rsn, feedback = None, None, None
        for it in range(max(1, num_refine)):
            title, reason, abstract = core_node
            core_prompt = f"title: {title}; abstract: {abstract}."
            if recombination_type == 2:
                assert this_mutation is not None and other_mutations is not None and len(other_mutations) == 3
                # other_mutations: [insp_title, insp_abstract, hyp]
                other_prompt = f"The selected complementary inspiration has title: {other_mutations[0]}, and abstract: {other_mutations[1]}. This complementary inspiration can lead to the hypothesis: {other_mutations[2]}."
                if hyp is None and feedback is None:
                    pmts = instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration")
                    full = pmts[0] + bkg_q + pmts[1] + survey + pmts[2] + f"{core_prompt}" + pmts[3] + this_mutation + pmts[4] + other_prompt + pmts[5]
                    tmpl = ['Hypothesis:', 'Reasoning Process:']
                else:
                    pmts = instruction_prompts("final_recombinational_mutation_hyp_gene_between_diff_inspiration_with_feedback")
                    full = pmts[0] + bkg_q + pmts[1] + survey + pmts[2] + f"{core_prompt}" + pmts[3] + this_mutation + pmts[4] + other_prompt + pmts[5] + hyp + pmts[6] + feedback + pmts[7]
                    tmpl = ['Refined Hypothesis:', 'Reasoning Process:']
            else:
                if other_mutations is None and feedback is None:
                    pmts = instruction_prompts("coarse_hypothesis_generation_only_core_inspiration")
                    full = pmts[0] + bkg_q + pmts[1] + survey + pmts[2] + f"{core_prompt}" + pmts[3]
                    tmpl = ['Hypothesis:', 'Reasoning Process:']
                elif other_mutations is None and feedback is not None:
                    pmts = instruction_prompts("hypothesis_generation_with_feedback_only_core_inspiration")
                    full = pmts[0] + bkg_q + pmts[1] + survey + pmts[2] + f"{core_prompt}" + pmts[3] + hyp + pmts[4] + feedback + pmts[5]
                    tmpl = ['Refined Hypothesis:', 'Reasoning Process:']
                else:
                    pmts = instruction_prompts("hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration")
                    other_prompt = ""
                    for i, o in enumerate(other_mutations or []):
                        other_prompt += f"Next is previous hypothesis {i}: {o}.\n"
                    full = pmts[0] + bkg_q + pmts[1] + survey + pmts[2] + f"{core_prompt}" + pmts[3] + other_prompt + pmts[4]
                    tmpl = ['Hypothesis:', 'Reasoning Process:']
            raw = llm_generation(full, self.cfg.model_name, self.client, temperature=self.cfg.temperature, api_type=self.cfg.api_type)
            st = get_structured_generation_from_raw_generation(raw, template=tmpl)[0]
            hyp, rsn = st[0], st[1]
            feedback = self._feedback_on_hypothesis(hyp, rsn, consider_extra_knowledge=(it == 1))
        return hyp, rsn

    def _feedback_on_hypothesis(self, hyp: str, rsn: str, consider_extra_knowledge: bool=False) -> str:
        cur = f"hypothesis: {hyp}; reasoning process: {rsn}."
        if consider_extra_knowledge:
            pmts = instruction_prompts("four_aspects_checking_and_extra_knowledge")
        else:
            pmts = instruction_prompts("four_aspects_checking")
        full = pmts[0] + cur + pmts[1]
        return llm_generation(full, self.cfg.model_name, self.client, temperature=self.cfg.temperature, api_type=self.cfg.api_type)

    def _self_eval(self, hyp: str) -> tuple[list[int], list[str], float]:
        pmts = instruction_prompts("four_aspects_self_numerical_evaluation")
        full = pmts[0] + f"hypothesis: {hyp}." + pmts[1]
        raw = llm_generation(full, self.cfg.model_name, self.client, temperature=0.2, api_type=self.cfg.api_type)
        scores, reasons, ok = pick_score(raw, full)
        if not ok:
            return [0,0,0,0], ["","","",""], 0.0
        avg = sum(scores) / max(1, len(scores))
        return scores, reasons, avg

    def _evo_pipeline(self, x: OneCaseInput) -> dict:
        bkg = x.bkg_question.strip()
        survey = (x.survey or "Survey not provided. Please overlook the survey.").strip()
        steps_limit = self.cfg.max_inspiration_steps or len(x.inspirations)
        used = x.inspirations[:steps_limit]

        trace = []
        best_hyp, best_rsn, best_avg = None, None, -1.0
        if not used:
            raise ValueError("No inspirations available for evo pipeline.")
        core0 = self._mk_core_node(used[0])
        hyp1, rsn1 = self._gen_with_optional_refine(bkg, survey, core0, num_refine=self.cfg.num_itr_self_refine, recombination_type=0)
        s1, r1, a1 = (self._self_eval(hyp1) if self.cfg.do_self_eval else ([0,0,0,0], ["","","",""], 0.0))
        best_hyp, best_rsn, best_avg = hyp1, rsn1, a1
        trace.append({
            "step": 1, "type": "single", "insp_title": used[0]['title'],
            "hypothesis": hyp1, "reasoning": rsn1, "scores": s1, "avg": a1
        })
        for i in range(1, len(used)):
            ci = self._mk_core_node(used[i])
            h_i, r_i = self._gen_with_optional_refine(bkg, survey, ci, num_refine=max(1, self.cfg.num_itr_self_refine-1), recombination_type=0)
            if self.cfg.do_recombination and best_hyp:
                other = [ci[0], ci[2], h_i]  # [title, abstract, hyp_i]
                h_rec, r_rec = self._gen_with_optional_refine(bkg, survey, ci,
                                                            num_refine=1,
                                                            recombination_type=2,
                                                            this_mutation=best_hyp,
                                                            other_mutations=other)
                s_rec, r_rec_scores, a_rec = (self._self_eval(h_rec) if self.cfg.do_self_eval else ([0,0,0,0], ["","","",""], 0.0))
                if a_rec >= best_avg:
                    best_hyp, best_rsn, best_avg = h_rec, r_rec, a_rec
                    trace.append({
                        "step": i+1, "type": "recombine", "insp_title": used[i]['title'],
                        "hypothesis": h_rec, "reasoning": r_rec, "scores": s_rec, "avg": a_rec
                    })
                else:
                    s_i, r_i_scores, a_i = (self._self_eval(h_i) if self.cfg.do_self_eval else ([0,0,0,0], ["","","",""], 0.0))
                    trace.append({
                        "step": i+1, "type": "single", "insp_title": used[i]['title'],
                        "hypothesis": h_i, "reasoning": r_i, "scores": s_i, "avg": a_i
                    })
            else:
                s_i, r_i_scores, a_i = (self._self_eval(h_i) if self.cfg.do_self_eval else ([0,0,0,0], ["","","",""], 0.0))
                if a_i >= best_avg:
                    best_hyp, best_rsn, best_avg = h_i, r_i, a_i
                trace.append({
                    "step": i+1, "type": "single", "insp_title": used[i]['title'],
                    "hypothesis": h_i, "reasoning": r_i, "scores": s_i, "avg": a_i
                })

        final_scores, final_reasons, final_avg = (self._self_eval(best_hyp) if self.cfg.do_self_eval else ([0,0,0,0], ["","","",""], best_avg))
        return {
            "final_hypothesis": best_hyp,
            "final_reasoning": best_rsn,
            "final_scores": final_scores,
            "final_avg": final_avg,
            "trace": trace
        }
    def _expected_template(self) -> Optional[List[str]]:
        return ['Hypothesis:', 'Reasoning Process:']

    def generate_one(self, x: OneCaseInput) -> OneCaseOutput:
        if x.inspirations:
            evo = self._evo_pipeline(x)
            raw_gene = f"Hypothesis: {evo['final_hypothesis']}\nReasoning Process: {evo['final_reasoning']}"
            return OneCaseOutput(
                prompt="(pipeline)",
                raw_generation=raw_gene,
                structured_generation=[[evo['final_hypothesis'], evo['final_reasoning']]],
                eval_summary={"judge_scores": evo["final_scores"], "judge_avg": evo["final_avg"], "trace": evo["trace"]}
            )
        prompt = self.build_prompt(x)
        template = self._expected_template()
        try:
            if template is not None:
                structured = llm_generation_while_loop(
                    prompt=prompt,
                    model_name=self.cfg.model_name,
                    client=self.client,
                    if_structured_generation=True,
                    template=template,
                    if_only_return_one_structured_gene_component=True,
                    temperature=self.cfg.temperature,
                    api_type=self.cfg.api_type,
                )
                raw_gene = f"{template[0]} {structured[0]}\n{template[1]} {structured[1]}"
                return OneCaseOutput(prompt=prompt, raw_generation=raw_gene, structured_generation=[structured])
            else:
                raw_gene = llm_generation(prompt=prompt, model_name=self.cfg.model_name, client=self.client, temperature=self.cfg.temperature, api_type=self.cfg.api_type)
                return OneCaseOutput(prompt=prompt, raw_generation=raw_gene, structured_generation=None)
        except Exception as e:
            print(f"[WARN] Structured generation failed: {repr(e)}. Fallback to raw + parse.")
            raw_gene = llm_generation(prompt=prompt, model_name=self.cfg.model_name, client=self.client, temperature=self.cfg.temperature, api_type=self.cfg.api_type)
            parsed = None
            if template is not None:
                try:
                    parsed = get_structured_generation_from_raw_generation(raw_gene, template)
                except Exception as e2:
                    print(f"[WARN] Parsing raw generation failed: {repr(e2)}")
            return OneCaseOutput(prompt=prompt, raw_generation=raw_gene, structured_generation=parsed)

    def _resolve_eval_file(self, expected_path: str) -> str:
        import os, glob
        if os.path.exists(expected_path):
            return expected_path
        d = os.path.dirname(expected_path)
        fname = os.path.basename(expected_path)
        stem, ext = os.path.splitext(fname)
        if ext.lower() not in (".xlsx", ".xls"):
            return expected_path
        base_try = os.path.join(d, f"{stem.rsplit('_', 1)[0]}.xlsx") if '_' in stem else None
        cand_list = []

        if base_try and os.path.exists(base_try):
            cand_list.append(base_try)
        prefix = None
        try:
            ds_token = f"_{self.dataset_name}"
            if ds_token in stem:
                prefix = stem.split(ds_token)[0]
        except Exception:
            prefix = None
        if prefix:
            cand2 = os.path.join(d, f"{prefix}_{self.NAME}.xlsx")
            if os.path.exists(cand2):
                cand_list.append(cand2)
        if not cand_list:
            xp = sorted(
                glob.glob(os.path.join(d, "*.xlsx")),
                key=lambda p: os.path.getmtime(p),
                reverse=True
            )
            if len(xp):
                cand_list.append(xp[0])
        return cand_list[0] if cand_list else expected_path

    def evaluate_one(self, x: OneCaseInput, y: OneCaseOutput) -> Dict[str, Any]:
        eval_summary: Dict[str, Any] = {"mode": self.cfg.eval_mode}
        hyp_text = y.raw_generation
        if y.structured_generation and len(y.structured_generation) > 0:
            # [['Hypothesis', 'Reasoning Process']]
            hyp_text = y.structured_generation[0][0] or y.raw_generation

        if self.cfg.eval_mode == "hit_groundtruth_insp":
            gd_insp = self.dict_bkg2insp.get(x.bkg_question, [])
            hits = []
            for t in gd_insp:
                hit = if_element_in_list_with_similarity_threshold([hyp_text], t, threshold=0.35)
                hits.append(int(hit))
            eval_summary["groundtruth_inspirations"] = gd_insp
            eval_summary["hits"] = hits
            eval_summary["hit_ratio"] = sum(hits) / max(1, len(hits))
            return eval_summary

        elif self.cfg.eval_mode == "llm_matched_score":
            gt = x.groundtruth_hypothesis or ""
            if not gt:
                eval_summary["error"] = "No groundtruth hypothesis for LLM scoring."
                return eval_summary
            keypoints_text = ""
            if x.note and "key" in x.note.lower():
                keypoints_text = x.note
            elif x.reasoning_process:
                keypoints_text = x.reasoning_process
            else:
                keypoints_text = "Not provided."

            pmts = instruction_prompts("eval_matched_score")
            eval_prompt = pmts[0] + hyp_text + pmts[1] + gt + pmts[2] + keypoints_text + pmts[3]
            eval_model = self.cfg.eval_model_name or self.cfg.model_name
            eval_raw = llm_generation(
                prompt=eval_prompt,
                model_name=eval_model,
                client=self.client,
                temperature=0.2,
                api_type=self.cfg.api_type
            )
            score = None
            for s in ["5", "4", "3", "2", "1", "0"]:
                token = f"Matched score: {s}"
                if token in eval_raw:
                    score = int(s)
                    break
            eval_summary["eval_raw"] = eval_raw
            eval_summary["matched_score"] = score
            return eval_summary

        else:
            eval_summary["error"] = f"Unknown eval_mode: {self.cfg.eval_mode}"
            return eval_summary

    @staticmethod
    def _extract_hypothesis_from_text(txt: str) -> str:
        import re
        if not isinstance(txt, str):
            return ""
        m = re.search(r'(?is)hypothesis\s*:\s*(.*?)(?:\n\s*reasoning\s*process\s*:|$)', txt)
        if m:
            return m.group(1).strip()
        return txt.strip()
    
    @staticmethod
    def _parse_judge_score(judge_text: str):
        import re
        if not isinstance(judge_text, str):
            return None
        m = re.search(r'(?i)matched\s*score\s*:\s*([0-5])', judge_text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        m2 = re.search(r'(?<!\d)([0-5])(?!\d)', judge_text)
        if m2:
            try:
                return int(m2.group(1))
            except Exception:
                pass
        return None
    
    def evaluate(self, eval_file: str, **judge_kwargs):
        import os
        import pandas as pd
        from pathlib import Path
        real_eval_file = self._resolve_eval_file(eval_file)
        if not os.path.exists(real_eval_file):
            raise FileNotFoundError(f"evaluate() {eval_file}")
        ext = os.path.splitext(real_eval_file)[1].lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(real_eval_file)
        elif ext == ".csv":
            df = pd.read_csv(real_eval_file)
        elif ext == ".tsv":
            df = pd.read_csv(real_eval_file, sep="\t")
        else:
            raise ValueError(f"{ext}")
        def pick_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            import re
            normed = { re.sub(r'\s+', '', str(col)).strip().lower(): col for col in df.columns }
            for c in cands:
                key = re.sub(r'\s+', '', c).strip().lower()
                if key in normed:
                    return normed[key]
            return None
        col_idx = pick_col(["index", "idx"])
        col_pred = pick_col(["prediction", "answer", "pred", "model_output"])
        col_gt = pick_col(["groundtruth_hypothesis", "gold_hypothesis", "gt_hypothesis", "gt"])
        if col_pred is None:
            raise ValueError("")
        if col_gt is None or df[col_gt].isna().all():
            if hasattr(self, "data") and isinstance(self.data, pd.DataFrame):
                if col_idx is None and "index" in df.columns:
                    col_idx = "index"
                if col_idx is None:
                    print("[WARN]")
                else:
                    base = self.data[["index", "groundtruth_hypothesis"]].rename(
                        columns={"groundtruth_hypothesis": "__gt__"}
                    )
                    df = df.merge(base, how="left", left_on=col_idx, right_on="index")
                    col_gt = "__gt__"
        eval_model = judge_kwargs.get("model", None) or self.cfg.eval_model_name or self.cfg.model_name
        eval_model = build_judge_model(model=self._normalize_model_for_eval(eval_model), **judge_kwargs)
        temperature = judge_kwargs.get("temperature", 0.2)
        pmts = instruction_prompts("eval_matched_score")
        pred_hyps, judge_scores, judge_logs = [], [], []
        for _, r in df.iterrows():
            pred_raw = r[col_pred]
            pred_h = self._extract_hypothesis_from_text(pred_raw if isinstance(pred_raw, str) else "")
            pred_hyps.append(pred_h)
            gt = (r[col_gt] if col_gt in df.columns else None)
            if not isinstance(gt, str) or not gt.strip():
                judge_scores.append(None)
                judge_logs.append("No groundtruth hypothesis; skipped.")
                continue
            keypoints_text = "Not provided."
            eval_prompt = pmts[0] + pred_h + pmts[1] + gt.strip() + pmts[2] + keypoints_text + pmts[3]
            eval_raw = self._chat_once(eval_prompt, model=eval_model, temperature=temperature, max_retry=3)
            score = self._parse_judge_score(eval_raw)
            judge_scores.append(score)
            judge_logs.append(eval_raw)
        df["pred_hypothesis"] = pred_hyps
        df["judge_score"] = judge_scores
        df["judge_log"] = judge_logs
        valid = [s for s in judge_scores if isinstance(s, (int, float))]
        avg_score = float(sum(valid) / len(valid)) if valid else None
        dist = {i: int(sum(1 for s in valid if s == i)) for i in range(6)} if valid else {i:0 for i in range(6)}
        p = Path(real_eval_file)
        judged_path = str(p.with_name(p.stem + "_judged.xlsx"))
        df.to_excel(judged_path, index=False)
        print(f"[OK] LLM Judge Completed, AVG Score: {avg_score if avg_score is not None else 'N/A'}, Saved at: {judged_path}")
        out_dir = os.path.dirname(eval_file)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(eval_file))[0]
        result_file_path = os.path.join(out_dir, f"{base}_eval.json")
        eval_result = {
            "items_scored": len(valid),
            "avg_score": avg_score/5,
            "score_dist": dist,
            "judged_file": judged_path,
            "judge_model": eval_model.model
        }
        # 将字典数据写入 JSON 文件
        with open(result_file_path, "w") as f:
            json.dump(eval_result, f, indent=4)
        return eval_result

    def run(self, cases: List[OneCaseInput]) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for x in cases:
            out = self.generate_one(x)
            if self.cfg.enable_eval:
                out.eval_summary = self.evaluate_one(x, out)
            results[x.bkg_question] = {
                "prompt": out.prompt,
                "raw_generation": out.raw_generation,
                "structured_generation": out.structured_generation,
                "eval": out.eval_summary,
                "context": {
                    "survey": x.survey,
                    "core_inspiration_title": x.core_inspiration_title,
                    "additional_inspiration_titles": x.additional_inspiration_titles,
                    "groundtruth_hypothesis": x.groundtruth_hypothesis,
                    "reasoning_process": x.reasoning_process,
                    "note": x.note
                }
            }
        return results

    def save(self, results: Dict[str, Dict[str, Any]], filename: str) -> str:
        out_path = os.path.join(self.cfg.save_dir, filename)
        save_with_json(results, out_path)
        print(f"[OK] saved generate results to: {out_path}")
        return out_path

if __name__ == "__main__":
    try:
        from openai import OpenAI
        _client = OpenAI()
    except Exception:
        _client = None
        print("[WARN] OpenAI client not initialized. This __main__ block is for local dry-run only.")

    cfg = GenerateConfig(
        module_name="coarse_hypothesis_generation",
        chem_annotation_path=None,
        inspiration_path=None,
        enable_eval=False,
        save_dir="outputs/researchbench_generate"
    )

    runner = ResearchbenchGenerate(cfg, _client)

    demo = OneCaseInput(
        bkg_question="How to improve sample efficiency of RL for complex manipulation?",
        survey="Prior methods: behavior cloning + offline RL; model-based RL; world models.",
        core_inspiration_title="Backpropagation through differentiable physics engine",
        additional_inspiration_titles=[
            "Hierarchical latent skill discovery",
            "Contrastive policy regularization"
        ],
        groundtruth_hypothesis=None
    )

    res = runner.run([demo])
    runner.save(res, "demo_generate.json")
