import json
import tempfile
import sys
import subprocess
import os
import pandas as pd
import re
from datasets import load_dataset
from .text_base import TextBaseDataset
from ..smp import *

class LiveCodeBench(TextBaseDataset):
    """
    LiveCodeBench Dataset Adapter for SciEvalKit.
    
    This dataset requires 'lcb_runner' to be installed.
    Install via: pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git
    """
    
    TYPE = "CODE"
    MODALITY = "TEXT"
    dataset_name = "LiveCodeBench"

    def __init__(
        self,
        dataset="LiveCodeBench",
        split="test",
        version="v6",
        prompt_style="codeqwen",
        **kwargs,
    ):
        self.split = split
        self.version = version
        self.prompt_style = str(prompt_style).lower()
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ["LiveCodeBench"]

    def load_data(self, dataset):
        from huggingface_hub import hf_hub_download
        
        # Mapping versions to files based on code_generation_lite.py
        files_map = {
            "release_v1": ["test.jsonl"],
            "release_v2": ["test.jsonl", "test2.jsonl"],
            "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
            "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
            "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
            "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
            "release_latest": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
            "v6": ["test6.jsonl"]
        }
        
        target_files = files_map.get(self.version)
        if not target_files:
             # Default fallback
             print(f"Warning: Unknown version {self.version}, defaulting to release_v6")
             target_files = files_map["release_v6"]

        dfs = []
        for file_name in target_files:
            try:
                # Download file from HF
                local_path = hf_hub_download(repo_id="livecodebench/code_generation_lite", filename=file_name, repo_type="dataset")
                with open(local_path, 'r') as f:
                    data = [json.loads(line) for line in f]
                    dfs.append(pd.DataFrame(data))
            except Exception as e:
                raise RuntimeError(f"Failed to download/load {file_name}: {e}")
                
        if not dfs:
             raise RuntimeError("Failed to load any data files.")
             
        df = pd.concat(dfs, ignore_index=True)
        
        
        # Standardize columns for SciEvalKit
        df['index'] = df.index
        if 'question_content' in df.columns:
            df['question'] = df['question_content']
        
        # Ensure other necessary columns are present if needed
        # TextBaseDataset mainly needs 'index' and 'question' for inference
        
        return df
        
        # Ensure we have required columns for TextBaseDataset
        # We map 'question_content' to 'question' which is picked up by build_prompt default if not overridden?
        # TextBaseDataset doesn't have a strict 'question' column requirement if build_prompt handles it,
        # but it's good practice.
        
        df = df.rename(columns={
            'question_id': 'id', # essential for matching
            'question_content': 'question'
        })
        
        # Add index if not present
        if 'index' not in df.columns:
            df['index'] = range(len(df))
            
        return df

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line["question"]
        starter_code = line.get("starter_code", "")

        formatting_with_starter = (
            "You will use the following starter code to write the solution to the problem "
            "and enclose your code within delimiters."
        )
        formatting_without_starter = (
            "Read the inputs from stdin solve the problem and write the answer to stdout "
            "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
            "Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
        )

        user_body = (
            "You will be given a question (problem specification) and will generate a correct Python program "
            "that matches the specification and passes all tests. \n\n"
            f"Question: {question}\n\n"
        )

        if starter_code:
            user_body += f"{formatting_with_starter}\n"
            user_body += f"```python\n{starter_code}\n```\n\n"
        else:
            user_body += f"{formatting_without_starter}\n"
            user_body += "```python\n# YOUR CODE HERE\n```\n\n"

        if self.prompt_style == "codeqwen":
            # Align with LiveCodeBench: LMStyle.CodeQwenInstruct
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n\n"
                f"{user_body}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif self.prompt_style == "qwq":
            prompt = (
                "<|im_start|>system\nYou are a helpful and harmless assistant. "
                "You are Qwen developed by Alibaba. You should think step-by-step.<|im_end|>\n"
                "<|im_start|>user\n\n"
                f"{user_body}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            prompt = user_body

        return [dict(type="text", value=prompt)]

    def extract_code(self, text):
        text = '' if text is None else str(text)
        if "```" in text:
            blocks = re.findall(r'```(?:[\w\+\.]*)\s*\n?(.*?)```', text, re.DOTALL)
            if blocks:
                # Use the last block as LiveCodeBench often suggests.
                return blocks[-1].strip()
        # Required behavior: if no fenced code block, code_list should be empty.
        return ''

    def evaluate(self, eval_file, **judge_kwargs):
        # Check dependency
        try:
            import lcb_runner
        except ImportError:
            raise ImportError(
                "LiveCodeBench runner is not installed. \n"
                "Please install it using: pip install git+https://github.com/LiveCodeBench/LiveCodeBench.git"
            )

        # Read the prediction file generated by SciEvalKit (usually Excel or CSV)
        # eval_file is the path to the prediction file
        
        logger = get_logger('LiveCodeBench_Eval')
        
        if eval_file.endswith('.xlsx'):
            df = pd.read_excel(eval_file)
        elif eval_file.endswith('.csv'):
            df = pd.read_csv(eval_file)
        else:
            # Fallback or error
            logger.error(f"Unsupported file format for evaluation: {eval_file}")
            return {}

        # Check for empty predictions
        if 'prediction' not in df.columns:
            logger.error("Prediction file missing 'prediction' column")
            return {}
            
        records = []
        raw_output_by_qid = {}
        code_by_qid = {}
        for _, row in df.iterrows():
            # Ensure question_id matches what LCB expects
            # We stored 'id' in load_data, checking if it is preserved in output
            # If SciEvalKit preserves inputs, 'id' should be there. 
            
            # Prefer 'question_id' if available (standard LCB column), else 'id'
            if 'question_id' in row:
                q_id = str(row['question_id'])
            else:
                q_id = str(row.get('id', ''))
            
            raw_pred = row['prediction']
            if pd.isna(raw_pred):
                raw_pred = ''
            else:
                raw_pred = str(raw_pred)
            code = self.extract_code(raw_pred)
            raw_output_by_qid[q_id] = raw_pred
            code_by_qid[q_id] = code
            
            # LCB expects list of codes (for pass@k)
            records.append({
                "question_id": q_id,
                "code_list": [code]
            })

        # Process evaluation
        # We will write to a temp file and call lcb_runner
        
        # Create temp directory in outputs to avoid permissions issues/cleanup issues
        # Or just use tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_f:
            json.dump(records, tmp_f)
            tmp_path = tmp_f.name
            
        logger.info(f"Prepared LCB evaluation data at {tmp_path}")
        
        # Construct command
        # python -m lcb_runner.runner.custom_evaluator --custom_output_file {tmp_path}
        
        cmd = [
            sys.executable, "-m", "lcb_runner.runner.custom_evaluator",
            "--custom_output_file", tmp_path,
            "--num_process_evaluate", str(judge_kwargs.get('nproc', 4)),
            "--release_version", self.version
        ]
        
        try:
            # Dynamically find where lcb_runner is installed
            import lcb_runner
            
            # Determine path safely, handling valid packages and namespace packages
            if hasattr(lcb_runner, '__file__') and lcb_runner.__file__:
                # Standard package: .../site-packages/lcb_runner/__init__.py -> .../site-packages
                lcb_path = os.path.dirname(os.path.dirname(lcb_runner.__file__))
            elif hasattr(lcb_runner, '__path__'):
                # Namespace package or executable zip: .../lcb_runner
                paths = list(lcb_runner.__path__)
                if paths:
                    lcb_path = os.path.dirname(paths[0])
                else:
                    lcb_path = os.getcwd()
            else:
                lcb_path = os.getcwd()

            logger.info(f"Running LiveCodeBench Evaluator from detected install path: {lcb_path}")
            
            # Run with cwd set to the package installation directory so it can find 'lcb_runner/prompts/...'
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=lcb_path)
            output = result.stdout
            logger.info("LCB Evaluator finished successfully.")

            def _rewrite_output_fields(path):
                if not os.path.exists(path):
                    return False
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    if not isinstance(data, list):
                        return False
                    for item in data:
                        if not isinstance(item, dict):
                            continue
                        qid = str(item.get('question_id', ''))
                        raw_text = raw_output_by_qid.get(qid, '')
                        code_text = code_by_qid.get(qid, '')
                        item['output_list'] = [raw_text]
                        item['code_list'] = [code_text]
                    with open(path, 'w') as f:
                        json.dump(data, f, indent=4)
                    return True
                except Exception as e:
                    logger.warning(f"Failed rewriting output fields for {path}: {e}")
                    return False
            
            # Parse output from the generated JSON file
            # The evaluator saves to [tmp_path without .json]_codegeneration_output_eval.json
            base_path = tmp_path.rsplit('.json', 1)[0]
            output_json_path = f"{base_path}_codegeneration_output.json"
            eval_path = f"{base_path}_codegeneration_output_eval.json"
            eval_all_path = f"{base_path}_codegeneration_output_eval_all.json"

            # Keep output_list as raw model outputs and code_list as extracted code only.
            _rewrite_output_fields(output_json_path)
            _rewrite_output_fields(eval_all_path)
            fixed_root = "/mnt/shared-storage-gpfs2/sciprismax2/guoxingjian/SciEvalKit/output_lcb/output.json"
            _rewrite_output_fields(fixed_root)
            _rewrite_output_fields(fixed_root.replace(".json", "_eval_all.json"))
            # shutil.copy(eval_path, "/mnt/shared-storage-gpfs2/sciprismax2/guoxingjian")
            #modified

            metrics = {}
            if os.path.exists(eval_path):
                try:
                    with open(eval_path, 'r') as f:
                        metrics_data = json.load(f)
                        if isinstance(metrics_data, list) and len(metrics_data) > 0:
                            metrics = metrics_data[0]
                except Exception as e:
                    logger.error(f"Failed to read evaluation results from {eval_path}: {e}")
            else:
                 logger.warning(f"Evaluation result file not found at {eval_path}")

            # Save metrics to score file (CSV)
            if metrics:
                score_file = eval_file.rsplit('.', 1)[0] + '_score.csv'
                acc = metrics.get("pass@1", 0) * 100
                score_df = pd.DataFrame([{"accuracy": acc, "split": "test", "version": self.version}])
                score_df.to_csv(score_file, index=False)
                logger.info(f"Saved evaluation score to {score_file}")
            
            # Record full log
            return {
                "accuracy": metrics.get("pass@1", 0) * 100, # Convert to percentage if needed, or keep as is. Usually LCB is 0-1.
                "full_metrics": metrics,
                "log": output
            }

        except subprocess.CalledProcessError as e:
            logger.error(f"LCB Evaluator failed: {e.stderr}")
            return {"error": str(e)}
        except Exception as e:
             logger.error(f"Error during LCB evaluation: {e}")
             return {"error": str(e)}
        finally:
             if os.path.exists(tmp_path):
                 os.remove(tmp_path)
