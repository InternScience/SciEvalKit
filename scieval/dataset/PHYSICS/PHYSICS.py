import warnings

import numpy as np
from jinja2 import Template
import os
import json
from ..text_base import TextBaseDataset
from scieval.smp.file import load
from tqdm import tqdm
from .reward_score  import compute_score
import signal
from contextlib import contextmanager
from .reward_score import Model_args
import wrapt_timeout_decorator

from ... import gpt_key_set


def generate_score_report(jsonl_file_path: str):
    if not os.path.isfile(jsonl_file_path):
        print(f"错误: 文件未找到 -> {jsonl_file_path}")
        return

    input_directory = os.path.dirname(jsonl_file_path)
    input_filename = os.path.basename(jsonl_file_path)

    report_path = os.path.join(input_directory, "score_report.txt")

    total_records = 0
    acc_true_count = 0
    rule_based_acc_true_count = 0

    print(f"正在处理文件: {input_filename}...")

    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    total_records += 1

                    if data.get('acc') is True:
                        acc_true_count += 1

                    if data.get('rule_based_acc') is True:
                        rule_based_acc_true_count += 1

                except json.JSONDecodeError:
                    print(f"警告: 发现无效的JSON行，已跳过: {line.strip()}")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    if total_records > 0:
        acc_percentage = (acc_true_count / total_records) * 100
        rule_based_acc_percentage = (rule_based_acc_true_count / total_records) * 100
    else:
        acc_percentage = 0
        rule_based_acc_percentage = 0
        print("警告: 文件为空或不包含任何有效的JSON记录。")

    report_content = (
        f"Score Report for: {input_filename}\n"
        f"========================================\n"
        f"Total Records Processed: {total_records}\n\n"
        f"Accuracy ('acc'):\n"
        f"  - True Count: {acc_true_count}\n"
        f"  - Correctness Rate: {acc_percentage:.2f}%\n\n"
        f"Rule-Based Accuracy ('rule_based_acc'):\n"
        f"  - True Count: {rule_based_acc_true_count}\n"
        f"  - Correctness Rate: {rule_based_acc_percentage:.2f}%\n"
    )

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        print("-" * 40)
        print(f"报告生成成功！已保存至: {report_path}")
    except Exception as e:
        print(f"写入报告文件时发生错误: {e}")


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        
class PHYSICS(TextBaseDataset):
    TYPE = "QA"
    DATASET_URL = {
        'PHYSICS': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/PHYSICS.tsv'
    }

    DATASET_MD5 = {
        'PHYSICS': '7303c8d8bcb11f78f420aa25216cc9ae'
    }



    def build_prompt(self, line):
        SYSTEM_PROMPT = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it.
    The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags."""
        
        INSTRUCTION_TEMPLATE = """Below is an open-ended problem in undergraduate-level Physics. Please answer this problem adhering to the following rules:
    1. Use LaTeX format for formulas.
    2. Put the final answer(s) in \\boxed{}, without units.
    3. If multiple answers exist, separate them by commas in \\boxed{}.
    Problem: {{prompt}}"""

        # 获取问题文本（支持 Series 或 dict）
        prompt_text = line["question"]

        # 应用指令模板
        prompt_text = Template(INSTRUCTION_TEMPLATE).render(prompt=prompt_text)

        # 拼接最终的完整 prompt
        full_prompt = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User: {prompt_text}\n"
        )

        return full_prompt
    
    def write_jsonl(self, data_path, dataset, indent=0, mode='w'):
        with open(data_path, mode, encoding='UTF-8') as f:
            if not isinstance(dataset, list):
                dataset = [dataset]
            for data in dataset:
                line = json.dumps(data, ensure_ascii=False, indent=indent if indent != 0 else None)
                f.write(line + '\n')
            
    def evaluate(self, eval_file, **judge_kwargs):
        directory = os.path.dirname(eval_file)
        basename = os.path.basename(eval_file)
        file_stem, _ = os.path.splitext(basename)
        output_filename = f"{file_stem}_results.jsonl"
        output_path = os.path.join(directory, output_filename)
        os.makedirs(directory, exist_ok=True)

        model_args = Model_args()
        if gpt_key_set():
            model_args.base_url = judge_kwargs.get('base_url', os.environ.get('OPENAI_API_BASE', None))

            model_args.api_key = judge_kwargs.get('api_key', os.environ.get('OPENAI_API_KEY', None))

            model_args.model_name = judge_kwargs.get('model', 'gpt-4o')
        else:
            warnings.warn("OPENAI_API_KEY is not set.")
            os._exit(0)

        data = load(eval_file)
        # iterrows 返回的是 (index, Series) 元组
        for index, row in tqdm(data.iterrows(), desc="Scoring", total=len(data)):
            try:
                # 调用被装饰的函数
                self.process_single_item(row, model_args, output_path)

            except TimeoutError:
                # wrapt_timeout_decorator 超时会抛出 TimeoutError
                # 注意：这里 row 是 Series，获取值建议用 .get() 或 []
                question_text = row.get('question', 'Unknown')
                print(f"Timeout processing item: {question_text}")
                continue

            except Exception as e:
                # 捕获其他可能的错误，防止程序中断
                print(f"Error processing item: {e}")
                continue
        generate_score_report(output_path)
        return eval_file

    @wrapt_timeout_decorator.timeout(30)
    def process_single_item(self, row, model_args, output_path):
        """
        处理单条数据的方法，被装饰器限制运行时间最多 30 秒。
        注意：row 是 pandas 的 Series 对象。
        """
        # 将 Series 转为 dict
        item = row.to_dict()

        # 计算分数
        result = compute_score(
            item['prediction'],
            item['answer'],
            item['question'],
            model_args=model_args
        )

        # 更新 item
        item['rule_based_acc'] = result['rule_based_acc']
        item['acc'] = result['acc']
        item['extracted_gt'] = result['extracted_gt']
        item['extracted_pred'] = result['extracted_pred']

        # 写入文件
        self.write_jsonl(output_path, item, mode='a')
