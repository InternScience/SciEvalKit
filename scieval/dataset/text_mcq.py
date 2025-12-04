from .text_base import TextBaseDataset
from .utils import DEBUG_MESSAGE
from .utils.judge_util import build_judge_model,build_judge
from ..smp import *
from ..smp.file import get_intermediate_file_path


class TextMCQDataset(TextBaseDataset):
    TYPE = 'MCQ'

    DATASET_URL = {}

    DATASET_MD5 = {}

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'

        msgs = []

        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc, report_acc_MMT, mcq_circular_eval, mcq_vanilla_eval
        # assert dataset is not None
        dataset_map = {
            'MMBench_TEST_EN': 'MMBench', 'MMBench_TEST_EN_V11': 'MMBench_V11',
            'MMBench_TEST_CN': 'MMBench_CN', 'MMBench_TEST_CN_V11': 'MMBench_CN_V11'
        }
        dataset = self.dataset_name
        if dataset in dataset_map:
            dataset = dataset_map[dataset]
        nproc = judge_kwargs.pop('nproc', 4)

        circular = False
        model = judge_kwargs.get('model', 'exact_matching')
        name_str_map = {'chatgpt-0125': 'openai', 'gpt-4-0125': 'gpt4'}
        name_str = name_str_map[model] if model in name_str_map else model

        if model == 'exact_matching':
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
            model = None

        result_file = get_intermediate_file_path(eval_file, f'_{name_str}_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        if circular:
            data = mcq_circular_eval(model, data, meta, nproc, result_file, self.dataset_name)
        else:
            data = mcq_vanilla_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        eval_name_result = get_intermediate_file_path(eval_file, f'_{name_str}_result')
        dump(data, eval_name_result)
        data = load(eval_name_result)

        # May have different report acc functions for different datasets
        if 'MMT' in dataset:
            acc = report_acc_MMT(data)
        else:
            acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        return acc


class CustomTextMCQDataset(TextMCQDataset):

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)


class ProteinLMBench(TextMCQDataset):
    
    # https://github.com/tsynbio/ProteinLMDataset/blob/main/benchmark/benchmark_your_model.py
    # defalt evaluation is exact matching

    DATASET_URL = {
        "ProteinLMBench": "https://huggingface.co/datasets/InternScience/SciEval/resolve/main/ProteinLMBench.tsv"
    }

    DATASET_MD5 = {
        "ProteinLMBench": '1ae6f56dee315e335cde71833799bb0e'
    }
    
    def build_prompt(self, line):

        prompt = ("""
Answer the multiple-choice question based solely on the provided context. 
If you are still unsure about the answer, output option 7.
Select only ONE correct option by its number. Start your response with 'The correct option is' followed by the option number ONLY. eg: "The correct option is Option X."
Think step by step.
    """)
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']
        options = ""
        for i in range(1,7):
            key = f"option {i}"
            opt = str(line[key]).strip()
            # option 3: Nuclear magnetic resonance (NMR)
            options += f"{key}: {opt}" +'\n'
            
        full_prompt = prompt + '\n Question: \n' + question + '\n Options: \n' + options + '\nThe correct option is:'


        msgs = []

        msgs.append(dict(type='text', value=full_prompt))

        return msgs
    
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.multiple_choice import report_acc
        import re
        data = load(eval_file)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )
        
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [re.search(r'\d+', str(x)).group() for x in data['answer']]
        
        extrac_pred = []
        for pred in data['prediction']:
            try:
                extrac_pred.append(re.search(r'\d+', pred).group())
            except:
                extrac_pred.append("7") # False Predict
                
        data['prediction'] = extrac_pred

        data['hit'] = [int(p==a) for p,a in zip(data['prediction'], data['answer'])]

        # load split
        eval_name_result = get_intermediate_file_path(eval_file, f'_result')
        dump(data, eval_name_result)
        data = load(eval_name_result)
        

        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        return acc