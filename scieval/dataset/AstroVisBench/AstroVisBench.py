import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from scieval.smp import *
from scieval.dataset.image_vqa import ImageVQADataset
from .evaluate_helper import *
from .vis_eval_helper import *
import ast
import multiprocessing
import time
import traceback
import os
import json
from scieval.dataset.utils.judge_util import *
SYS_PROMPT = """
    You are tasked with completing a jupyter notebook about astronomy. You will be given some markdown cells and some python code cells for context, and in response you must output only python code that accurately fulfills the goals of the notebook as described by the markdown text.
"""
def worker_wrapper(queue, task_id, kwargs):

    try:
        result = execute_notebook_w_pro_test(**kwargs)
        queue.put({'status': 'success', 'task_id': task_id, 'data': result})
    except Exception as e:
        queue.put({'status': 'error', 'task_id': task_id, 'error': str(e)})

def vis_worker_wrapper(queue, uid, kwargs):
    try:
        result = execute_notebook_visout(**kwargs)
        queue.put({'status': 'success', 'task_id': uid, 'data': result})
    except Exception as e:
        # 捕获常规异常
        queue.put({'status': 'error', 'task_id': uid, 'error': str(e)})

def remove_ticks(code):
    code_lines = code.split("\n")
    new_code_lines = [line for line in code_lines if not line.startswith("```") and not "<CELL END>" in line]
    return "\n".join(new_code_lines)

class AstroVisBench(ImageVQADataset):
    DATASET_URL = {
        'AstroVisBench_Processing': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/AstroVisBench_Processing.tsv',
        'AstroVisBench_Visualization': 'https://huggingface.co/datasets/InternScience/SciEval/resolve/main/AstroVisBench_Processing.tsv'
    }
    DATASET_MD5 = {
        'AstroVisBench_Processing': '18160b423a5174e26c7832eb4180f30b',
        'AstroVisBench_Visualization': '18160b423a5174e26c7832eb4180f30b'
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL.keys())

    def __init__(self, dataset, **kwargs):
        super().__init__(dataset)
        self.dataset = dataset
        self.dataset_name = dataset
        self.TYPE = 'QA'
        self.MODALITY = 'IMAGE'
        self.add_pro_uds = True
        self.add_vis_uds = False
        self.env_path = os.environ.get('AstroVisBench_Env')
        if not self.env_path:
            raise ValueError(
                "Environment variable 'AstroVisBench_Env' is not set. "
                "Please set it to the AstroVisBench environment directory."
            )
        self.vis_cache_path = None
        self.gen_cache_path = None
        self.true_cache_path = None

    def build_prompt(self, line):
        msgs = [{'type': 'text', 'value': SYS_PROMPT, 'role': 'system'}]
        if self.dataset_name == 'AstroVisBench_Processing':
            context = [line['setup_query'],
                       '```\n\n' + line['setup_gt_code'] + '\n\n```',
                       line['processing_query'],
                       ]

            if self.add_pro_uds:
                context.append(line['processing_underspecifications'])
        elif self.dataset_name == 'AstroVisBench_Visualization':
            context = [line['setup_query'],
                       '```\n\n' + line['setup_gt_code'] + '\n\n```',
                       line['processing_query'],
                       '```\n\n' + line['processing_gt_code'] + '\n\n```',
                       line['visualization_query'],
                       ]

            if self.add_vis_uds:
                context.append(line['visualization_underspecifications'])
        else:
            raise Exception('Unknown dataset')

        context = '\n\n'.join(context)
        msgs.append({'type': 'text', 'value': context, 'role': 'user'})
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        eval_root = os.path.dirname(eval_file)
        vis_cache_dir = os.path.join(eval_root,'vis_cache')
        os.makedirs(vis_cache_dir, exist_ok=True)
        print(f'vis cache dir: {vis_cache_dir}')
        gen_cache_dir = os.path.join(eval_root,'gen_cache')
        os.makedirs(gen_cache_dir, exist_ok=True)
        print(f'gen cache dir: {gen_cache_dir}')
        model_root_dir = os.path.dirname(eval_root)
        true_cache_dir = os.path.join(model_root_dir, 'true_cache')
        os.makedirs(true_cache_dir, exist_ok=True)
        print(f'true cache dir: {true_cache_dir}')
        self.vis_cache_path = vis_cache_dir
        self.gen_cache_path = gen_cache_dir
        self.true_cache_path = true_cache_dir

        if self.dataset_name == 'AstroVisBench_Processing':
            self.evaluate_process(eval_file, **judge_kwargs)
        elif self.dataset_name == 'AstroVisBench_Visualization':
            self.evaluate_visualization(eval_file, **judge_kwargs)
        else:
            raise Exception('Unknown dataset')
        pass

    def evaluate_process(self, eval_file, **judge_kwargs):
        print(f"Starting evaluation for {self.dataset_name}...")
        MAX_WORKERS = judge_kwargs.pop('max_workers', 4)
        ctx = multiprocessing.get_context('spawn')
        try:
            df = pd.DataFrame(load(eval_file))
        except Exception as e:
            print(f"Error loading evaluation file {eval_file}: {e}")
            return

        all_evaluation_results = []
        os.makedirs(self.true_cache_path, exist_ok=True)
        os.makedirs(self.gen_cache_path, exist_ok=True)

        print(f"Executing tasks with parallelism: {MAX_WORKERS} workers...")

        manager = ctx.Manager()
        result_queue = manager.Queue()

        pending_tasks = []
        for index, row in df.iterrows():
            task_dict = self._prepare_jdict(row)
            func_kwargs = {
                'jdict': task_dict,
                'true_cache': self.true_cache_path,
                'gen_cache': self.gen_cache_path,
                'skip_test': False,
                'temp_caching': True,
                'min_diff_only': True
            }
            pending_tasks.append((row.get('uid'), task_dict, func_kwargs))

        running_procs = []

        pbar = tqdm(total=len(pending_tasks))

        task_idx = 0
        while task_idx < len(pending_tasks) or len(running_procs) > 0:

            while len(running_procs) < MAX_WORKERS and task_idx < len(pending_tasks):
                uid, task_dict, func_kwargs = pending_tasks[task_idx]

                p = ctx.Process(target=worker_wrapper, args=(result_queue, uid, func_kwargs))
                p.start()
                
                running_procs.append({
                    'p': p,
                    'uid': uid,
                    'task_dict': task_dict,
                    'start_time': time.time()
                })
                task_idx += 1

            for i in range(len(running_procs) - 1, -1, -1):
                proc_info = running_procs[i]
                p = proc_info['p']
                uid = proc_info['uid']

                if time.time() - proc_info['start_time'] > 600:
                    print(f"Task {uid} timed out. Killing...")
                    if p.is_alive():
                        p.terminate()
                        p.join()

                    err_res = proc_info['task_dict'].copy()
                    err_res['processing_test'] = {'error': 'Timeout', 'pro_success': False}
                    all_evaluation_results.append(err_res)
                    
                    running_procs.pop(i)
                    pbar.update(1)
                    continue

                if not p.is_alive():
                    p.join()

                    if p.exitcode != 0:
                        print(f"Task {uid} crashed with exit code {p.exitcode} (Segfault).")
                        err_res = proc_info['task_dict'].copy()
                        err_res['processing_test'] = {'error': f'Process crashed (Exit code: {p.exitcode})', 'pro_success': False}
                        all_evaluation_results.append(err_res)

                    running_procs.pop(i)
                    pbar.update(1)

            while not result_queue.empty():
                try:
                    res = result_queue.get_nowait()
                    if res['status'] == 'success':
                        all_evaluation_results.append(res['data'])
                    else:
                        pass
                except:
                    break

            time.sleep(0.1)

        pbar.close()

        print("Saving evaluation results to josnl file...")
        try:
            eval_root = os.path.dirname(eval_file)
            output_filename = os.path.join(eval_root, os.path.splitext(os.path.basename(eval_file))[
                0] + "_processing_results.jsonl")

            with open(output_filename, 'w') as f:
                for result_dict in all_evaluation_results:
                    f.write(json.dumps(result_dict, default=str) + '\n')

            print(f"Evaluation finished. Results saved to {output_filename}")

        except Exception as e:
            print(f"Failed to save results to jsonl file: {e}")
            fallback_filename = "processing_results_fallback.json"
            print(f"Saving to {fallback_filename} as a fallback.")
            with open(fallback_filename, 'w') as f:
                f.write(json.dumps(all_evaluation_results, indent=4, default=str))

        self.aggregate_results_for_pro(all_evaluation_results, eval_root)

        return

    def evaluate_visualization(self, eval_file, **judge_kwargs):
        print(f"Starting evaluation for {self.dataset_name}...")

        eval_root = os.path.dirname(eval_file)
        processing_results_filename = os.path.join(eval_root, os.path.splitext(os.path.basename(eval_file))[0].replace("_Visualization",
                                                                                               "_Processing") + "_processing_results.jsonl")

        vis_exec_output_filename = processing_results_filename.replace(".jsonl", "_vis_exec_results.jsonl")

        if not os.path.exists(processing_results_filename):
            raise FileNotFoundError(
                f"Prerequisite file not found: {processing_results_filename}. Please run the AstroVisBench_Processing evaluation first.")

        print(f"Loading prerequisite processing results from: {processing_results_filename}")
        df_process_results = load(processing_results_filename)
        df_process_results = pd.DataFrame(df_process_results)
        if 'prediction' in df_process_results.columns:
            df_process_results.rename(columns={'prediction': 'prediction_process'}, inplace=True)

        print(f"Loading visualization code from: {eval_file}")
        df_vis_code = pd.DataFrame(load(eval_file))

        if 'uid' not in df_process_results.columns or 'uid' not in df_vis_code.columns:
            raise KeyError("Column 'uid' is required for merging but not found in one of the files.")

        df_merged = pd.merge(df_process_results, df_vis_code[['uid', 'prediction']], on='uid', how='inner')

        if len(df_merged) != len(df_vis_code):
            print(f"Warning: Merging resulted in {len(df_merged)} rows, but visualization input has {len(df_vis_code)} rows. Some tasks may be missing.")

        print("\n--- Starting Stage 1: Code Execution (Parallel & Resumable) ---")
        os.makedirs(self.vis_cache_path, exist_ok=True)

        finished_uids = set()
        existing_results = []
        
        if os.path.exists(vis_exec_output_filename):
            print(f"Found existing execution results at {vis_exec_output_filename}. Checking for resumable tasks...")
            try:
                with open(vis_exec_output_filename, 'r') as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            item = json.loads(line)
                            if 'uid' in item:
                                finished_uids.add(item['uid'])
                                existing_results.append(item)
                        except json.JSONDecodeError:
                            print("Warning: Skipping a corrupted line in existing results file.")
            except Exception as e:
                print(f"Error reading existing file: {e}. Starting fresh.")
                finished_uids = set()
                existing_results = []

        print(f"Total tasks: {len(df_merged)}. Already finished: {len(finished_uids)}.")

        pending_tasks = []
        for index, row in df_merged.iterrows():
            if row['uid'] in finished_uids:
                continue
            
            task_dict = self._prepare_jdict_for_vis(row)
            func_kwargs = {
                'jdict': task_dict,
                'gen_vis_cache': self.vis_cache_path
            }
            pending_tasks.append((row.get('uid'), task_dict, func_kwargs))

        vis_exec_results_new = []
        
        if len(pending_tasks) > 0:
            print(f"Resuming execution for {len(pending_tasks)} remaining tasks.")

            MAX_WORKERS = judge_kwargs.pop('max_workers', 4)
            ctx = multiprocessing.get_context('spawn')
            manager = ctx.Manager()
            result_queue = manager.Queue()
            
            running_procs = []
            pbar = tqdm(total=len(pending_tasks), desc="Stage 1 Execution (Running)")
            
            task_idx = 0

            with open(vis_exec_output_filename, 'a') as f_out:
                
                while task_idx < len(pending_tasks) or len(running_procs) > 0:

                    while len(running_procs) < MAX_WORKERS and task_idx < len(pending_tasks):
                        uid, task_dict, func_kwargs = pending_tasks[task_idx]
                        p = ctx.Process(target=vis_worker_wrapper, args=(result_queue, uid, func_kwargs))
                        p.start()
                        running_procs.append({
                            'p': p, 'uid': uid, 'task_dict': task_dict, 'start_time': time.time()
                        })
                        task_idx += 1

                    for i in range(len(running_procs) - 1, -1, -1):
                        proc_info = running_procs[i]
                        p = proc_info['p']
                        uid = proc_info['uid']
                        error_result = None

                        if time.time() - proc_info['start_time'] > 600:
                            print(f"Vis Task {uid} timed out. Killing...")
                            if p.is_alive(): p.terminate(); p.join()
                            error_result = proc_info['task_dict'].copy()
                            error_result['visualization_test'] = {'error': 'Timeout', 'vis_success': False, 'gen_vis_list': []}

                        elif not p.is_alive():
                            p.join()
                            if p.exitcode != 0:
                                print(f"Vis Task {uid} crashed (Exit: {p.exitcode}).")
                                error_result = proc_info['task_dict'].copy()
                                error_result['visualization_test'] = {'error': f'Crashed (Code {p.exitcode})', 'vis_success': False, 'gen_vis_list': []}
                            else:
                                pass

                        if error_result:
                            f_out.write(json.dumps(error_result, default=str) + '\n')
                            f_out.flush()
                            vis_exec_results_new.append(error_result)
                            
                            running_procs.pop(i)
                            pbar.update(1)
                        elif not p.is_alive():

                            running_procs.pop(i)

                    while not result_queue.empty():
                        try:
                            res = result_queue.get_nowait()
                            final_data = None
                            
                            if res['status'] == 'success':
                                final_data = res['data']
                            else:
                                err_task_dict = next((t[1] for t in pending_tasks if t[0] == res['task_id']), None)
                                if err_task_dict:
                                    final_data = err_task_dict.copy()
                                    final_data['visualization_test'] = {'error': res['error'], 'vis_success': False, 'gen_vis_list': []}
                            
                            if final_data:
                                f_out.write(json.dumps(final_data, default=str) + '\n')
                                f_out.flush()
                                
                                vis_exec_results_new.append(final_data)
                                pbar.update(1)
                        except Exception:
                            break
                    
                    time.sleep(0.1)

            pbar.close()
            manager.shutdown()
            print(f"Stage 1 finished. New results saved to {vis_exec_output_filename}")
        
        else:
            print("All tasks are already completed in Stage 1. Skipping execution.")

        print("\n--- Starting Stage 2: VLM Evaluation ---")

        full_vis_results = existing_results + vis_exec_results_new

        if len(full_vis_results) == 0:
            print("Warning: No results available for VLM evaluation!")
            return

        model = judge_kwargs.pop('model', 'Claude4_5_Sonnet')
        model = build_judge_model(model=model, **judge_kwargs)
        try:

            final_results_list = do_vis_eval(full_vis_results,model)

            final_output_filename = vis_exec_output_filename.replace(".jsonl", "_final_results.jsonl")
            print(f"Stage 2 finished. Saving final evaluation results to {final_output_filename}")
            
            with open(final_output_filename, 'w') as f:
                for result_dict in final_results_list:
                    f.write(json.dumps(result_dict, default=str) + '\n')

            print("\nEvaluation complete!")

        except Exception as e:
            print(f"An error occurred during VLM evaluation: {e}")
            traceback.print_exc()


        self.aggregate_results_for_vis(final_results_list, eval_root)
        return
    
    def _prepare_jdict(self, row):

        jdict = row.to_dict()

        jdict['processing_gen_code'] = remove_ticks(jdict['prediction'])

        jdict['visualization_gen_code'] = ""

        jdict['env_path'] = get_env_path(jdict, self.env_path)

        return jdict

    def _prepare_jdict_for_vis(self, row):
        jdict = row.to_dict()

        jdict['visualization_gen_code'] = remove_ticks(jdict['prediction'])

        jdict['env_path'] = get_env_path(jdict, self.env_path)


        if isinstance(jdict.get('processing_test'), str):
            try:
                jdict['processing_test'] = ast.literal_eval(jdict['processing_test'])
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Could not parse 'processing_test' field for uid {jdict.get('uid')}. Treating as empty.")
                jdict['processing_test'] = {}

        return jdict

    def aggregate_results_for_pro(self, results_list, output_dir):
        print("Aggregating results for Processing tasks...")

        process_success_flags = [
            q.get('processing_test', {}).get('pro_success', False) for q in results_list
        ]
        avg_pro_success_rate = np.mean(process_success_flags) if process_success_flags else 0

        processing_vi_scores = [
            q['processing_test']['inspection_results']['agg_scores']['unweighted']
            for q in results_list
            if q.get('processing_test', {}).get('pro_success', False) and \
               q.get('processing_test', {}).get('inspection_results', {}).get('results')
        ]
        avg_vi_score = np.mean(processing_vi_scores) if processing_vi_scores else 0

        final_stats = {
            "processing_task_execution_success_rate": avg_pro_success_rate,
            "average_variable_inspection_score": avg_vi_score,
            "total_tasks_evaluated": len(results_list)
        }

        output_filename = os.path.join(output_dir, "processing_aggregation_results.json")
        try:
            with open(output_filename, 'w') as f:
                json.dump(final_stats, f, indent=4)
            print(f"Processing aggregation results saved to {output_filename}")
        except Exception as e:
            print(f"Error saving processing aggregation results: {e}")

        return final_stats

    def aggregate_results_for_vis(self, results_list, output_dir):

        from collections import Counter
        print("Aggregating results for Visualization tasks...")

        vis_success_flags = [
            q.get('visualization_test', {}).get('vis_success', False) for q in results_list
        ]
        avg_vis_success_rate = np.mean(vis_success_flags) if vis_success_flags else 0

        vis_eval_errors = [
            q.get('visualization_llm_eval', {}).get('errors', 'Evaluation_Error') for q in results_list
        ]
        vis_error_distribution = dict(Counter(vis_eval_errors))

        final_stats = {
            "visualization_task_execution_success_rate": avg_vis_success_rate,
            "vlm_evaluation_error_distribution": vis_error_distribution,
            "total_tasks_evaluated": len(results_list)
        }

        output_filename = os.path.join(output_dir, "visualization_aggregation_results.json")
        try:
            with open(output_filename, 'w') as f:
                json.dump(final_stats, f, indent=4)
            print(f"Visualization aggregation results saved to {output_filename}")
        except Exception as e:
            print(f"Error saving visualization aggregation results: {e}")

        return final_stats