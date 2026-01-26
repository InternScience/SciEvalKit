import argparse
import copy
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from scieval.agents import create_agent, get_available_agents
from scieval.agents.records import EvalRecord, TrajectoryStore
from scieval.dataset import build_dataset
from scieval.smp import dump, get_logger, load, timestr, githash, ls


def _build_dataset_from_config(cfg: Dict[str, Any], dataset_name: str):
    import inspect
    import scieval.dataset as dataset_mod

    config = copy.deepcopy(cfg[dataset_name])
    if config == {}:
        return build_dataset(dataset_name)
    if "class" not in config:
        return build_dataset(dataset_name, **config)
    cls_name = config.pop("class")
    if hasattr(dataset_mod, cls_name):
        cls = getattr(dataset_mod, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        return cls(**valid_params)
    raise ValueError(f"Dataset class {cls_name} is not supported in scieval.dataset")


def _build_agent_from_config(cfg: Dict[str, Any], agent_name: str):
    """
    Build an agent from configuration.
    
    Args:
        cfg: Configuration dictionary
        agent_name: Name of the agent in the config
    
    Returns:
        Agent instance
    
    Raises:
        ValueError: If agent class is not supported
        ImportError: If agent dependencies are not installed
    """
    config = copy.deepcopy(cfg[agent_name])
    cls_name = config.pop("class", "SmolAgentsAgent")
    
    # Handle legacy name mapping
    if cls_name == "smolagents":
        cls_name = "SmolAgentsAgent"
    
    try:
        return create_agent(cls_name, **config)
    except ImportError as e:
        available = get_available_agents()
        available_list = [name for name, avail in available.items() if avail]
        raise ImportError(
            f"Failed to create agent '{cls_name}'. "
            f"Required dependencies may not be installed.\n"
            f"Available agents: {', '.join(available_list) if available_list else 'None'}\n"
            f"Error: {e}"
        ) from e


def _run_one_sample(
    idx: int,
    agent,
    dataset,
    store: TrajectoryStore,
    judge_kwargs: Dict[str, Any],
    reuse: bool,
    do_infer: bool,
    do_eval: bool,
) -> Tuple[int, Dict[str, Any], str]:
    final_answer = ""
    traj = store.load_traj(idx) if reuse else None
    if do_infer:
        if traj and traj.get("success"):
            final_answer = traj.get("final_answer", "")
        else:
            sample = dataset.build_agent_sample(idx)
            result = agent.run(sample)
            store.save_traj(idx, result)
            final_answer = result.final_answer
    elif traj:
        final_answer = traj.get("final_answer", "")

    if not do_eval:
        return idx, {}, final_answer

    eval_cached = store.load_eval(idx) if reuse else None
    if eval_cached is not None:
        cached_score = eval_cached.get("score", eval_cached)
        cached_final = eval_cached.get("final_answer", final_answer)
        return idx, cached_score, cached_final

    score = dataset.score_agent_sample(idx, final_answer, **judge_kwargs)
    metadata = {}
    if "question" in score:
        metadata["question"] = score["question"]
    if "answer" in score:
        metadata["answer"] = score["answer"]
    record = EvalRecord(index=idx, final_answer=final_answer, score=score, metadata=metadata)
    store.save_eval(idx, record)
    return idx, score, final_answer


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def run_agent_eval(
    agent,
    dataset,
    work_dir: str,
    nproc: int = 1,
    reuse: bool = False,
    mode: str = "all",
    judge_kwargs: Dict[str, Any] = None,
):
    logger = get_logger("AGENT_EVAL")
    judge_kwargs = judge_kwargs or {}
    dataset_name = getattr(dataset, "dataset_name", dataset.__class__.__name__)
    root_dir = osp.join(work_dir, "agent_eval", dataset_name, agent.name, agent.model_version)
    eval_id = f"T{timestr('day')}_G{githash(digits=8)}"
    log_dir = osp.join(root_dir, eval_id)
    if reuse and osp.exists(root_dir):
        prev_runs = ls(root_dir, mode="dir")
        if prev_runs:
            prev_runs.sort()
            log_dir = prev_runs[-1]
    store = TrajectoryStore(log_dir)
    logger.info(f"Logging directory: {log_dir}")

    do_infer = mode in ["all", "infer"]
    do_eval = mode in ["all", "eval"]

    results: List[Tuple[int, Dict[str, Any], str]] = []
    tasks = list(range(len(dataset)))
    tasks_to_run = tasks
    if reuse:
        tasks_to_run = []
        for idx in tasks:
            if do_eval:
                eval_cached = store.load_eval(idx)
                if eval_cached is not None:
                    cached_score = eval_cached.get("score", eval_cached)
                    cached_final = eval_cached.get("final_answer", "")
                    if not cached_final:
                        traj = store.load_traj(idx)
                        if traj is not None:
                            cached_final = traj.get("final_answer", "")
                    results.append((idx, cached_score, cached_final))
                    continue
                tasks_to_run.append(idx)
                continue

            if do_infer:
                traj = store.load_traj(idx)
                if traj and traj.get("success"):
                    results.append((idx, {}, traj.get("final_answer", "")))
                else:
                    tasks_to_run.append(idx)
            else:
                tasks_to_run.append(idx)

    if nproc > 1:
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [
                executor.submit(
                    _run_one_sample,
                    idx,
                    agent,
                    dataset,
                    store,
                    judge_kwargs,
                    reuse,
                    do_infer,
                    do_eval,
                )
                for idx in tasks_to_run
            ]
            with tqdm(total=len(tasks_to_run), desc="Agent Eval", unit="sample") as pbar:
                for fut in as_completed(futures):
                    results.append(fut.result())
                    pbar.update(1)
    else:
        with tqdm(total=len(tasks_to_run), desc="Agent Eval", unit="sample") as pbar:
            for idx in tasks_to_run:
                results.append(
                    _run_one_sample(
                        idx, agent, dataset, store, judge_kwargs, reuse, do_infer, do_eval
                    )
                )
                pbar.update(1)

    results.sort(key=lambda x: x[0])
    predictions = [{"index": idx, "prediction": final_answer} for idx, _, final_answer in results]
    pred_file = osp.join(log_dir, f"{agent.name}_{dataset_name}.json")
    dump(predictions, pred_file)

    agg: Dict[str, List[float]] = {}
    for _, score, _ in results:
        for k, v in score.items():
            if _is_number(v):
                agg.setdefault(k, []).append(float(v))

    summary = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    summary_file = osp.join(log_dir, "summary.json")
    dump(summary, summary_file)
    return summary


def run_agent_eval_from_config(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    logger = get_logger("AGENT_RUN")
    agent_cfg = cfg.get("agent") or cfg.get("agents")
    data_cfg = cfg.get("data")
    if not agent_cfg or not data_cfg:
        raise ValueError("Config must include 'agent' and 'data' sections for agent evaluation.")

    if isinstance(agent_cfg, dict) and "class" in agent_cfg:
        agents_cfg = {"agent": agent_cfg}
    else:
        agents_cfg = agent_cfg

    results = {}
    for agent_name in agents_cfg:
        agent = _build_agent_from_config(agents_cfg, agent_name)
        for dataset_name in data_cfg:
            dataset = _build_dataset_from_config(data_cfg, dataset_name)
            if dataset is None:
                logger.error(f"Dataset {dataset_name} is not valid, skipping.")
                continue
            summary = run_agent_eval(
                agent,
                dataset,
                work_dir=args.work_dir,
                nproc=args.api_nproc,
                reuse=args.reuse,
                mode=args.mode,
                judge_kwargs={
                    "model": getattr(args, "judge", None),
                    "api_key": os.environ.get("OPENAI_API_KEY", ""),
                    "api_base": os.environ.get("OPENAI_API_BASE", ""),
                },
            )
            results[f"{agent_name}:{dataset_name}"] = summary
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Agent evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to agent eval config JSON")
    parser.add_argument("--work-dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "infer", "eval"])
    parser.add_argument("--api-nproc", type=int, default=1, help="Parallel agent calls")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--judge", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load(args.config)
    run_agent_eval_from_config(cfg, args)


if __name__ == "__main__":
    main()
