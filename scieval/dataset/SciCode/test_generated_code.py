import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datasets import load_dataset

import numpy as np

# Import numeric test data constants and helpers from the local parse module.  We
# use relative imports here so that this file can reside within the SciCode
# package without requiring installation of a top-level ``scicode`` package.
from .parse import H5PY_FILE  # type: ignore
from .parse import read_from_hf_dataset  # type: ignore

PROB_NUM = 80
DEV_PROB_NUM = 15
STEP_NUM = sum(len(p["sub_steps"]) for p in load_dataset("SciCode1/SciCode", split="test"))
DEV_STEP_NUM = 50


def _get_background_dir(with_background: bool) -> str:
    return "with_background" if with_background else "without_background"


def test_code(
    model_name: str,
    split: str,
    code_dir: Path,
    log_dir: Path,
    output_dir: Path,
    with_background: bool = False,
):
    """Run SciCode unit tests on generated code files.

    This function iterates over the generated Python files in ``code_dir`` and
    executes each one against its associated test cases.  The results are
    recorded in ``output_dir`` and a per-function status is cached in
    ``log_dir`` to avoid re-running tests unnecessarily.

    Parameters
    ----------
    model_name : str
        Name of the model used to generate the code.  This is used as part of
        the directory structure for logs and outputs.
    split : str
        Dataset split (``"validation"`` or ``"test"``).
    code_dir : Path
        Path to the directory containing generated code files.  Each file is
        expected to be named ``{problem_id}.{step}.py``.
    log_dir : Path
        Directory where execution logs are stored.  Each test script writes
        ``pass``, ``fail`` or ``time out`` into a .txt file to indicate
        the outcome.
    output_dir : Path
        Directory to write summary statistics and JSON outputs.
    with_background : bool, optional
        Whether the prompts included scientific background.  This flag only
        affects the names of the log and output files.
    """

    scicode_data = read_from_hf_dataset(split)
    scicode_data = [data for data in scicode_data]
    json_dct: Dict[str, int] = {}
    json_idx: Dict[str, int] = {}

    for prob_data in scicode_data:
        json_dct[prob_data["problem_id"]] = len(prob_data["sub_steps"])
        json_idx[prob_data["problem_id"]] = scicode_data.index(prob_data)
    start_time = time.time()

    code_dir_ = Path(code_dir, _get_background_dir(with_background))
    tmp_dir = Path(f"tmp_{start_time}")

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for file_path in code_dir_.iterdir():
        if file_path.is_file():
            file_name = file_path.stem
            file_id = file_name.split(".")[0]
            file_step = file_name.split(".")[1]

            code_content = file_path.read_text(encoding="utf-8")
            # Strip placeholder error lines to avoid syntax errors in tmp scripts
            code_content = "\n".join(
                ln for ln in code_content.splitlines()
                if not ln.lstrip().startswith("Failed to obtain answer via API.") and ln.strip() != ""
            )
            json_content = scicode_data[json_idx[file_id]]
            step_id = json_content["sub_steps"][int(file_step) - 1]["step_number"]
            test_lst = json_content["sub_steps"][int(file_step) - 1]["test_cases"]
            assert_file = Path(tmp_dir, f"{step_id}.py")
            with open(assert_file, "w", encoding="utf-8") as f:
                f.write(code_content)
                f.write(
                    """
# --- Ensure repository root is importable so 'scieval' resolves ---
from pathlib import Path as _Path
import sys as _sys, importlib as _imp
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

# --- Add upstream src/scicode path if it exists ---
_scicode_src = _repo_root / "scieval" / "dataset" / "SciCode" / "src"
if _scicode_src.exists() and str(_scicode_src) not in _sys.path:
    _sys.path.insert(0, str(_scicode_src))

# --- Remove any stubbed or partially loaded 'scicode' modules injected by LLM code ---
for _name in list(_sys.modules):
    if _name == "scicode" or _name.startswith("scicode."):
        del _sys.modules[_name]

# --- Alias top-level 'SciCode' to canonical package path ---
if 'SciCode' not in _sys.modules:
    _sys.modules['SciCode'] = _imp.import_module('scieval.dataset.SciCode')

# --- Reload parse module to guarantee fresh, full definition ---
_parse_mod = _imp.import_module('scieval.dataset.SciCode.parse')
_parse_mod = _imp.reload(_parse_mod)
process_hdf5_to_tuple = _parse_mod.process_hdf5_to_tuple  # noqa: F401, E402

"""
                )
                f.write(
                    f"targets = process_hdf5_to_tuple('{step_id}', {len(test_lst)})"
                    + "\n"
                )
                for idx in range(len(test_lst)):
                    f.write(f"target = targets[{idx}]\n\n")
                    for line in test_lst[idx].split("\n"):
                        f.write(line + "\n")

    def run_script(script_path: Path) -> tuple[int, str, str]:
        try:
            res = subprocess.run(
                [sys.executable, str(script_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=1800,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired as e:
            return 124, "", f"TIMEOUT: {e}"

    correct_prob = np.zeros(PROB_NUM)
    tot_prob = np.zeros(PROB_NUM)
    correct_step: List[str] = []
    correct_dict: Dict[str, List[str]] = {}

    for i in range(PROB_NUM):
        correct_dict[f"{i + 1}"] = []

    for file_path in tmp_dir.iterdir():
        if file_path.is_file():
            func_id = file_path.stem
            prob_id = func_id.split(".")[0]
            print(f"Testing function {func_id} ...")
            tot_prob[int(prob_id) - 1] += 1
            logs_dir_ = Path(log_dir, _get_background_dir(with_background))
            logs_dir_.mkdir(parents=True, exist_ok=True)
            logs_file = Path(logs_dir_, f"{file_path.stem}.txt")
            if logs_file.exists():
                with open(logs_file, "r") as f:
                    content = f.read().splitlines()
                    if content[0] == "pass":
                        correct_prob[int(prob_id) - 1] += 1
                        correct_step.append(func_id)
                        correct_dict[prob_id].append(func_id)
                continue
            ret, out, err = run_script(file_path)
            if ret == 0:
                correct_prob[int(prob_id) - 1] += 1
                correct_step.append(func_id)
                correct_dict[str(prob_id)].append(func_id)
                with open(logs_file, "w", encoding="utf-8") as f:
                    f.write("pass")
            elif ret in (1,):
                with open(logs_file, "w", encoding="utf-8") as f:
                    f.write("fail\n\n")
                    if out:
                        f.write("STDOUT:\n" + out + "\n\n")
                    if err:
                        f.write("STDERR:\n" + err + "\n")
            else:
                with open(logs_file, "w", encoding="utf-8") as f:
                    f.write("time out\n\n")
                    if out:
                        f.write("STDOUT:\n" + out + "\n\n")
                    if err:
                        f.write("STDERR:\n" + err + "\n")

    test_time = time.time() - start_time

    correct_prob_num = sum(
        1
        for i in range(PROB_NUM)
        if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0
    )

    print(
        f"correct problems: {correct_prob_num}/"
        f"{DEV_PROB_NUM if (split == 'validation') else PROB_NUM - DEV_PROB_NUM}"
    )
    print(
        f"correct steps: {len(correct_step)}/"
        f"{DEV_STEP_NUM if (split == 'validation') else STEP_NUM}"
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(
        f"{output_dir}/{model_name}_{_get_background_dir(with_background)}.txt",
        "w",
    ) as f:
        f.write(
            f"correct problems: {correct_prob_num}/"
            f"{DEV_PROB_NUM if (split == 'validation') else PROB_NUM - DEV_PROB_NUM}\n"
        )
        f.write(
            f"correct steps: {len(correct_step)}/"
            f"{DEV_STEP_NUM if (split == 'validation') else STEP_NUM}\n\n"
        )
        f.write(f"duration: {test_time} seconds\n")
        f.write("\ncorrect problems: ")
        f.write(
            f"\n\n{[i + 1 for i in range(PROB_NUM) if correct_prob[i] == tot_prob[i] and tot_prob[i] != 0]}\n"
        )

    with open(
        f"{output_dir}/{model_name}_{_get_background_dir(with_background)}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(correct_dict, f, indent=4)

    shutil.rmtree(tmp_dir)


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model name")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["validation", "test"],
        help="Data split",
    )
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path("eval_results", "generated_code"),
        help="Code directory",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_results"),
        help="Eval results directory",
    )
    parser.add_argument(
        "--with-background",
        action="store_true",
        help="Include problem background if enabled",
    )
    return parser


def main(
    model: str,
    split: str,
    code_dir: Path,
    log_dir: Path,
    output_dir: Path,
    with_background: bool,
) -> None:
    if not Path(H5PY_FILE).exists():
        raise FileNotFoundError(
            "Please download the numeric test results before testing generated code."
        )
    model = Path(model).parts[-1]
    test_code(model, split, code_dir, log_dir, output_dir, with_background)


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))