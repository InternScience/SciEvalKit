# Quick Start

Before running the evaluation script, you need to configure the VLMs and correctly set the model paths or API keys. Then, you can use the `run.py` script with relevant arguments to perform inference and evaluation on multiple VLMs and benchmarks.

## Step 0: Installation and Key Setup

### Installation

```bash
git clone https://github.com/InternScience/SciEvalKit.git
cd SciEvalKit
pip install -e .
```

### Setup Keys

To use API models (e.g., GPT-4v, Gemini-Pro-V) for inference, you must set up API keys first.

> **Note:** Some datasets require an LLM as a Judge and have default evaluation models configured (see *Extra Notes*). You also need to configure the corresponding APIs when evaluating these datasets.

You can place the required keys in `$SciEvalKit/.env` or set them directly as environment variables. If you choose to create a `.env` file, the content should look like this:

```bash
# .env file, place it under $SciEvalKit

# --- API Keys for Proprietary VLMs ---
# QwenVL APIs
DASHSCOPE_API_KEY=
# Gemini w. Google Cloud Backends
GOOGLE_API_KEY=
# OpenAI API
OPENAI_API_KEY=
OPENAI_API_BASE=
# StepAI API
STEPAI_API_KEY=
# REKA API
REKA_API_KEY=
# GLMV API
GLMV_API_KEY=
# CongRong API
CW_API_BASE=
CW_API_KEY=
# SenseNova API
SENSENOVA_API_KEY=
# Hunyuan-Vision API
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=
# LMDeploy API
LMDEPLOY_API_BASE=

# --- Evaluation Specific Settings ---
# You can set an evaluation proxy; API calls generated during the evaluation phase will go through this proxy.
EVAL_PROXY=
# You can also set keys and base URLs dedicated for evaluation by appending the _EVAL suffix:
OPENAI_API_KEY_EVAL=
OPENAI_API_BASE_EVAL=
```

Fill in your keys where applicable. These API keys will be automatically loaded during inference and evaluation.

---

## Step 1: Configuration

**VLM Configuration:** All VLMs are configured in `scieval/config.py`. For some VLMs (e.g., MiniGPT-4, LLaVA-v1-7B), additional configuration is required (setting the code/model weight root directory in the config file).

When evaluating, you should use the model name specified in `supported_VLM` in `scieval/config.py`. Ensure you can successfully run inference with the VLM before starting the evaluation.

**Check Command:**

```bash
vlmutil check {MODEL_NAME}
```

---

## Step 2: Evaluation

We use `run.py` for evaluation. You can use `$SciEvalKit/run.py` or create a soft link to the script.

### Basic Arguments

*   `--data` (list[str]): Set the dataset names supported in SciEvalKit (refer to `scieval/dataset/__init__.py` or use `vlmutil dlist all` to check).
*   `--model` (list[str]): Set the VLM names supported in SciEvalKit (defined in `supported_VLM` in `scieval/config.py`).
*   `--mode` (str, default `'all'`): Running mode, choices are `['all', 'infer', 'eval']`.
    *   `"all"`: Perform both inference and evaluation.
    *   `"infer"`: Perform inference only.
    *   `"eval"`: Perform evaluation only.
*   `--api-nproc` (int, default 4): The number of threads for API calling.
*   `--work-dir` (str, default `'.'`): The directory to save the results.
*   `--config` (str): Path to a configuration JSON file. This is a more fine-grained configuration method compared to specifying data and model (**Recommended**). See *ConfigSystem* for details.

### Example Commands

You can use `python` or `torchrun` to run the script.

#### 1. Using python
Instantiates only one VLM, which may use multiple GPUs. Recommended for evaluating very large VLMs (e.g., IDEFICS-80B-Instruct).

```bash
# Inference and Evaluation on MaScQA and ChemBench using IDEFICS-80B-Instruct
python run.py --data MaScQA ChemBench --model idefics_80b_instruct --verbose

# Inference only on MaScQA and ChemBench using IDEFICS-80B-Instruct
python run.py --data MaScQA ChemBench --model idefics_80b_instruct --verbose --mode infer
```

#### 2. Using torchrun
Instantiates one VLM instance per GPU. This speeds up inference but is only suitable for VLMs that consume less GPU memory.

```bash
# Inference and Eval on MaScQA and ChemBench using IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2
# On a node with 8 GPUs
torchrun --nproc-per-node=8 run.py --data MaScQA ChemBench --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose

# On MaScQA using Qwen-VL-Chat. On a node with 2 GPUs
torchrun --nproc-per-node=2 run.py --data MaScQA --model qwen_chat --verbose
```

#### 3. API Model Evaluation

```bash
# Inference and Eval on SFE using GPT-4o
# Set API concurrency to 32. Requires OpenAI base URL and Key.
# Note: SFE evaluation requires OpenAI configuration by default.
python run.py --data SFE --model GPT4o --verbose --api-nproc 32
```

#### 4. Using Config File

```bash
# Evaluate using config. Do not use --data and --model in this case.
python run.py --config config.json
```

**Results:** Evaluation results will be printed as logs. Additionally, result files will be generated in the directory `$YOUR_WORKING_DIRECTORY/{model_name}`. Files ending in `.csv` contain the evaluation metrics.

---

## Extra Settings

### Additional Arguments

*   `--judge` (str): Specify the evaluation model for datasets that require model-based evaluation.
    *   If not specified, the configured default model will be used.
    *   The model can be a VLM supported in SciEvalKit or a custom model.
*   `--judge-args` (str): Arguments for the judge model (in JSON string format).
    *   You can pass parameters like `temperature`, `max_tokens` when specifying the judge via `--judge`.
    *   Specific args depend on the model initialization class (e.g., `scieval.api.gpt.OpenAIWrapper`).
    *   You can specify the instantiation class via the `class` argument (e.g., `OpenAIWrapper` or `Claude_Wrapper`).
    *   You can also specify the model attribute here, but it has lower priority than the model specified by `--judge`.
    *   *Some datasets require unique evaluation parameter settings, see Extra Notes below.*
*   `--reuse` (bool, default `false`): Reuse previous results.
*   `--ignore` (bool, default `false`):
    *   By default (`false`), when loading old inference results, if failed items (exceptions) are found, the program will rerun them.
    *   If set to `true`, failed items will be ignored, and only successful ones will be evaluated.
*   `--fail-fast` (bool, default `false`):
    *   If enabled, the program will stop immediately upon encountering an exception during inference, instead of writing the exception to the result file.
    *   Effective only for API inference.
*   `--ignore-patterns` (list[str]):
    *   Used with `fail-fast`.
    *   Scenario: You enabled fail-fast but want to ignore specific non-fatal errors (e.g., "content policy violation").
    *   Set this to a list of string patterns. Exceptions containing these patterns will be recorded as results instead of crashing the program.
    *   *Some common safety policy violation patterns are configured by default.*
*   `--stream` (bool, default `false`):
    *   Enable streaming output for the model.
    *   Effective only for API inference.
    *   Highly recommended for slow-responding models to prevent HTTP connection timeouts.
    *   *Tip:* When using `--config`, you can also configure this per model in the config file, which takes precedence over the command line.

---

## Extra Notes

### Special Dataset Configurations

Some datasets have specific requirements during evaluation:

*   **Clima_QA:**
    *   Does not calculate FA score by default.
    *   Specify via `--judge-args '{"use_fa": true}'`.
    *   Requires an LLM for evaluation (default is GPT-4).
    *   You can specify the judge model via `--judge`, but it must follow the OpenAI format and have Base URL/Key configured.
*   **PHYSICS:**
    *   Uses model-based evaluation incompatible with the framework's standard model access.
    *   If you want to use a model other than the default GPT-4o, you must specify `base_url` and `api_key` separately (defaults to `OPENAI_API_KEY`, `OPENAI_API_BASE` in env).
*   **AstroVisBench:**
    *   **Environment Dependencies:** Before running, you need to download the runtime dependencies according to the [official instructions](https://github.com/SebaJoe/AstroVisBench), and specify the value of `AstroVisBench_Env` in the environment variables.
    *   **Python Environment:** Due to the complexity of its Python environment, it is recommended to create a separate environment, install the project dependencies again, and then follow the official team's instructions to install the dependencies to avoid conflicts and slowing down the startup speed of testing other datasets.
    *   **Concurrency Settings:** Concurrency logic is set for dataset evaluation, with a default value of 4. This can be specified using `--judge-args '{"max_workers": <nums>}'`.
    *   **Evaluation Model:** This model requires Claude 4.5 Sonnet for evaluation, and the `ANTHROPIC_API_KEY` environment variable needs to be configured.
    *   **Evaluation Files:** The framework stores the model's inference results in `xlsx` format files by default for easy viewing. However, for AstroVisBench, some fields in the data may exceed the length limit of an `xlsx` cell. Therefore, you need to set the environment variable `PRED_FORMAT` to `json` or `tsv` (currently only these three formats are supported).
*   **SciCode:**
    *   **Environment Dependencies:** Before running, you need to download the runtime dependency file `test_data.h5` according to the [official instructions](https://github.com/scicode-bench/SciCode) and place it in the `scieval/dataset/SciCode/eval/data` directory. 
    *   **Evaluation Files:** By default, the framework stores the model's inference results in an `xlsx` format file for easy viewing. However, for SciCode, the output length of some models, such as `deepseek-R1`, may exceed the cell length limit of `xlsx`. In this case, you need to set the environment variable `PRED_FORMAT` to `json` or `tsv` (currently only these three formats are supported).
*   **SGI-Bench-1.0:**
    *   **Instructions for use：** See details at：`scieval/dataset/SGI_Bench_1_0/readme.md`

### Default Judge Models

The following datasets use specific models as default Judges:

| Dataset Name | Default Judge | Note |
| :--- | :--- | :--- |
| **SFE** | `gpt-4o-1120` | |
| **EarthSE** | `gpt-4o-1120` | |
| **ResearchbenchGenerate** | `gpt-4o-mini` | |
| **TRQA** | `chatgpt-0125` | Used only if rule parsing fails |
| **MaScQA** | `chatgpt-0125` | Used only if rule parsing fails |

---

## FAQ

### Building Input Prompt: `build_prompt()`

If the model output for a benchmark does not match expectations, it might be due to incorrect prompt construction.

In SciEvalKit, each dataset class has a `build_prompt()` function. For example, `ImageMCQDataset.build_prompt()` combines hint, question, and options into a standard format:

```text
HINT
QUESTION
Options:
A. Option A
B. Option B
···
Please select the correct answer from the options above.
```

SciEvalKit also supports **Model-Level custom prompt building** via `model.build_prompt()`.
*   **Priority:** `model.build_prompt()` overrides `dataset.build_prompt()`.

**Custom `use_custom_prompt()`:**
You can define `model.use_custom_prompt()` to decide when to use the model-specific prompt logic:

```python
def use_custom_prompt(self, dataset: str) -> bool:
    from scieval.dataset import DATASET_TYPE, DATASET_MODALITY
    dataset_type = DATASET_TYPE(dataset, default=None)
    
    if not self._use_custom_prompt:
        return False
    if listinstr(['MMVet'], dataset):
        return True
    if dataset_type == 'MCQ':
        return True
    return False
```

### Model Splitting & GPU Allocation

SciEvalKit supports automatic GPU resource division for `lmdeploy` or `transformers` backends.

*   **Python:** Defaults to all visible GPUs. Use `CUDA_VISIBLE_DEVICES` to restrict.
*   **Torchrun:**
    *   GPUs per instance = $N_{GPU} // N_{PROC}$.
    *   $N_{PROC}$: Process count from `-nproc-per-node`.
    *   $N_{GPU}$: Count of GPUs in `CUDA_VISIBLE_DEVICES` (or all if unset).

**Example (8 GPU Node):**

```bash
# 2 instances, 4 GPUs each
torchrun --nproc-per-node=2 run.py --data MaScQA --model InternVL3-78B

# 1 instance, 8 GPUs
python run.py --data MaScQA --model InternVL3-78B

# 3 instances, 2 GPUs each (GPUs 0 and 7 unused)
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc-per-node=3 run.py --data MaScQA --model InternVL3-38B
```

> **Note:** This does not apply to the `vllm` backend. For `vllm`, use the python command, which uses all visible GPUs by default.

### Deploying Local LLM as Judge

You can use LMDeploy to serve a local LLM as a judge replacement for OpenAI.

**1. Install**
```bash
pip install lmdeploy openai
```

**2. Serve (e.g., internlm2-chat-1.8b)**
```bash
lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
```

**3. Get Model ID**
```python
from openai import OpenAI
client = OpenAI(api_key='sk-123456', base_url="http://0.0.0.0:23333/v1")
print(client.models.list().data[0].id)
```

**4. Configure Env (in .env)**
```bash
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=<model_ID_you_got>
```

**5. Run Evaluation**
Execute `run.py` as normal.
