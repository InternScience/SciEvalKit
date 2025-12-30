# 快速开始

在运行评测脚本之前，你需要先配置 VLMs，并正确设置模型路径或者 API 的地址以及密钥。然后你可以使用脚本 `run.py` 并输入相关参数来进行多个 VLMs 和基准测试的推理和评估。

## 第 0 步：安装和设置必要的密钥

### 安装

```bash
git clone https://github.com/InternScience/SciEvalKit.git
cd SciEvalKit
pip install -e .
```

### 设置密钥

要使用 API 模型（如 GPT-4v, Gemini-Pro-V 等）进行推理，需要首先设置 API 密钥。

> **注意：** 部分数据集需要使用 LLM 作为评判者（Judge）并且设置了默认的评测模型（详见 *额外说明*），进行这些数据集的评测的时候也需要配置好相应的 API。

你可以将所需的密钥放在 `$SciEvalKit/.env` 中，或直接将它们设置为环境变量。如果你选择创建 `.env` 文件，其内容参考如下：

```bash
# .env 文件，将其放置在 $SciEvalKit 下

# --- 专有 VLMs 的 API 密钥 ---
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

# --- 评估专用设置 ---
# 你可以设置一个评估时代理，评估阶段产生的 API 调用将通过这个代理进行
EVAL_PROXY=
# 你也可以设置评估时专用的 KEY 和 BASE，只需在上面的变量之后加上 _EVAL 后缀即可，例如：
OPENAI_API_KEY_EVAL=
OPENAI_API_BASE_EVAL=
```

如果需要使用 API，请在对应键值空白处填写上你的密钥。这些 API 密钥将在进行推理和评估时自动加载。

---

## 第 1 步：配置

**VLM 配置：** 所有 VLMs 都在 `scieval/config.py` 中配置。对于某些 VLMs（如 MiniGPT-4、LLaVA-v1-7B），需要额外的配置（在配置文件中配置代码 / 模型权重根目录）。

在评估时，你应该使用 `scieval/config.py` 中 `supported_VLM` 指定的模型名称来选择 VLM。确保在开始评估之前，你可以成功使用 VLM 进行推理。

**检查命令：**

```bash
vlmutil check {MODEL_NAME}
```

---

## 第 2 步：评测

我们使用 `run.py` 进行评估。你可以使用 `$SciEvalKit/run.py` 或创建脚本的软链接运行（以便在任何地方使用该脚本）。

### 基本参数

*   `--data` (list[str]): 设置在 SciEvalKit 中支持的数据集名称（详见 `scieval/dataset/__init__.py` 或使用 `vlmutil dlist all` 查看）。
*   `--model` (list[str]): 设置在 SciEvalKit 中支持的 VLM 名称（在 `scieval/config.py` 中的 `supported_VLM` 中定义）。
*   `--mode` (str, 默认 `'all'`): 运行模式，可选值为 `['all', 'infer', 'eval']`。
    *   `"all"`: 执行推理和评估。
    *   `"infer"`: 只执行推理。
    *   `"eval"`: 只进行测评。
*   `--api-nproc` (int, 默认 4): 调用 API 的并发线程数。
*   `--work-dir` (str, 默认 `'.'`): 存放测试结果的目录。
*   `--config` (str): 配置 JSON 文件的路径。相比于指定 data 和 model，这是更为精细的配置方式（推荐）。详情见 *ConfigSystem*。

### 示例命令

你可以使用 `python` 或 `torchrun` 来运行脚本：

#### 1. 使用 python 运行
只实例化一个 VLM，并且它可能使用多个 GPU。这推荐用于评估参数量非常大的 VLMs（如 IDEFICS-80B-Instruct）。

```bash
# 在 MaScQA 和 ChemBench 上使用 IDEFICS-80B-Instruct 进行推理和评估
python run.py --data MaScQA ChemBench --model idefics_80b_instruct --verbose

# 在 MaScQA 和 ChemBench 上使用 IDEFICS-80B-Instruct 仅进行推理
python run.py --data MaScQA ChemBench --model idefics_80b_instruct --verbose --mode infer
```

#### 2. 使用 torchrun 运行
每个 GPU 上实例化一个 VLM 实例。这可以加快推理速度。但是，这仅适用于消耗少量 GPU 内存的 VLMs。

```bash
# 在 MaScQA 和 ChemBench 上使用 IDEFICS-9B-Instruct、Qwen-VL-Chat、mPLUG-Owl2
# 在具有 8 个 GPU 的节点上进行推理和评估
torchrun --nproc-per-node=8 run.py --data MaScQA ChemBench --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose

# 在 MaScQA 上使用 Qwen-VL-Chat。在具有 2 个 GPU 的节点上进行推理和评估
torchrun --nproc-per-node=2 run.py --data MaScQA --model qwen_chat --verbose
```

#### 3. API 模型评测

```bash
# 在 SFE 上使用 gpt-4o 进行推理和评估
# api并发量设置为32，需要设置openai的base url和key
# 注意：SFE 评估的时候默认也需要配置 OpenAI API
python run.py --data SFE --model GPT4o --verbose --api-nproc 32
```

#### 4. 使用 Config 文件

```bash
# 使用 config 进行评测，此时不可使用 --data 和 --model 指定评测数据集和模型
python run.py --config config.json
```

**结果：** 评估结果将作为日志打印出来。此外，结果文件也会在目录 `$YOUR_WORKING_DIRECTORY/{model_name}` 中生成。以 `.csv` 结尾的文件包含评估的指标。

---

## 额外设置

### 其他参数详解

*   `--judge` (str): 针对于评估阶段需要用到模型进行评估的数据集，设置评估模型。
    *   不指定则会使用配置好的默认模型。
    *   模型可以是 SciEvalKit 中支持的 VLM 名称，也可以指定自定义模型。
*   `--judge-args` (str): 设置评测时需要用到的参数（JSON 字符串格式）。
    *   当通过 `--judge` 指定评测模型时，可以传入如 `temperature`, `max_tokens` 等参数。
    *   具体参数视模型初始化的类决定（例如 `scieval.api.gpt.OpenAIWrapper`）。
    *   可以通过 `class` 参数指定模型实例化类（如 `OpenAIWrapper` 或 `Claude_Wrapper`）。
    *   也可以在此指定 model 属性配置评估模型，但优先级弱于 `--judge` 所指定的模型。
    *   *部分数据集需要独特的评测参数设置，详见下文 额外说明。*
*   `--reuse` (bool, 默认 `false`): 复用之前的结果。
*   `--ignore` (bool, 默认 `false`):
    *   默认情况下（`false`），当加载旧的推理结果时，如果发现里面包含异常信息（Failed），程序会重跑这些样本。
    *   如果开启此项（`true`），则会直接剔除这些失败的样本，只评测成功的样本。
*   `--fail-fast` (bool, 默认 `false`):
    *   当推理产生异常时，开启此项则会立即停止程序，而不是将异常信息写入推理结果。
    *   只对 API 推理方式有效。
*   `--ignore-patterns` (list[str]):
    *   和 `fail-fast` 配合使用。
    *   场景：开启了 fail-fast 但想忽略某些特定的非致命错误（如“内容违反安全政策”）。
    *   设置此参数指定允许忽略的异常字符串片段，命中这些模式的异常将被记录为结果而不是导致程序崩溃。
    *   *系统已默认配置了一些常见的违反安全政策的 patterns。*
*   `--stream` (bool, 默认 `false`):
    *   让模型以流式（Stream）输出。
    *   只对 API 推理有效。
    *   对于响应很慢的模型，为了防止 HTTP 连接超时，设置此参数很有必要。
    *   *Tip:* 使用 `--config` 时，也可以在配置文件中为单独为模型配置此参数，优先级高于命令行配置。

---

## 额外说明

### 特殊数据集配置

部分数据集在评测时有特殊要求：

*   **Clima_QA:**
    *   默认不计算 FA 分数。
    *   需要通过 `--judge-args '{"use_fa": true}'` 进行指定。
    *   此时需要用 LLM 进行评估（默认是 GPT-4）。
    *   可以通过 `--judge` 指定评测模型，但模型必须符合 OpenAI 格式同时确保配置好 Base URL 及 Key。
*   **PHYSICS:**
    *   评测时用到了模型评估，且使用方式和框架中提供的模型访问并不兼容。
    *   如果想使用其他模型评估而不是默认的 GPT-4o，需单独指定 `base_url`, `api_key`（默认读取环境中的 `OPENAI_API_KEY`, `OPENAI_API_BASE`，运行前需指定）。
*   **AstroVisBench:**
    *   **环境依赖：** 运行前需要按照 [官方说明](https://github.com/SebaJoe/AstroVisBench) 下载运行依赖环境，并在环境变量中指定 `AstroVisBench_Env` 的值。
    *   **Python 环境：** 由于其运行 Python 环境比较复杂，建议单独建立一个环境再次安装本项目依赖，然后按照官方团队的指示安装依赖，以免产生冲突和拖慢测试其他数据集的启动速度。
    *   **并发设置：** 数据集评测设置了并发逻辑，默认为 4，可通过 `--judge-args '{"max_workers": <nums>}'` 进行指定。
    *   **评测模型：** 该模型需要用到 Claude 4.5 Sonnet 进行评测，需要配置 `ANTHROPIC_API_KEY` 环境变量。
    *   **评测文件：** 框架默认将模型的推理结果存储在`xlsx`格式的文件中以方便查看，但是对于AstroVisBench来说数据中的某些字段会超出xlsx单元格的长度限制，需要设置环境变量PRED_FORMAT为`json`或`tsv`（目前只支持这三种格式）。
*   **SciCode:**
    *   **环境依赖：** 运行前需要按照 [官方说明](https://github.com/scicode-bench/SciCode) 下载运行依赖文件`test_data.h5`，并放置在`scieval/dataset/SciCode/eval/data`目录下。  
    *   **评测文件：** 框架默认将模型的推理结果存储在`xlsx`格式的文件中以方便查看，但是对于SciCode来说部分模型如`deepseek-R1`的输出长度可能会超出xlxs单元格长度限制，此时需要设置环境变量PRED_FORMAT为`json`或`tsv`（目前只支持这三种格式）
*   **SGI-Bench-1.0:**
    *   **运行说明：** 详情见：`scieval/dataset/SGI_Bench_1_0/readme.md`

### 默认评判模型列表

以下数据集在评估阶段默认使用特定的模型作为 Judge：

| 数据集名称 | 默认评判模型 (Default Judge) | 备注 |
| :--- | :--- | :--- |
| **SFE** | `gpt-4o-1120` | |
| **EarthSE** | `gpt-4o-1120` | |
| **ResearchbenchGenerate** | `gpt-4o-mini` | |
| **TRQA** | `chatgpt-0125` | 仅在规则解析失败时使用 |
| **MaScQA** | `chatgpt-0125` | 仅在规则解析失败时使用 |

---

## 常见问题

### 构建输入 Prompt：`build_prompt()` 函数

如果您在评测某个 benchmark 时，发现模型输出的结果与预期不符，可能是因为您使用的模型没有正确构建输入 prompt。

在 SciEvalKit 中，每个 dataset 类都包含一个名为 `build_prompt()` 的函数，用于构建输入问题的格式。不同的 benchmark 可以选择自定义 `build_prompt()` 函数，也可以使用默认的实现。

例如，在处理默认的多选题/Multi-Choice QA 时，`ImageMCQDataset.build_prompt()` 类会将 hint、question、options 等元素组合成如下格式：

```text
HINT
QUESTION
Options:
A. Option A
B. Option B
···
Please select the correct answer from the options above.
```

此外，SciEvalKit 也支持在模型层面自定义对不同 benchmark 构建 prompt 的方法，即 `model.build_prompt()`。
*   **优先级：** 当同时定义了 `model.build_prompt()` 以及 `dataset.build_prompt()` 时，`model.build_prompt()` 将优先于 `dataset.build_prompt()`。

**自定义 `use_custom_prompt()`：**
为了更灵活地适应不同的 benchmark，SciEvalKit 支持在模型中自定义 `model.use_custom_prompt()` 函数来决定何时使用模型特定的 Prompt。示例如下：

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

### 模型切分与 GPU 分配

SciEvalKit 支持在同机上进程间自动划分 GPU 资源（支持 `lmdeploy` 或 `transformers` 后端）。

*   **Python 启动：** 默认使用所有可用 GPU。使用 `CUDA_VISIBLE_DEVICES` 指定特定 GPU。
*   **Torchrun 启动：**
    *   每个模型实例分配的 GPU 数量 = $N_{GPU} // N_{PROC}$。
    *   $N_{PROC}$: torchrun 参数 `-nproc-per-node` 指定的进程数。
    *   $N_{GPU}$: `CUDA_VISIBLE_DEVICES` 指定的 GPU 数量（未设置则为全部可用数量）。

**示例（8 GPU 机器）：**

```bash
# 起两个模型实例数据并行，每个实例用 4 GPU
torchrun --nproc-per-node=2 run.py --data MaScQA --model InternVL3-78B

# 起一个模型实例，每个实例用 8 GPU
python run.py --data MaScQA --model InternVL3-78B

# 起三个模型实例，每个实例用 2 GPU，0 号、7 号 GPU 未被使用
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --nproc-per-node=3 run.py --data MaScQA --model InternVL3-38B
```

> **注：** 此方式不支持 `vllm` 后端。基于 `vllm` 后端起评测任务时，请用 python 命令启动，默认调用所有可见的 GPU。

### 部署本地语言模型作为评判 (Local Judge)

你可以使用 LMDeploy 部署本地 LLM 来替代 OpenAI GPT 作为评判。

**1. 安装**
```bash
pip install lmdeploy openai
```

**2. 部署（示例：internlm2-chat-1.8b）**
```bash
lmdeploy serve api_server internlm/internlm2-chat-1_8b --server-port 23333
```

**3. 获取模型 ID**
```python
from openai import OpenAI
client = OpenAI(api_key='sk-123456', base_url="http://0.0.0.0:23333/v1")
print(client.models.list().data[0].id)
```

**4. 配置环境（在 .env 中）**
```bash
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=<你获取到的模型ID>
```

**5. 运行评测**
执行正常的 `run.py` 命令即可。
