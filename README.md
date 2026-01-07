<h1 align="center"><img src="assets/icon/opencompass.png" alt="OpenCompass" height="50" style="vertical-align:middle;" />&nbsp;SciEval ToolKit</h1>

<p align="center"><strong>
A unified evaluation toolkit and leaderboard for rigorously assessing the scientific intelligence of large language and vision–language models across the full research workflow.
</strong></p>

<hr style="width:100%;margin:16px 0;border:0;border-top:0.1px solid #d0d7de;" />

<div align="center">

[![Tutorial](https://img.shields.io/badge/Tutorial-SciEval-b8dcff?style=for-the-badge&logo=google-chrome&logoColor=white)](https://scievalkit-docs.readthedocs.io/en/latest)&#160;
[![Leaderboard](https://img.shields.io/badge/LEADERBOARD-Scieval-f6e58d?style=for-the-badge&logo=huggingface)](https://opencompass.org.cn/Intern-Discovery-Eval/rank)&#160;
[![Report](https://img.shields.io/badge/REPORT-Technical-f4c2d7?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2512.22334)&#160;
[![GitHub](https://img.shields.io/badge/GitHub-Repository-c7b9e2?style=for-the-badge&logo=github&logoColor=white)](https://github.com/InternScience/SciEvalKit)

<img src="assets/icon/welcome.png" alt="welcome" height="24" style="vertical-align:middle;" />
&nbsp;Welcome to the official repository of <strong>SciEval</strong>!

<div align="center">
  <img src="assets/SciEvalKit.png" alt="SciEval capability radar" width="90%">
</div>

</div>

## <img src="assets/icon/why.png" alt="why" height="28" style="vertical-align:middle;" />&nbsp;Why SciEval?

**SciEval** is an open‑source evaluation framework and leaderboard aimed at measuring the **scientific intelligence** of large language and vision–language models.  
Although modern frontier models often achieve *~90* on general‑purpose benchmarks, their performance drops sharply on rigorous, domain‑specific scientific tasks—revealing a persistent **general‑versus‑scientific gap** that motivates the need for SciEval.
Its design is shaped by following core ideas:

- **Beyond general‑purpose benchmarks ▸** Traditional evaluations focus on surface‑level correctness or broad‑domain reasoning, hiding models’ weaknesses in realistic scientific problem solving.  SciEval makes this **general‑versus‑scientific gap** explicit and supplies the evaluation infrastructure needed to guide the integration of broad instruction‑tuned abilities with specialised skills in coding, symbolic reasoning and diagram understanding.
- **End‑to‑end workflow coverage ▸** SciEval spans the full research pipeline—such as **image interpretation, symbolic reasoning, executable code generation, and hypothesis generation**—instead of isolated subtasks.  
- **Capability‑oriented & reproducible ▸** A unified toolkit for **dataset construction, prompt engineering, inference, and expert‑aligned scoring** ensures transparent and repeatable comparisons.  
- **Grounded in real scenarios ▸** Benchmarks use domain‑specific data and tasks so performance reflects **actual scientific practice**, not synthetic proxies.

For a detailed and systematic introduction to SciEvalKit, please refer to the [SciEvalKit Tutorial](https://scievalkit-docs.readthedocs.io/en/latest).


## <img src="assets/icon/progress.png" alt="progress" height="28" style="vertical-align:middle;" />&nbsp;Progress in Scientific Intelligence

*Realtime updates — scores are synchronized with the [Intern‑Discovery‑Eval](https://opencompass.org.cn/Intern-Discovery-Eval/rank) leaderboard.*

<div align="center">
  <img src="assets/general_scientific_comparison.png" alt="SciEval capability radar" width="100%">
</div>


- **General benchmarks overestimate scientific competence.** Even the strongest frontier models (e.g., **Gemini 3 Pro**) score below **60** on **Scientific Text Capability** , despite scoring near *90* on widely used general‑purpose benchmarks.
- **Multimodal capability is breaking the 60‑point barrier.** **Gemini 3 Pro** leads **Scientific Multimodal Capability** with **62.88**, reflecting strong performance in multimodal perception and reasoning.
- **Open‑source systems are rapidly closing the gap.** *Qwen3‑VL‑235B‑A22B* and *Qwen3‑Max* now match or surpass several proprietary models in symbolic reasoning and code generation, signalling healthy community progress.
- **Symbolic reasoning and code generation remain bottlenecks.** No model exceeds **50** in equation‑level manipulation or **30** in end‑to‑end executable code tasks, indicating that scientific workflows requiring programmatic pipelines still fail frequently.


## <img src="assets/icon/key.png" alt="key" height="28" style="vertical-align:middle;" />&nbsp;Key Features

<div align="center">
  <img src="assets/radar.png" alt="SciEval capability radar" width="70%">
</div>


| Category                                  | Highlights                                                                                                                                                       |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Seven Core Dimensions**            | Scientific Knowledge Understanding, Scientific Code Generation, Scientific Symbolic Reasoning, Scientific Hypothesis Generation, Scientific Multimodal Perception, Scientific Multimodal Reasoning, Scientific Multimodal Understanding |
| **Discipline Coverage**             | Life Science • Astronomy • Earth Science • Chemistry • Materials Science • Physics.                                                                         |
| **Multimodal & Executable Scoring** | Supports text, code, and image inputs; integrates code tasks and LLM-judge fallback for open-ended answers.                                                      |
| **Reproducible & Extensible**       | Clear dataset and model registries, minimised hard-coding and modular evaluators make new tasks or checkpoints easy to plug in.                                  |

<div align="center">
  <img src="assets/framework.png" alt="SciEval framework overview" width="65%">
</div>

<p align="left">
  <em>
  An overview of the SciEval framework, illustrating how heterogeneous scientific datasets, unified prompt construction, model inference, and capability-oriented evaluators are integrated into a single reproducible evaluation pipeline.
  </em>
</p>


## <img src="assets/icon/news.png" alt="news" height="28" style="vertical-align:middle;" />&nbsp;News
* **[2025‑12‑12] · 📰 Evaluation Published on OpenCompass**
  - SciEval’s benchmark results are now live on the [OpenCompass](https://opencompass.org.cn/Intern-Discovery-Eval) platform, providing broader community visibility and comparison.

* **[2025‑12‑05] · 🚀 SciEval v1 Launch**
  - Initial public release of a science‑focused evaluation toolkit and leaderboard devoted to realistic research workflows.
  - Coverage: seven scientific capability dimensions × six major disciplines in the initial benchmark suite.

* **[2025‑12‑05] · 🌟 Community Submissions Open**
  - Submit your benchmarks via pull request to appear on the official leaderboard.

## <img src="assets/icon/start.png" alt="start" height="28" style="vertical-align:middle;" />&nbsp;Quick Start

Get from clone to first scores in minutes&mdash;see our local [QuickStart](docs/en/Quickstart.md) / [快速开始](docs/zh-CN/Quickstart.md) guides, or refer to the [SciEvalKit Tutorial](https://scievalkit-docs.readthedocs.io/en/latest/Quickstart.html) for additional guidance.

### 1 · Install

```bash
git clone https://github.com/InternScience/SciEvalKit.git
cd SciEvalKit
pip install -e .[all]    # brings in vllm, openai‑sdk, hf_hub, etc.
```

### 2 · (Optional) add API keys

Create a `.env` at the repo root **only if** you will call API models or
use an LLM‑as‑judge backend:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
DASHSCOPE_API_KEY=...
```

If no keys are provided, SciEval falls back to rule‑based scoring
whenever possible.

### 3 · Run a API demo test

```bash
python run.py \
  --dataset SFE \
  --model gpt-4o \
  --mode all \
  --work-dir outputs/demo_api \
  --verbose
```

### 4 · Evaluate a local/GPU model

```bash
python run.py \
  --dataset MaScQA \
  --model qwen_chat \
  --mode infer \
  --work-dir outputs/demo_qwen \
  --verbose

# ➜ Re‑run with --mode all after adding an API key
#     if the benchmark requires an LLM judge.
```

## <img src="assets/icon/update.png" alt="update" height="28" style="vertical-align:middle;" />&nbsp;Codebase Updates

* **Execution‑based Scoring**
  - Code‑generation tasks (SciCode, AstroVisBench) are now graded via sandboxed unit tests.

---

## 📬 Contact Us

- 💬 **GitHub Issues**: Please open an issue for bug reports or feature requests

- 🤝 **Community**: 

<p align="center">
  <img src="https://raw.githubusercontent.com/InternScience/SGI-Bench/main/assets/wechat.jpg" alt="WeChat" width="200">
</p>

---

## <img src="assets/icon/thanks.png" alt="thanks" height="30" style="vertical-align:middle;" />&nbsp;Acknowledgements

SciEval ToolKit is built on top of the excellent **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)** framework and we thank the OpenCompass team not only for open‑sourcing their engine, but also for publishing thorough deployment and development guides ([Quick Start](https://vlmevalkit.readthedocs.io/en/latest/Quickstart.html), [Development Notes](https://vlmevalkit.readthedocs.io/en/latest/Development.html)) that streamlined our integration.

We also acknowledge the core SciEval contributors for their efforts on dataset curation, evaluation design, and engine implementation: Jun Yao, Han Deng, Yizhou Wang, Jiabei Xiao, Jiaqi Liu, Encheng Su, Yujie Liu, Weida Wang, Junchi Yao, Haoran Sun, Runmin Ma, Bo Zhang, Dongzhan Zhou, Shufei Zhang, Peng Ye, Xiaosong Wang, and Shixiang Tang, as well as all community testers who provided early feedback.

SciEvalKit contributors can join the author list of the report based on their contribution to the repository. Specifically, it requires 3 major contributions (implement a new benchmark, foundation model, or contribute a major feature). We will update the report quarterly and an additional section that details each developer’s contribution will be appended in the next update.
