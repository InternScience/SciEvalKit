from __future__ import annotations

from .text_base import TextBaseDataset
from .utils.human_ppi import evaluate_human_ppi_binary
from ..smp import *


class HumanPPIDataset(TextBaseDataset):
    """
    Human PPI 文本数据集（用于简单 Yes/No 分类测试）。

    - 数据来源：LMUDataRoot()/human_ppi_test.tsv
      其列包含：
        - index: 样本索引
        - question: 模型输入（自然语言 PPI 问题）
        - answer: 参考回答（可选，用于人工检查）
        - category: 标签（0/1，是否存在 PPI）
    - 评测方式：不依赖额外 tokenizer / OpenAI judge，仅做基础准确率统计。
    """

    TYPE = "QA"

    # 显式声明支持的数据集名称，确保在 DATASET_CLASSES / SUPPORTED_DATASETS 中可见
    DATASET_URL = {"human_ppi_test": ""}
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        # 覆盖基类逻辑，直接返回我们支持的名字
        return list(cls.DATASET_URL.keys())

    def load_data(self, dataset):
        """
        不从远程下载，直接从 LMUDataRoot 下读取本地 tsv：
            <LMUDataRoot>/human_ppi_test.tsv
        """
        data_path = osp.join(LMUDataRoot(), f"{dataset}.tsv")
        if not osp.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} does not exist.")
        return load(data_path)

    def build_prompt(self, line):
        """
        直接复用 TextBaseDataset 的简单文本提示：
        prompt 内容就是 `question` 字段。
        """
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line["question"]
        return [dict(type="text", value=question)]

    def evaluate(self, eval_file, **judge_kwargs):
        """
        基础评测：
        - 不调用外部大模型充当 judge；
        - 只把 prediction 文本解析为 0/1，与 `category` 做准确率统计。
        """
        _ = judge_kwargs  # 暂不使用，避免未使用变量告警
        return evaluate_human_ppi_binary(eval_file, self.data)


