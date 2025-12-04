# Researchbench

本数据集支持三类子任务：

* **Retrieve 任务**：从候选文献中筛选灵感论文。
* **Generate 任务**：基于科研背景生成新的研究想法或摘要。
* **Rank 任务**：对候选研究结果进行排序或偏好评估。



## 一、Retrieve 任务


### 1. 输出结果示例

```text
Evaluation Results:
{
    "dataset": "researchbench_retrieve",
    "size": 23,
    "weighted": false,
    "hit@1": 0.087,
    "hit@3": 0.174
}
```

### 2. 说明

* `hit@1` 与 `hit@3` 表示前 1 篇 / 前 3 篇命中率。
---

## 二、Generate 任务

### 1. 输出结果示例

```text
Evaluation Results:
{
    "items_scored": 1084,
    "avg_score": 2.615,
    "judged_file": "outputs/.../.../judged.xlsx",
    "judge_model": "gpt-4o-mini"
}
```
### 2. 说明

* `avg_score` 表示生成任务的平均得分，范围通常为 0–5。
* `score_dist` 给出了不同分值的分布情况。
* `judged_file` 为评测结果文件路径，包含每个样本的详细得分。

---

## 三、Rank 任务

### 1. 输出结果示例

```text
Evaluation Results:
{
    "overall": {
        "num_pairs": 195,
        "num_parsable": 195,
        "pairwise_acc": 0.728,
        "mean_rank_position": 5.077,
        "mean_rank16": 11.923,
        "mean_rank_score": 0.683
    }
}
```

### 2. 说明

* `pairwise_acc`：两两比较准确率，用于衡量模型排序的一致性。
* `mean_rank_position`：平均排名位置。
* `mean_rank_score`：综合排名得分。
