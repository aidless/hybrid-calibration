# PREREG — hybrid-calibration (pilot 阶段)

> 预注册规格：本文件声明**本仓库在 pilot 阶段可检验的研究假设、评测方案、
> 成功标准与明确边界**。内容全部对应仓库真实实现与真实结果
> （`calibration.py`/`models.py`/`experiment.py`/`data_loader.py`、
> `config.py` 的 `RANDOM_SEEDS`、`results/newsgroups_*.json` 的真实
> ECE/MCE/Brier/accuracy、`tests/test_committed_results.py` 的 3 项离线测试），
> 不含虚构数字或未实现能力。
>
> 状态：**quick test only**（2 seeds、1 dataset、4/14 models）。这是 pilot，
> 不是结论；禁止引用为已发表结果（见 DEEP_DIVE.md 的 blocking issues）。

## 1. 研究问题

把 LLM embedding（768 维 GPT-2）加入树集成（14 维表格特征）会不会**买来校准、
却损失准确率**？混合视图会以纯准确率指标掩盖的方式改变模型的校准轮廓
（ECE / MCE / Brier / reliability）。本仓库直接测量它。

## 2. 假设（H1–H3，全部可证伪）

- **H1（纯表格高准确、中校准）**：14 维表格特征给出高准确率（~0.80）与中等
  校准（ECE ~0.07）；RF-Tabular 是准确率最高的一档。
- **H2（纯 embedding 校准极好但接近无用）**：纯 768 维 embedding 模型校准极好
  （ECE ~0.007）但准确率接近猜（~0.247），不能作为预测器。
- **H3（hybrid 校准退化）**：hybrid 保留表格准确率（~0.79）但校准**比纯表格差**
  （ECE 0.087 vs 0.074）——这是纯准确率指标隐藏的「隐形代价」。

## 3. 评测方案

- **特征集**：tabular（14 维）| embeddings（768 维 GPT-2）| hybrid（并置）。
- **模型**：RF / XGB / LGB / MLP，加 Platt 与 isotonic 事后校准。
- **设计**：14 模型变体 × 10 seeds × 4 datasets = 至多 560 次训练；
  统计分析用**配对 Wilcoxon + Bonferroni 校正**。
- **pilot 已跑**：2 seeds、1 dataset（newsgroups）、4/14 模型（RF-Tabular /
  RF-Hybrid / RF-Embed / MLP-Embed），结果落 `results/newsgroups_*.json`。

## 4. 成功标准

1. `tests/test_committed_results.py` 3 项离线测试全过：每个 committed 结果带
   可复现元数据（dataset / n_seeds / model_variants）；ECE/MCE/Brier/accuracy
   均值与标准差在合法范围；README pilot 表与具名 JSON 源**字节级一致**。
2. pilot 表四组数字（RF-Tabular 0.074/0.351/0.273/0.796 等）可从
   `newsgroups_20260626_232910.json` 的 `aggregated` 逐字段重算（round 3）。
3. 统计检验：n=2 seeds 时配对 Wilcoxon **不可行**——如实披露，不假装有显著性。

## 5. 明确边界（不证明什么）

- **不证明混合 LLM embedding 造成校准退化**：2 seeds 无法从噪声中分离真实
  效应；n=2 的配对检验接近无信息。**禁止引用 pilot 数字**。
- **不证明跨数据集泛化**：仅 1 个数据集（newsgroups），需 ≥3（IMDB + AG News +
  Newsgroups）。
- **不证明完整模型矩阵**：仅 4/14 模型变体；需至少 Tabular/Hybrid/Embed ×
  RF/XGB/LGB + MLP。
- **不证明使用真实 embedding**：当前 `--skip-embeddings` 为 True，embedding 是
  合成的；需真实 GPT-2 embedding 才可发表。
- **不证明 GPT-4o 代际**：用的是 GPT-2（合理但需在 limitation 中说明代际差异）。

## 6. 数据与复现

- 复现（无需重跑训练）：
  ```bash
  python3 -m pytest tests/ -q      # 3 passed（纯离线，不重跑模型训练）
  ```
- 全量跑（需本地算力，~2-4 小时 CPU）：
  ```bash
  python3 experiment.py --dataset imdb --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
  python3 experiment.py --dataset ag_news --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
  python3 experiment.py --dataset newsgroups --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
  ```
