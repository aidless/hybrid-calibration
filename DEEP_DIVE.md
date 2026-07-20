# 🔬 Hybrid Calibration — Deep Dive Analysis

**Project**: `hybrid_calibration/`
**Question**: How does combining LLM embeddings with tree ensembles affect probability calibration?
**Status**: Quick-test only (2 seeds, 1 dataset, 4/14 models)

---

## 1. Architecture Summary

```
Data Pipeline:
  Text → [Tabular Features (14 dims)] + [GPT-2 Embeddings (768 dims)]
         → 3 Feature Sets: tabular | embeddings | hybrid (tabular+embeddings)
              → Model: RF / XGB / LGB / MLP / Calibrated(Platt/Isotonic)
                   → Metrics: ECE, MCE, Brier, Accuracy, Reliability

Experiment Design:
  14 model variants × 10 seeds × 4 datasets = up to 560 training runs
  Statistical: Wilcoxon paired test + Bonferroni correction
```

## 2. Current Results (Quick Test)

| Model | ECE ↓ | MCE ↓ | Brier ↓ | Acc ↑ |
|-------|:-----:|:-----:|:-----:|:-----:|
| RF-Tabular (纯表格) | 0.074 | 0.351 | 0.273 | **0.796** |
| RF-Hybrid (表格+LLM) | 0.087 | 0.334 | 0.280 | 0.790 |
| RF-Embed (纯LLM) | **0.007** | **0.008** | 0.750 | 0.247 |
| MLP-Embed (纯LLM) | 0.006 | 0.006 | 0.750 | 0.247 |

**关键模式**：
- 纯表格 → 高准确率、中等校准
- 纯 LLM → 极好校准、极差准确率（模型几乎在猜）
- **Hybrid → 准确率≈表格，校准比表格差** ← 这就是"隐形代价"

## 3. Current Limitations (Blocking Publication)

| 问题 | 严重度 | 修复 |
|------|:------:|------|
| 仅 2 seeds | 🔴 Critical | 需 ≥10 seeds（配置已支持） |
| 仅 1 数据集 | 🔴 Critical | 需 ≥3 datasets（IMDB + AG News + Newsgroups） |
| 仅 4/14 模型 | 🟡 Major | 至少跑 Tabular/Hybrid/Embed 三组 × RF/XGB/LGB + MLP |
| 合成 embedding | 🟡 Major | 需真实 GPT-2 embeddings（`--skip-embeddings` 当前为 True） |
| 无显著性检验 | 🔴 Critical | 2 seeds 无法做 Wilcoxon，需 ≥5 seeds |
| 用 GPT-2 而非 GPT-4o | 🟡 Minor | GPT-2 是合理的，但需在 limitation 中说明代际差异 |

## 4. Strengthening Plan

### Phase H1: Full Run (最低可发表标准)
```bash
python experiment.py --dataset imdb --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
python experiment.py --dataset ag_news --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
python experiment.py --dataset newsgroups --models RF-Tabular RF-Embed RF-Hybrid MLP-Embed
```
- 3 datasets × 4 models × 10 seeds = 120 training runs
- 需真实 GPT-2 embeddings（去掉 `--skip-embeddings`）
- 时间：~2-4 小时（取决于 CPU）
- 成本：$0（全部本地计算）

### Phase H2: Full Matrix (最强版本)
加上全部 14 个模型变体：
- + XGB-Tabular, LGB-Tabular（更多树模型）
- + XGB-Embed, LGB-Embed（更多 embedding 模型）
- + XGB-Hybrid, LGB-Hybrid（更多 hybrid 模型）
- + RF-Hybrid-Platt, RF-Hybrid-Isotonic（事后校准）
- + MLP-Embed-Platt, MLP-Embed-Isotonic（事后校准）

## 5. Connection to BOUNDARY_SYNC

### Narrative Arc (跨论文)
```
BOUNDARY_SYNC:   LLM 通信 → 输出同质化 (隐形代价 #1)
Hybrid Cal:      LLM 特征 → 校准退化   (隐形代价 #2)

共同主题: LM 集成方式引入标准指标(Accuracy/F1)捕捉不到的副作用
共同方法: Bootstrap CI + 配对检验 + 效应量
```

### 论文中可引用的具体位置

**BOUNDARY_SYNC Related Work (§2.3/§2.4)**：
> "Recent work has identified a parallel hidden cost in a different LLM integration paradigm: concatenating frozen LLM embeddings with tabular features for tree-ensemble classification degrades probability calibration without improving accuracy — an effect invisible to standard metrics [cite hybrid_calibration]. This pattern of LLM integration producing 'silent side effects' motivates our systematic measurement of communication-induced coupling."

**BOUNDARY_SYNC Discussion (§5.2)**：
> "Our finding that communication homogenizes LM outputs joins a growing body of evidence that LLM integration can produce metric-invisible side effects. For instance, [cite hybrid_calibration] showed that LLM features degrade calibration in downstream classifiers without changing accuracy. Both cases underscore the need for measurement protocols that go beyond task performance."

### Integration Priority
| 行动 | 优先级 | 成本 | BOUNDARY_SYNC 收益 |
|------|:------:|:----:|:-------------------|
| H1: 跑 3 datasets × 4 models | ❶ | ~3h CPU | 可正式引用 |
| H2: 跑全部 14 models + calibration | ❷ | ~6h CPU | 可做主要引用 + 校准修复方案 |
| 写入 BOUNDARY_SYNC Related Work | ❸ | 10min | 1 段即可 |
| 写入 BOUNDARY_SYNC Discussion | ❹ | 5min | 1 句即可 |

## 6. Quick Start

```powershell
cd C:\Users\Administrator\ZCodeProject\hybrid_calibration

# Install deps if needed
pip install torch transformers xgboost lightgbm scikit-learn scipy matplotlib seaborn datasets

# Run full experiment (real GPT-2 embeddings + 10 seeds)
python experiment.py --dataset newsgroups
python experiment.py --dataset imdb
python experiment.py --dataset ag_news

# Generate figures
python visualize.py results/newsgroups_*.json
```
