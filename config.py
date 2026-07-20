"""
Experiment configuration for Hybrid Tree-Ensemble + LLM Calibration Study

Research Question: How does combining LLM embeddings with tree ensembles
affect probability calibration compared to pure approaches?
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================
# Global Settings
# ============================================================
RANDOM_SEEDS = list(range(10))  # 10 seeds for statistical rigor
N_BINS = 10                      # Equal-width bins for ECE
TEST_SIZE = 0.2
VAL_SIZE = 0.1                   # From training portion

# ============================================================
# Datasets
# ============================================================
# Small subsets for CPU-friendly experimentation
DATASET_CONFIGS = {
    "imdb": {
        "name": "imdb",
        "task": "binary",
        "n_samples": 5000,       # Subset for CPU
        "text_col": "text",
        "label_col": "label",
        "num_classes": 2,
    },
    "ag_news": {
        "name": "ag_news",
        "task": "multiclass",
        "n_samples": 5000,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 4,
    },
    "yelp_polarity": {
        "name": "yelp_polarity",
        "task": "binary",
        "n_samples": 5000,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 2,
    },
    "newsgroups": {
        "name": "newsgroups",
        "task": "multiclass",
        "n_samples": 3000,
        "text_col": "text",
        "label_col": "label",
        "num_classes": 8,
    },
}


@dataclass
class ModelConfig:
    """Configuration for a single model variant."""

    name: str
    """Human-readable name."""

    model_type: str
    """rf | xgb | lgb | mlp | calibrated"""

    feature_set: str
    """tabular | embeddings | hybrid"""

    base_model: Optional[str] = None
    """Underlying model to calibrate (for calibrated variants)."""

    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1

    # MLP-specific
    hidden_dim: int = 256
    dropout: float = 0.3
    mlp_epochs: int = 200

    # Calibration
    calibration_method: Optional[str] = None
    """platt | isotonic | None"""


# ============================================================
# Model Variants (Experiment Design)
# ============================================================
MODEL_VARIANTS: List[ModelConfig] = [
    # --- Tabular-only baselines ---
    ModelConfig(name="RF-Tabular", model_type="rf", feature_set="tabular"),
    ModelConfig(name="XGB-Tabular", model_type="xgb", feature_set="tabular"),
    ModelConfig(name="LGB-Tabular", model_type="lgb", feature_set="tabular"),

    # --- Embedding-only tree ensembles ---
    ModelConfig(name="RF-Embed", model_type="rf", feature_set="embeddings"),
    ModelConfig(name="XGB-Embed", model_type="xgb", feature_set="embeddings"),
    ModelConfig(name="LGB-Embed", model_type="lgb", feature_set="embeddings"),

    # --- Hybrid (embeddings + tabular) ---
    ModelConfig(name="RF-Hybrid", model_type="rf", feature_set="hybrid"),
    ModelConfig(name="XGB-Hybrid", model_type="xgb", feature_set="hybrid"),
    ModelConfig(name="LGB-Hybrid", model_type="lgb", feature_set="hybrid"),

    # --- MLP on embeddings (LLM proxy baseline) ---
    ModelConfig(name="MLP-Embed", model_type="mlp", feature_set="embeddings",
                hidden_dim=256, dropout=0.3, mlp_epochs=200),

    # --- Calibrated variants of the best-performing models ---
    ModelConfig(name="RF-Hybrid-Platt", model_type="calibrated", feature_set="hybrid",
                base_model="RF-Hybrid", calibration_method="platt"),
    ModelConfig(name="RF-Hybrid-Isotonic", model_type="calibrated", feature_set="hybrid",
                base_model="RF-Hybrid", calibration_method="isotonic"),
    ModelConfig(name="MLP-Embed-Platt", model_type="calibrated", feature_set="embeddings",
                base_model="MLP-Embed", calibration_method="platt"),
    ModelConfig(name="MLP-Embed-Isotonic", model_type="calibrated", feature_set="embeddings",
                base_model="MLP-Embed", calibration_method="isotonic"),
]

# ============================================================
# Embedding Extraction — Multi-Model Comparison
# ============================================================

# All available embedding models (local cache paths)
EMBEDDING_MODELS = {
    "bge-zh": {
        "name": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "language": "zh",
        "source": "modelscope",
        "cache_path": "~/.cache/modelscope/BAAI/bge-small-zh-v1___5",
    },
    "minilm-en": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "language": "en",
        "source": "huggingface",
        "cache_path": "sentence-transformers/all-MiniLM-L6-v2",
    },
    # Qwen-0.5B embeddings (extract from generation model)
    # "qwen-0.5b": {
    #     "name": "Qwen/Qwen2.5-0.5B-Instruct",
    #     "dim": 896,
    #     "language": "zh/en",
    #     "source": "huggingface",
    #     "cache_path": "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*",
    # },
}

# Default embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  # 512-dim Chinese embedding
EMBEDDING_DIM = 512
MAX_SEQ_LENGTH = 256
BATCH_SIZE_EMBED = 16

# Sample-size sweep (EPC-style N-sensitivity ablation)
SAMPLE_SIZE_SWEEP = [500, 1000, 2000, 5000]

# ============================================================
# Tabular Feature Extraction
# ============================================================
TABULAR_FEATURES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length",
    "punctuation_ratio",
    "capital_ratio",
    "digit_ratio",
    "unique_word_ratio",       # TTR (Type-Token Ratio)
    "stopword_ratio",
    "exclamation_count",
    "question_count",
    "url_count",
    "has_quotes",
]

# ============================================================
# Paths
# ============================================================
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
