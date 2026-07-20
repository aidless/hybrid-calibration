"""
Data loading, preprocessing, and tabular feature extraction.

Provides:
  - load_dataset(): Load a text dataset (IMDb, AG News, Yelp, 20news)
  - extract_tabular_features(): Generate hand-crafted features from text
  - prepare_data(): Full pipeline: load -> features -> split

Supports both HuggingFace download and local sklearn fallback.
"""

import os
import ssl
import re
import pickle
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_CONFIGS,
    TABULAR_FEATURES,
    TEST_SIZE,
    VAL_SIZE,
    DATA_DIR,
)

# Fix SSL certificate issues on Windows
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Disable HuggingFace SSL warning
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"


# ============================================================
# Dataset Loading
# ============================================================

def load_dataset(dataset_name: str, n_samples: Optional[int] = None):
    """
    Load a text classification dataset.

    Args:
        dataset_name: One of 'imdb', 'ag_news', 'yelp_polarity', 'newsgroups'.
        n_samples: Subset size (stratified). None = full dataset.

    Returns:
        texts: List[str]
        labels: np.ndarray (int)
        class_names: List[str]
    """
    cfg = DATASET_CONFIGS.get(dataset_name, DATASET_CONFIGS["imdb"])

    data_source = "real"  # Default for downloadable datasets

    # Try loading from cache first
    cache_path = os.path.join(DATA_DIR, f"{dataset_name}_cached.pkl")
    if os.path.exists(cache_path):
        print(f"  [DataLoader] Loading cached: {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        texts, labels, class_names = cached["texts"], cached["labels"], cached["class_names"]
        data_source = cached.get("data_source", "cached_unknown")
    else:
        try:
            if dataset_name == "imdb":
                texts, labels, class_names, data_source = _load_imdb()
            elif dataset_name == "ag_news":
                texts, labels, class_names, data_source = _load_ag_news()
            elif dataset_name == "yelp_polarity":
                texts, labels, class_names, data_source = _load_yelp()
            elif dataset_name == "newsgroups":
                texts, labels, class_names, data_source = _load_newsgroups()
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}")

            # Cache (include data source flag)
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "texts": texts, "labels": labels, "class_names": class_names,
                    "data_source": data_source,
                }, f)
            print(f"  [DataLoader] Cached to: {cache_path}")

        except Exception as e:
            print(f"  [DataLoader] Download failed ({type(e).__name__}), using synthetic data ...")
            texts, labels, class_names, data_source = _load_synthetic(n_samples=n_samples or 2000)

    # Stratified subset
    if n_samples is not None and n_samples < len(texts):
        texts, labels = _stratified_subset(texts, labels, n_samples)

    print(f"  [DataLoader] Loaded {len(texts)} samples, {len(np.unique(labels))} classes  [source: {data_source}]")
    return texts, labels, class_names, data_source


def _load_imdb():
    """Load IMDb sentiment dataset from HuggingFace."""
    from datasets import load_dataset as hf_load
    import requests
    # Disable SSL verification for this session
    old_verify = requests.Session().verify
    try:
        data = hf_load("imdb", split="train", trust_remote_code=True,
                       download_mode="reuse_dataset_if_exists")
    except Exception:
        # Try with explicit download config
        data = hf_load("imdb", split="train", trust_remote_code=True)
    texts = data["text"]
    labels = np.array(data["label"], dtype=np.int64)
    return texts, labels, ["negative", "positive"], "real"


def _load_ag_news():
    """Load AG News topic dataset from HuggingFace."""
    from datasets import load_dataset as hf_load
    data = hf_load("ag_news", split="train", trust_remote_code=True)
    texts = data["text"]
    labels = np.array(data["label"], dtype=np.int64)
    return texts, labels, ["World", "Sports", "Business", "Sci/Tech"], "real"


def _load_yelp():
    """Load Yelp Polarity dataset from HuggingFace."""
    from datasets import load_dataset as hf_load
    data = hf_load("yelp_polarity", split="train", trust_remote_code=True)
    texts = data["text"]
    labels = np.array(data["label"], dtype=np.int64)
    return texts, labels, ["negative", "positive"], "real"


def _load_newsgroups():
    """Load 20 Newsgroups dataset via sklearn (requires internet for first download)."""
    from sklearn.datasets import fetch_20newsgroups
    # Use 4 main categories for simpler classification
    categories = [
        "comp.graphics", "comp.os.ms-windows.misc",
        "rec.autos", "rec.motorcycles",
        "sci.crypt", "sci.electronics",
        "talk.politics.guns", "talk.politics.mideast",
    ]
    data = fetch_20newsgroups(
        subset="all", categories=categories,
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    texts = data.data
    labels = np.array(data.target, dtype=np.int64)
    class_names = [data.target_names[i] for i in range(len(data.target_names))]
    return texts, labels, class_names, "real"



def _load_synthetic(n_samples: int = 2000):
    """Generate a STYLE-BASED synthetic text classification dataset.

    Returns:
        texts, labels, class_names, data_source_flag
        data_source_flag = "synthetic" to distinguish from real data.
    """
    rng = np.random.RandomState(42)

    TOPICS = {
        "ai": ["artificial intelligence applications", "deep learning architectures", "language model development", "AI ethics and bias"],
        "climate": ["carbon reduction strategies", "renewable energy adoption", "extreme weather trends", "sustainable agriculture"],
        "education": ["digital transformation of education", "standardized testing reform", "online learning platforms", "curriculum reform"],
        "healthcare": ["telemedicine adoption", "clinical trial design", "public health infrastructure", "health insurance reform"],
        "work": ["distributed team collaboration", "hybrid work environments", "office space utilization"],
        "commerce": ["consumer behavior analysis", "supply chain optimization", "digital payment systems"],
    }

    STYLES = {
        "formal_report": {
            "o": ["This report presents analysis of {t}. Findings indicate that", "A comprehensive review of {t} reveals key trends."],
            "b": ["quantitative evidence supports the conclusion", "stakeholder feedback has been incorporated", "metrics show consistent improvement", "comparative analysis reveals competitive positioning"],
            "c": ["Recommendations are outlined in the appendix.", "Quarterly monitoring should continue."],
        },
        "casual_blog": {
            "o": ["so I have been thinking about {t} lately and honestly,", "okay here is the thing about {t} that nobody talks about:"],
            "b": ["it is pretty wild when you stop and think about it", "my friend in this space told me something surprising", "the whole thing reminds me of when everything changed", "honestly I am not even sure what to think anymore"],
            "c": ["anyway just my thoughts what do you think?", "would love to hear other perspectives on this!"],
        },
        "technical_manual": {
            "o": ["Configuration of {t} requires the following:", "To implement {t} execute these procedures."],
            "b": ["ensure all dependencies are resolved first", "default config can be overridden by custom params", "refer to Section 4.2 for troubleshooting", "maximum throughput requires calibrated batch sizes"],
            "c": ["Run verification to confirm deployment.", "Consult changelog for deprecated features."],
        },
        "opinion_piece": {
            "o": ["It is unacceptable that {t} continues to be mishandled.", "We must confront the truth about {t}: the status quo is failing."],
            "b": ["how much longer can we tolerate this incompetence", "the evidence is overwhelming and response inadequate", "any reasonable person would conclude reform is needed", "defenders of the current approach have no arguments left"],
            "c": ["The time for half-measures has passed.", "History will judge us harshly for this failure."],
        },
        "news_brief": {
            "o": ["According to a new report Tuesday, {t} has", "Officials announced yesterday that {t} will undergo"],
            "b": ["the development was confirmed by sources", "industry analysts expressed cautious optimism", "the announcement comes amid growing pressure", "regulatory authorities will review the proposal"],
            "c": ["Further details expected in coming weeks.", "The organization declined to comment further."],
        },
        "academic_paper": {
            "o": ["Prior work on {t} focused on narrow assumptions. We extend by", "A gap remains in understanding {t}. This paper addresses"],
            "b": ["our framework builds on established foundations", "empirical analysis employs robust specifications", "we acknowledge limitations including endogeneity", "findings are robust to alternative specifications"],
            "c": ["These findings have implications for theory and practice.", "We encourage replication to verify these conclusions."],
        },
    }

    style_names = sorted(STYLES.keys())
    n_styles = len(style_names)
    texts, labels = [], []
    per_class = n_samples // n_styles

    for si, (sn, tmpl) in enumerate(STYLES.items()):
        for _ in range(per_class):
            tc = rng.choice(list(TOPICS.keys()))
            tp = rng.choice(TOPICS[tc])
            opener = rng.choice(tmpl["o"]).format(t=tp)
            nb = rng.choice([2, 3])
            body = list(rng.choice(tmpl["b"], size=nb, replace=False))
            closer = rng.choice(tmpl["c"])
            text = opener + " " + " ".join(body) + " " + closer
            if rng.random() < 0.15:
                rs = rng.choice([s for s in style_names if s != sn])
                text += " " + rng.choice(STYLES[rs]["b"])
            texts.append(text)
            labels.append(si)

    combined = list(zip(texts, labels))
    rng.shuffle(combined)
    texts, labels = zip(*combined)

    print(f"  [DataLoader] Style-based synthetic: {len(texts)} samples, {n_styles} classes")
    return list(texts), np.array(labels, dtype=np.int64), style_names, "synthetic"



def _stratified_subset(texts, labels, n_samples):
    """Create a stratified random subset."""
    unique_labels = np.unique(labels)
    subset_texts, subset_labels = [], []

    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        n_per_class = max(1, n_samples // len(unique_labels))
        chosen = np.random.RandomState(42).choice(idx, size=min(n_per_class, len(idx)), replace=False)
        subset_texts.extend([texts[i] for i in chosen])
        subset_labels.extend([labels[i] for i in chosen])

    # Shuffle
    combined = list(zip(subset_texts, subset_labels))
    np.random.RandomState(42).shuffle(combined)
    subset_texts, subset_labels = zip(*combined)
    return list(subset_texts), np.array(subset_labels)


# ============================================================
# Tabular Feature Extraction
# ============================================================

# Simple stopword list (top English stopwords)
STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
    "or", "an", "will", "my", "one", "all", "would", "there", "their",
    "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just",
    "him", "know", "take", "people", "into", "year", "your", "good",
    "some", "could", "them", "see", "other", "than", "then", "now",
    "look", "only", "come", "its", "over", "think", "also", "back",
    "after", "use", "two", "how", "our", "work", "first", "well",
    "way", "even", "new", "want", "because", "any", "these", "give",
    "day", "most", "us",
}


def extract_tabular_features(texts):
    """
    Extract hand-crafted tabular features from text.

    Returns:
        X_tabular: np.ndarray of shape (n_samples, n_features)
        feature_names: List[str]
    """
    features = []
    for text in texts:
        feats = _extract_single(text)
        features.append(feats)

    X = np.array(features, dtype=np.float32)

    # Handle NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, TABULAR_FEATURES


def _extract_single(text: str):
    """Extract features from a single text sample."""
    text = str(text) if text is not None else ""
    chars = len(text)
    words = text.split()
    word_count = len(words)

    # Word-level
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    unique_words = set(w.lower() for w in words)
    unique_ratio = len(unique_words) / max(word_count, 1)
    stopword_count = sum(1 for w in words if w.lower() in STOPWORDS)
    stopword_ratio = stopword_count / max(word_count, 1)

    # Sentence-level
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    avg_sent_len = word_count / max(sent_count, 1)

    # Character-level ratios
    punct_count = sum(1 for c in text if c in '.,!?;:\'"-()[]{}')
    punct_ratio = punct_count / max(chars, 1)
    capital_count = sum(1 for c in text if c.isupper())
    capital_ratio = capital_count / max(chars, 1)
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / max(chars, 1)

    # Special tokens
    exclamation_count = text.count('!')
    question_count = text.count('?')
    url_count = len(re.findall(r'https?://\S+', text))
    has_quotes = 1.0 if '"' in text or '"' in text or '\'' in text else 0.0

    return [
        chars,
        word_count,
        avg_word_len,
        sent_count,
        avg_sent_len,
        punct_ratio,
        capital_ratio,
        digit_ratio,
        unique_ratio,
        stopword_ratio,
        exclamation_count,
        question_count,
        url_count,
        has_quotes,
    ]


# ============================================================
# Data Splitting
# ============================================================

def prepare_data(
    dataset_name: str,
    n_samples: Optional[int] = None,
    seed: int = 42,
):
    """
    Full data pipeline: load, extract tabular features, split.

    Returns:
        data: Dict with keys:
            - X_train_tab, X_val_tab, X_test_tab
            - y_train, y_val, y_test
            - texts_train, texts_val, texts_test
            - feature_names: List[str]
            - num_classes: int
            - class_names: List[str]
            - tabular_scaler: StandardScaler (fit on train)
    """
    texts, labels, class_names, data_source = load_dataset(dataset_name, n_samples=n_samples)

    # Split: train / val / test
    texts_train, texts_temp, y_train, y_temp = train_test_split(
        texts, labels, test_size=TEST_SIZE + VAL_SIZE,
        random_state=seed, stratify=labels,
    )
    val_ratio = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    texts_val, texts_test, y_val, y_test = train_test_split(
        texts_temp, y_temp, test_size=1 - val_ratio,
        random_state=seed, stratify=y_temp,
    )

    # Tabular features
    X_train_tab, feature_names = extract_tabular_features(texts_train)
    X_val_tab, _ = extract_tabular_features(texts_val)
    X_test_tab, _ = extract_tabular_features(texts_test)

    # Scale tabular features
    scaler = StandardScaler()
    X_train_tab = scaler.fit_transform(X_train_tab)
    X_val_tab = scaler.transform(X_val_tab)
    X_test_tab = scaler.transform(X_test_tab)

    # NaN check after scaling
    X_train_tab = np.nan_to_num(X_train_tab, nan=0.0)
    X_val_tab = np.nan_to_num(X_val_tab, nan=0.0)
    X_test_tab = np.nan_to_num(X_test_tab, nan=0.0)

    return {
        "X_train_tab": X_train_tab,
        "X_val_tab": X_val_tab,
        "X_test_tab": X_test_tab,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "texts_train": texts_train,
        "texts_val": texts_val,
        "texts_test": texts_test,
        "feature_names": feature_names,
        "num_classes": DATASET_CONFIGS[dataset_name]["num_classes"],
        "class_names": class_names,
        "tabular_scaler": scaler,
        "data_source": data_source,  # "real" or "synthetic"
    }
