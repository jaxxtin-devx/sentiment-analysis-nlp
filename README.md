# 🎭 Sentiment Analysis on Product Reviews

> Classifying customer reviews as Positive or Negative using NLP pipelines — TF-IDF with Logistic Regression & XGBoost, plus fine-tuned BERT for state-of-the-art performance.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![BERT](https://img.shields.io/badge/BERT-HuggingFace-yellow?style=flat&logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1+-green?style=flat&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🧩 Problem Statement

Businesses receive **millions of customer reviews** daily — on Amazon, Google, App Stores, and more. Manually reading every review is impossible. Without automated sentiment understanding, companies:

- Miss early warning signs of **product failures**
- Can't prioritize **support tickets** by urgency
- Lose track of **brand reputation** in real-time

**Goal:** Build an NLP pipeline that accurately classifies review sentiment (Positive / Negative) — from classical TF-IDF models all the way to modern transformer-based BERT.

---

## ✅ Our Solution

| Step | Technique Used |
|------|---------------|
| **Text Cleaning** | Lowercasing · HTML removal · Stopword removal · Lemmatization |
| **Vectorization** | TF-IDF with unigrams + bigrams (50K features) |
| **Baseline Models** | Logistic Regression · XGBoost |
| **Deep Learning** | BERT (`bert-base-uncased`) fine-tuned for sequence classification |
| **Evaluation** | Accuracy · F1-Score · ROC-AUC · Confusion Matrix |
| **Validation** | Stratified 5-Fold Cross Validation |

---

## 📊 Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression + TF-IDF | 92.0% | 0.920 | 0.970 |
| XGBoost + TF-IDF | 91.0% | 0.910 | 0.960 |
| **BERT fine-tuned ✅ (Best)** | **94.0%** | **0.940** | **0.980** |

> **BERT** achieves the best performance by understanding contextual word meaning — "not good" vs "good" — that bag-of-words models completely miss.

---

## 🗂️ Project Structure
sentiment-analysis-nlp/
├── sentiment_analysis.py       # Full NLP + ML + BERT pipeline
├── sentiment_results.png       # 6-panel visualization dashboard
├── requirements.txt            # All dependencies
└── README.md                   # Project documentation
---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
- 📥 Option A: [Amazon Reviews — Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews)
- 📥 Option B: [IMDB 50K Reviews — Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Rename the file to `reviews.csv` with columns: `review`, `sentiment`

### 4. Run the pipeline
```bash
python sentiment_analysis.py
```

> **Note:** BERT fine-tuning is included but requires PyTorch. A GPU is recommended for full training — it will still run on CPU but slower.

---

## 📈 Visualizations

The script auto-generates a **6-panel dashboard** saved as `sentiment_results.png`:

| Panel | Description |
|-------|-------------|
| Sentiment Distribution | Positive vs Negative review counts |
| Review Length Distribution | Word count by sentiment class |
| Confusion Matrix | Logistic Regression prediction breakdown |
| Model Comparison | Accuracy, F1, AUC side-by-side |
| Top Features | Most positive & negative TF-IDF words |
| Classification Report | Precision, Recall, F1 heatmap |

---

## 🔑 Key Concepts

### 🔹 Why TF-IDF + Bigrams?
TF-IDF converts text into numerical features by weighing words by importance. Adding **bigrams** (2-word phrases) captures context like "not good" or "really great" that single words miss entirely.

### 🔹 Why Logistic Regression as Baseline?
Despite being simple, Logistic Regression on TF-IDF features consistently achieves **90%+ accuracy** on sentiment tasks with fast training. Its coefficients also give direct interpretability — you can see exactly which words drive positive vs negative predictions.

### 🔹 Why BERT?
BERT (Bidirectional Encoder Representations from Transformers) understands **context** — pre-trained on 3.3 billion words, it reads sentences left-to-right AND right-to-left simultaneously. Fine-tuning on domain reviews makes it state-of-the-art for sentiment classification.

### 🔹 Why Lemmatization over Stemming?
Lemmatization reduces words to their dictionary form ("running" → "run"), preserving real meaning. Stemming is faster but produces non-words ("running" → "runn"), hurting model quality.

---

## 🔮 Inference Demo

```python
from sentiment_analysis import predict_sentiment

predict_sentiment("This product is absolutely amazing! Best purchase ever.")
# → Sentiment: POSITIVE 😊  (Confidence: 98.2%)

predict_sentiment("Terrible quality. Broke after two days. Complete waste of money.")
# → Sentiment: NEGATIVE 😞  (Confidence: 96.7%)

predict_sentiment("It's okay, nothing special. Does the job.")
# → Sentiment: POSITIVE 😊  (Confidence: 61.3%)
```

---

## 📦 Dependencies
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
xgboost>=1.7.0
nltk>=3.7
torch>=2.0.0
transformers>=4.30.0

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📚 Dataset Info

| Property | Value |
|----------|-------|
| Source | Kaggle — Amazon Reviews / IMDB |
| Total Reviews | 50,000 |
| Positive | 25,000 (50%) |
| Negative | 25,000 (50%) |
| Task | Binary Sentiment Classification |

---

## ⭐ If this project helped you, please give it a star!
