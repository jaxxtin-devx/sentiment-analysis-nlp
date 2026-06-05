"""
=============================================================
  Sentiment Analysis on Amazon Product Reviews
  Author: [Your Name]
  Dataset: Amazon Product Reviews (Kaggle) — 50,000 reviews
  Models: TF-IDF + Logistic Regression | TF-IDF + XGBoost | BERT (Fine-tuned)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  STEP 1: Loading & Exploring Data")
print("=" * 60)

# Dataset: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews
# OR use the IMDB dataset as alternative:
# https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
# Expected columns: 'review' (text), 'sentiment' (positive/negative)

df = pd.read_csv("reviews.csv")   # Replace with your file name

# Standardize column names (adjust if needed)
# df.columns = ['sentiment', 'review']  # Uncomment if needed

print(f"\nDataset Shape     : {df.shape}")
print(f"Columns           : {df.columns.tolist()}")
print(f"\nSentiment Distribution:\n{df['sentiment'].value_counts()}")
print(f"\nSample Review:\n{df['review'].iloc[0][:200]}...")
print(f"\nMissing Values    : {df.isnull().sum().sum()}")

# Drop nulls
df.dropna(subset=['review', 'sentiment'], inplace=True)

# Encode labels: positive=1, negative=0
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ─────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 2: Text Preprocessing")
print("=" * 60)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Full NLP preprocessing pipeline."""
    text = text.lower()                                          # Lowercase
    text = re.sub(r'<.*?>', '', text)                           # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)                  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)                        # Remove punctuation/numbers
    tokens = text.split()                                        # Tokenize
    tokens = [t for t in tokens if t not in stop_words]         # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]          # Lemmatize
    tokens = [t for t in tokens if len(t) > 2]                  # Remove short tokens
    return ' '.join(tokens)

print("Cleaning text... (this may take a minute)")
df['clean_review'] = df['review'].apply(clean_text)
print(f"Sample cleaned text:\n  {df['clean_review'].iloc[0][:150]}")

# Review length analysis
df['review_length'] = df['review'].apply(lambda x: len(x.split()))
df['clean_length']  = df['clean_review'].apply(lambda x: len(x.split()))
print(f"\nAvg review length (raw)   : {df['review_length'].mean():.0f} words")
print(f"Avg review length (cleaned): {df['clean_length'].mean():.0f} words")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X = df['clean_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train):,}  |  Test size: {len(X_test):,}")

# ─────────────────────────────────────────────
# 4. TF-IDF VECTORIZATION + MODELS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 3: TF-IDF Vectorization + Model Training")
print("=" * 60)

# TF-IDF with bigrams
tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),         # unigrams + bigrams
    min_df=3,
    max_df=0.9,
    sublinear_tf=True           # log normalization
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print(f"TF-IDF vocab size : {len(tfidf.vocabulary_):,} features")
print(f"Matrix shape      : {X_train_tfidf.shape}")

# Model 1: Logistic Regression
print("\n→ Training Logistic Regression...")
lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
lr_model.fit(X_train_tfidf, y_train)
lr_pred  = lr_model.predict(X_test_tfidf)
lr_proba = lr_model.predict_proba(X_test_tfidf)[:, 1]
lr_acc   = accuracy_score(y_test, lr_pred)
lr_f1    = f1_score(y_test, lr_pred)
lr_auc   = roc_auc_score(y_test, lr_proba)
print(f"   Accuracy: {lr_acc:.4f} | F1: {lr_f1:.4f} | ROC-AUC: {lr_auc:.4f}")

# Model 2: XGBoost
print("\n→ Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_tfidf, y_train)
xgb_pred  = xgb_model.predict(X_test_tfidf)
xgb_proba = xgb_model.predict_proba(X_test_tfidf)[:, 1]
xgb_acc   = accuracy_score(y_test, xgb_pred)
xgb_f1    = f1_score(y_test, xgb_pred)
xgb_auc   = roc_auc_score(y_test, xgb_proba)
print(f"   Accuracy: {xgb_acc:.4f} | F1: {xgb_f1:.4f} | ROC-AUC: {xgb_auc:.4f}")

# ─────────────────────────────────────────────
# 5. BERT FINE-TUNING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 4: BERT Fine-Tuning (Transformers)")
print("=" * 60)

try:
    import torch
    from transformers import (
        BertTokenizer, BertForSequenceClassification,
        Trainer, TrainingArguments
    )
    from torch.utils.data import Dataset

    BERT_AVAILABLE = True
    print("PyTorch & Transformers detected ✓")

    class ReviewDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts     = texts.tolist()
            self.labels    = labels.tolist()
            self.tokenizer = tokenizer
            self.max_len   = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = self.tokenizer(
                self.texts[idx],
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids':      enc['input_ids'].squeeze(),
                'attention_mask': enc['attention_mask'].squeeze(),
                'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
            }

    # Use a small subset for fine-tuning (full dataset needs GPU)
    BERT_TRAIN_SIZE = 5000
    BERT_TEST_SIZE  = 1000

    X_bert_train = X_train.reset_index(drop=True)[:BERT_TRAIN_SIZE]
    y_bert_train = y_train.reset_index(drop=True)[:BERT_TRAIN_SIZE]
    X_bert_test  = X_test.reset_index(drop=True)[:BERT_TEST_SIZE]
    y_bert_test  = y_test.reset_index(drop=True)[:BERT_TEST_SIZE]

    print(f"\nFine-tuning on {BERT_TRAIN_SIZE} samples (use GPU for full training)")

    tokenizer  = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )

    train_dataset = ReviewDataset(X_bert_train, y_bert_train, tokenizer)
    test_dataset  = ReviewDataset(X_bert_test,  y_bert_test,  tokenizer)

    training_args = TrainingArguments(
        output_dir='./bert_output',
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy='epoch',
        save_strategy='no',
        logging_dir='./logs',
        logging_steps=50,
        load_best_model_at_end=False,
        report_to='none'
    )

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("Training BERT... (2 epochs)")
    trainer.train()

    # Evaluate BERT
    bert_preds_raw = trainer.predict(test_dataset)
    bert_logits    = bert_preds_raw.predictions
    bert_pred      = np.argmax(bert_logits, axis=1)
    bert_proba     = torch.softmax(torch.tensor(bert_logits), dim=1).numpy()[:, 1]

    bert_acc = accuracy_score(y_bert_test, bert_pred)
    bert_f1  = f1_score(y_bert_test, bert_pred)
    bert_auc = roc_auc_score(y_bert_test, bert_proba)

    print(f"\nBERT Results:")
    print(f"   Accuracy: {bert_acc:.4f} | F1: {bert_f1:.4f} | ROC-AUC: {bert_auc:.4f}")

except ImportError:
    BERT_AVAILABLE = False
    print("PyTorch/Transformers not installed.")
    print("Install with: pip install torch transformers")
    print("Continuing with TF-IDF models only...")

# ─────────────────────────────────────────────
# 6. CROSS-VALIDATION (Best TF-IDF Model)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 5: Stratified 5-Fold Cross-Validation")
print("=" * 60)

lr_cv = LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1)
pipeline_cv = Pipeline([('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)),
                         ('clf',   lr_cv)])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline_cv, X_train, y_train, cv=skf, scoring='accuracy')
print(f"CV Accuracy Scores : {cv_scores}")
print(f"Mean ± Std         : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 7. INFERENCE FUNCTION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 6: Inference Demo")
print("=" * 60)

def predict_sentiment(text, model=lr_model, vectorizer=tfidf):
    """Predict sentiment for any review text."""
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    proba   = model.predict_proba(vec)[0]
    label   = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
    conf    = max(proba) * 100
    return f"Sentiment: {label}  (Confidence: {conf:.1f}%)"

sample_reviews = [
    "This product is absolutely amazing! Best purchase I've ever made.",
    "Terrible quality. Broke after two days. Complete waste of money.",
    "It's okay, nothing special. Does the job but could be better.",
]

print("\nSample Predictions:")
for rev in sample_reviews:
    print(f"\n  Review : {rev[:60]}...")
    print(f"  {predict_sentiment(rev)}")

# ─────────────────────────────────────────────
# 8. VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  STEP 7: Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Sentiment Analysis — Results Dashboard", fontsize=16, fontweight='bold')

# (A) Sentiment Distribution
ax = axes[0, 0]
counts = df['label'].value_counts()
colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(['Negative', 'Positive'], [counts[0], counts[1]], color=colors, edgecolor='black', width=0.5)
ax.set_title('Sentiment Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, count in zip(bars, [counts[0], counts[1]]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count:,}', ha='center', va='bottom', fontweight='bold')

# (B) Review Length Distribution
ax = axes[0, 1]
ax.hist(df[df['label']==1]['review_length'], bins=50, alpha=0.6, color='#2ecc71', label='Positive')
ax.hist(df[df['label']==0]['review_length'], bins=50, alpha=0.6, color='#e74c3c', label='Negative')
ax.set_title('Review Length Distribution', fontweight='bold')
ax.set_xlabel('Word Count')
ax.set_ylabel('Frequency')
ax.legend()
ax.set_xlim(0, 600)

# (C) Confusion Matrix — Logistic Regression
ax = axes[0, 2]
cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
ax.set_title('Confusion Matrix — Logistic Regression', fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')

# (D) Model Comparison
ax = axes[1, 0]
model_names = ['Logistic Regression', 'XGBoost']
accuracies  = [lr_acc, xgb_acc]
f1_scores   = [lr_f1, xgb_f1]
auc_scores  = [lr_auc, xgb_auc]

if BERT_AVAILABLE:
    model_names.append('BERT')
    accuracies.append(bert_acc)
    f1_scores.append(bert_f1)
    auc_scores.append(bert_auc)

x = np.arange(len(model_names))
w = 0.25
ax.bar(x - w, accuracies, w, label='Accuracy', color='#3498db')
ax.bar(x,     f1_scores,  w, label='F1-Score', color='#9b59b6')
ax.bar(x + w, auc_scores, w, label='ROC-AUC',  color='#1abc9c')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=10, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title('Model Performance Comparison', fontweight='bold')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# (E) Top Positive & Negative TF-IDF Features
ax = axes[1, 1]
feature_names = np.array(tfidf.get_feature_names_out())
coef = lr_model.coef_[0]
top_pos_idx = coef.argsort()[-15:]
top_neg_idx = coef.argsort()[:15]

top_words   = np.concatenate([feature_names[top_neg_idx], feature_names[top_pos_idx]])
top_coefs   = np.concatenate([coef[top_neg_idx], coef[top_pos_idx]])
colors_coef = ['#e74c3c' if c < 0 else '#2ecc71' for c in top_coefs]

ax.barh(range(len(top_words)), top_coefs, color=colors_coef)
ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words, fontsize=8)
ax.set_title('Top Positive & Negative Words (LR)', fontweight='bold')
ax.set_xlabel('Coefficient Value')
ax.axvline(0, color='black', linewidth=0.8)

# (F) Classification Report Heatmap
ax = axes[1, 2]
report = classification_report(y_test, lr_pred,
                                target_names=['Negative', 'Positive'],
                                output_dict=True)
report_df = pd.DataFrame(report).T.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
report_df  = report_df[['precision', 'recall', 'f1-score']].astype(float)
sns.heatmap(report_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, vmin=0.8, vmax=1.0)
ax.set_title('Classification Report Heatmap', fontweight='bold')

plt.tight_layout()
plt.savefig('sentiment_results.png', dpi=150, bbox_inches='tight')
print("Saved → sentiment_results.png")
plt.show()

print("\n" + "=" * 60)
print("  DONE! Best Model: Logistic Regression + TF-IDF")
print(f"  Accuracy : {lr_acc:.4f}")
print(f"  F1-Score : {lr_f1:.4f}")
print(f"  ROC-AUC  : {lr_auc:.4f}")
print("=" * 60)
