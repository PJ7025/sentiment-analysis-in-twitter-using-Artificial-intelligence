# Sentiment Analysis on Twitter Data — Assignment Solution

# 1) Imports
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 2) Insert dataset path (flexible input)
# For Sentiment140, use: training.1600000.processed.noemoticon.csv
dataset_path = input("Enter the path to your dataset CSV file: ").strip()

# 3) Load data (Sentiment140 has no headers; we add them)
df = pd.read_csv(
    dataset_path,
    encoding="latin-1",
    names=["target", "id", "date", "flag", "user", "text"]
)

# 4) Keep only target and text; remap labels (0 -> 0, 4 -> 1)
df = df[["target", "text"]].copy()
df["target"] = df["target"].map({0: 0, 4: 1})

# Optional: Drop rows with missing text or unmapped labels
df = df.dropna(subset=["text"])
df = df[df["target"].isin([0, 1])]

# 5) Preprocess text
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)      # URLs
    text = re.sub(r"@\w+", "", text)                  # mentions
    text = re.sub(r"#", " ", text)                    # keep hashtag word but remove symbol
    text = re.sub(r"[^a-z\s]", " ", text)             # non-letters
    text = re.sub(r"\s+", " ", text).strip()          # extra spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# 6) Train-test split
X = df["clean_text"]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7) Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)  # unigrams + bigrams often help
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 8) Train models
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1)
log_reg.fit(X_train_vec, y_train)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)

# 9) Evaluate models
print("\n=== Logistic Regression Results ===")
y_pred_lr = log_reg.predict(X_test_vec)
print(classification_report(y_test, y_pred_lr, digits=3))
print("Accuracy (LR):", round(accuracy_score(y_test, y_pred_lr), 4))

print("\n=== Naive Bayes Results ===")
y_pred_nb = nb.predict(X_test_vec)
print(classification_report(y_test, y_pred_nb, digits=3))
print("Accuracy (NB):", round(accuracy_score(y_test, y_pred_nb), 4))

# Confusion matrices
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_nb = confusion_matrix(y_test, y_pred_nb)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix - Logistic Regression")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

sns.heatmap(cm_nb, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title("Confusion Matrix - Naive Bayes")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
plt.tight_layout()
plt.show()

# 10) Class distribution visualization
plt.figure(figsize=(5, 4))
sns.countplot(x=y)
plt.title("Sentiment Distribution (0 = Negative, 1 = Positive)")
plt.xlabel("Sentiment"); plt.ylabel("Count")
plt.show()

# 11) Trend visualization (rolling mean over index)
df_sorted = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle to avoid grouped bias
df_sorted["rolling_sentiment"] = df_sorted["target"].rolling(1000).mean()
plt.figure(figsize=(10, 4))
df_sorted["rolling_sentiment"].plot()
plt.title("Rolling Average Sentiment (window = 1000 tweets)")
plt.ylabel("Average sentiment (0..1)"); plt.xlabel("Tweet index")
plt.ylim(0, 1)
plt.show()

# 12) Sample predictions preview
sample_texts = [
    "I love this phone! Battery life is amazing.",
    "Worst customer service ever, totally disappointed.",
    "Not bad, but could be better."
]
sample_clean = [clean_text(t) for t in sample_texts]
sample_vec = vectorizer.transform(sample_clean)

print("\n=== Sample Predictions (Logistic Regression) ===")
for t, p in zip(sample_texts, log_reg.predict(sample_vec)):
    print(f"Tweet: {t}  --> Predicted Sentiment: {p}")

print("\n=== Sample Predictions (Naive Bayes) ===")
for t, p in zip(sample_texts, nb.predict(sample_vec)):
    print(f"Tweet: {t}  --> Predicted Sentiment: {p}")
#C:\Users\tomch\Downloads\ai assignment\pj_data.csv