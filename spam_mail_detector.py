"""
QSkill Internship - AI & ML Domain
Task 2: Spam Mail Detector
------------------------------------
Objective: Build a classifier that distinguishes between spam and
non-spam (ham) messages using textual data.
Dataset: SMS Spam Collection (UCI) - 5,572 labeled real-world SMS messages.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report,
                              roc_curve, auc)

sns.set_style("whitegrid")
OUT = "/home/claude/qskill/outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load the dataset
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/qskill/sms.tsv", sep="\t", header=None,
                  names=["label", "message"])

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:\n", df["label"].value_counts())
print("\nMissing values:\n", df.isnull().sum())

# Class balance chart
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="label", hue="label", palette="Set2", legend=False, ax=ax)
ax.set_title("Class Distribution: Spam vs Ham")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
fig.tight_layout()
fig.savefig(f"{OUT}/spam_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------
STOPWORDS = set("""a about above after again against all am an and any are aren't
as at be because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for from
further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is
isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs them
themselves then there there's these they they'd they'll they're they've this
those through to too under until up very was wasn't we we'd we'll we're we've
were weren't what what's when when's where where's which while who who's whom
why why's with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves""".split())


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"\d+", " ", text)                      # numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


df["clean_message"] = df["message"].apply(clean_text)
df["message_length"] = df["message"].apply(len)

print("\nSample before/after cleaning:")
for i in [0, 2]:
    print(f"  RAW  : {df['message'].iloc[i]}")
    print(f"  CLEAN: {df['clean_message'].iloc[i]}\n")

# Message length by class
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.histplot(data=df, x="message_length", hue="label", bins=40, kde=True,
             palette="Set2", ax=ax)
ax.set_title("Message Length Distribution by Class")
ax.set_xlabel("Message length (characters)")
fig.tight_layout()
fig.savefig(f"{OUT}/spam_message_length.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 3. Feature extraction (TF-IDF) & train/test split
# ---------------------------------------------------------------
X = df["clean_message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------------
# 4. Train models & compare
# ---------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
}

results = []
best_name, best_f1, best_model, best_preds, best_proba = None, -1, None, None, None

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    proba = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-Score": f1})

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    if f1 > best_f1:
        best_f1, best_name, best_model = f1, name, model
        best_preds, best_proba = preds, proba

results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
results_df.to_csv(f"{OUT}/spam_model_comparison.csv", index=False)
print("\nModel comparison summary:\n", results_df.to_string(index=False))
print(f"\nBest model: {best_name} (F1-Score = {best_f1:.4f})")

# ---------------------------------------------------------------
# 5. Confusion matrix
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["ham", "spam"], yticklabels=["ham", "spam"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_name}")
fig.tight_layout()
fig.savefig(f"{OUT}/spam_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 6. ROC Curve
# ---------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, best_proba)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve - {best_name}")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/spam_roc_curve.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 7. Model comparison chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax)
ax.set_ylim(0.8, 1.02)
ax.set_title("Model Performance Comparison")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/spam_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 8. Try it on new/unseen example messages
# ---------------------------------------------------------------
sample_msgs = [
    "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow at 1pm?",
    "URGENT: Your account has been suspended. Verify your details immediately to avoid closure.",
    "Don't forget to bring the documents for tomorrow's meeting.",
]
sample_clean = [clean_text(m) for m in sample_msgs]
sample_tfidf = vectorizer.transform(sample_clean)
sample_preds = best_model.predict(sample_tfidf)

print("\n" + "=" * 60)
print("PREDICTIONS ON NEW MESSAGES")
print("=" * 60)
for msg, pred in zip(sample_msgs, sample_preds):
    label = "SPAM" if pred == 1 else "HAM"
    print(f"[{label}] {msg}")

print("\nAll charts and results saved to:", OUT)
