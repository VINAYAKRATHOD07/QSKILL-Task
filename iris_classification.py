"""
QSkill Internship - AI & ML Domain
Task 1: Iris Flower Classification
------------------------------------
Objective: Classify iris flowers into three species (Setosa, Versicolor,
Virginica) based on measurements of their petals and sepals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

sns.set_style("whitegrid")
OUT = "/home/claude/qskill/outputs"
import os
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. Load & explore the dataset
# ---------------------------------------------------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:\n", df["species"].value_counts())
print("\nSummary statistics:\n", df.describe())

# ---------------------------------------------------------------
# 2. Visual exploration
# ---------------------------------------------------------------
# Pairplot
pair = sns.pairplot(df, hue="species", diag_kind="hist", palette="Set2")
pair.fig.suptitle("Iris Dataset - Pairwise Feature Relationships", y=1.02)
pair.savefig(f"{OUT}/iris_pairplot.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Boxplots per feature
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.flat, iris.feature_names):
    sns.boxplot(data=df, x="species", y=col, hue="species", palette="Set2",
                ax=ax, legend=False)
    ax.set_title(col)
fig.suptitle("Feature Distributions by Species", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/iris_boxplots.png", dpi=150, bbox_inches="tight")
plt.close("all")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap")
fig.tight_layout()
fig.savefig(f"{OUT}/iris_correlation.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 3. Train/test split & preprocessing
# ---------------------------------------------------------------
X = df[iris.feature_names]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train multiple classifiers & compare
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
}

results = []
best_model_name, best_acc, best_model, best_preds = None, -1, None, None

print("\n" + "=" * 60)
print("MODEL TRAINING & EVALUATION")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-Score": f1})

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(classification_report(y_test, preds, target_names=iris.target_names))

    if acc > best_acc:
        best_acc, best_model_name, best_model, best_preds = acc, name, model, preds

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(f"{OUT}/iris_model_comparison.csv", index=False)
print("\nModel comparison summary:\n", results_df.to_string(index=False))
print(f"\nBest model: {best_model_name} (Accuracy = {best_acc:.4f})")

# ---------------------------------------------------------------
# 5. Confusion matrix for the best model
# ---------------------------------------------------------------
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix - {best_model_name}")
fig.tight_layout()
fig.savefig(f"{OUT}/iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")

# ---------------------------------------------------------------
# 6. Model comparison bar chart
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax)
ax.set_ylim(0.8, 1.02)
ax.set_title("Model Performance Comparison")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(f"{OUT}/iris_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close("all")

print("\nAll charts and results saved to:", OUT)
