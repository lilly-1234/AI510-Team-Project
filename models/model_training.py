"""
=============================================================
AI Customer Support Agent — E-Commerce
Part 2: Model Development & Evaluation
=============================================================
Task    : Multi-class Intent Classification (27 intents)
Input   : Customer message text  (instruction column)
Output  : Predicted intent label (intent column)
Pipeline: Text → TF-IDF Vectorizer → ML Classifier → Intent
=============================================================
"""

import os
import pandas as pd # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib # type: ignore
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# ── STEP 1: Load Data ──────────────────────────────────────
print("STEP 1: Loading data...")
df = pd.read_csv("data/cleaned_customer_support_dataset.csv")
print(f"  Rows: {df.shape[0]:,}  |  Intents: {df['intent'].nunique()}\n")

# ── STEP 2: Features & Labels ──────────────────────────────
X = df["instruction"]
y = df["intent"]

# Save a response lookup for the chatbot (intent → example response)
response_map = df.groupby("intent")["response"].first().to_dict()
joblib.dump(response_map, "models/response_map.pkl")
print("STEP 2: response_map saved (used by chatbot to reply)\n")

# ── STEP 3: Label Encoding ─────────────────────────────────
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "models/label_encoder.pkl")
print("STEP 3: Labels encoded (27 intent classes)\n")

# ── STEP 4: Train/Test Split ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"STEP 4: Split done — Train: {len(X_train):,}  Test: {len(X_test):,}\n")

# ── STEP 5: TF-IDF Vectorization ──────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=30_000, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print(f"STEP 5: TF-IDF done — vocab size: {len(vectorizer.vocabulary_):,}\n")

# ── STEP 6: Train & Compare Models ────────────────────────
print("STEP 6: Training models...")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=5, random_state=42),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"  Training {name} ...", end=" ", flush=True)
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"Accuracy: {acc*100:.2f}%")

# ── STEP 7: Select Best Model ──────────────────────────────
best_name  = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
joblib.dump(best_model, "models/best_model.pkl")
print(f"\nSTEP 7: Best model → {best_name} ({results[best_name]['accuracy']*100:.2f}%)")
print("  Saved: models/best_model.pkl\n")

# ── STEP 8: Classification Report ─────────────────────────
print("STEP 8: Classification Report")
report = classification_report(y_test, best_preds, target_names=label_encoder.classes_)
print(report)
with open("results/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_name}\n\n{report}")

# ── STEP 9: Charts ─────────────────────────────────────────
# Confusion matrix
cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(18, 15))
ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
    ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues"
)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=15)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()

# Model comparison bar chart
names      = list(results.keys())
accuracies = [results[n]["accuracy"] * 100 for n in names]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(names, accuracies, color=["#4C72B0", "#55A868", "#C44E52"], height=0.5)
ax.set_xlim(70, 100)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Model Comparison — Intent Classification")
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            f"{acc:.2f}%", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()
print("STEP 9: Charts saved → results/\n")

# ── STEP 10: Chatbot Inference Test ───────────────────────
def predict_and_respond(message):
    vec     = vectorizer.transform([message])
    encoded = best_model.predict(vec)[0]
    intent  = label_encoder.inverse_transform([encoded])[0]
    response = response_map.get(intent, "Sorry, I couldn't understand your request.")
    return intent, response

print("STEP 10: Chatbot Inference Test")
test_messages = [
    "I need to cancel my order immediately",
    "Where is my refund? It has been 10 days",
    "How do I change my email address?",
    "What payment methods do you accept?",
    "I want to track my shipment",
]
for msg in test_messages:
    intent, response = predict_and_respond(msg)
    print(f"  Customer : {msg}")
    print(f"  Intent   : {intent}")
    print(f"  Response : {response[:90]}...")
    print()

print("All done! Models and results are saved and ready for Flask deployment.")
