


import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "hinglish_reviews.csv"
MODEL_PATH = BASE_DIR / "sentiment_model.joblib"
ERROR_PATH = BASE_DIR / "error_analysis.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset missing. Run python create_dataset.py first."
    )

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["review", "sentiment"]).copy()
df["review"] = df["review"].astype(str)
df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()

labels = ["negative", "neutral", "positive"]

# Keep only supported sentiment classes
df = df[df["sentiment"].isin(labels)].reset_index(drop=True)

if df.empty or df["sentiment"].nunique() < 2:
    raise ValueError("Dataset must contain at least two sentiment classes.")

X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    output_dict=True,
    zero_division=0
)

print("\n===== HINGLISH SENTIMENT MODEL =====")
print(f"Total reviews: {len(df)}")
print(f"Training reviews: {len(X_train)}")
print(f"Testing reviews: {len(X_test)}")
print(f"Test accuracy: {accuracy:.2%}")

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
        digits=3
    )
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

matrix_df = pd.DataFrame(
    matrix,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels]
)

print("\n===== CONFUSION MATRIX =====")
print(matrix_df.to_string())

# Export incorrect predictions for error analysis
error_df = pd.DataFrame({
    "review": X_test.to_numpy(),
    "actual_sentiment": y_test.to_numpy(),
    "predicted_sentiment": predictions
})

error_df = error_df[
    error_df["actual_sentiment"] != error_df["predicted_sentiment"]
].copy()

error_df.to_csv(ERROR_PATH, index=False, encoding="utf-8-sig")

print("\n===== ERROR ANALYSIS =====")
print(f"Incorrect predictions: {len(error_df)} / {len(X_test)}")
print(f"Error report saved to: {ERROR_PATH}")

if not error_df.empty:
    print("\nMisclassified reviews:")
    print(error_df.to_string(index=False))
else:
    print("No incorrect predictions in this test split.")

# Save model and evaluation details for Streamlit dashboard
joblib.dump(
    {
        "model": model,
        "accuracy": float(accuracy),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "test_size": len(X_test),
        "train_size": len(X_train),
        "total_reviews": len(df)
    },
    MODEL_PATH
)

print(f"\nModel saved successfully: {MODEL_PATH}")

print("\n===== SAMPLE PREDICTIONS =====")

samples = [
    "Product ki quality bahut achhi hai",
    "Delivery bahut late hui aur service bekar hai",
    "Product theek-thaak hai"
]

for review in samples:
    sentiment = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    confidence = max(probabilities)

    print(f"\nReview: {review}")
    print(f"Predicted sentiment: {sentiment}")
    print(f"Model confidence: {confidence:.2%}")
