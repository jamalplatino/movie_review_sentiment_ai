import os
import tarfile
import urllib.request
import glob
from pathlib import Path
import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------- CONFIG ----------
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATASET_DIR = Path("aclImdb")
TAR_PATH = Path("aclImdb_v1.tar.gz")
MODEL_DIR = Path("sentiment_model")
MAX_FEATURES = 10_000
LOGISTIC_C = 0.5
RANDOM_STATE = 42
# ---------------------------

def download_and_extract() -> None:
    if not TAR_PATH.exists():
        print("Downloading IMDb dataset (84 MB)...")
        urllib.request.urlretrieve(DATASET_URL, TAR_PATH)
        print("Download complete.")
    else:
        print("Tar file already exists, skipping download.")

    if not DATASET_DIR.exists():
        print("Extracting...")
        with tarfile.open(TAR_PATH, "r:gz") as tar:
            tar.extractall()
        print("Extraction done.")
    else:
        print("Dataset already extracted.")

def load_imdb_reviews(split: str) -> list[tuple[str, int]]:
    data = []
    for sentiment, label in [("pos", 1), ("neg", 0)]:
        pattern = f"aclImdb/{split}/{sentiment}/*.txt"
        for file_path in glob.glob(pattern):
            text = Path(file_path).read_text(encoding="utf-8")
            data.append((text, label))
    return data

def main() -> None:
    download_and_extract()

    # Check if model already exists
    model_path = f'{MODEL_DIR}/model.pkl'
    vectorizer_path = f'{MODEL_DIR}/vectorizer.pkl'

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        print("✅ Saved model found! Skipping training and loading artifacts...")
        
        # Load the saved artifacts
        loaded_model = joblib.load(model_path)
        loaded_vec = joblib.load(vectorizer_path)
        
        # Run sanity check immediately
        sample_text = "This horror movie was fantastic!"
        X_sample = loaded_vec.transform([sample_text])
        prediction = loaded_model.predict(X_sample)[0]
        confidence = loaded_model.predict_proba(X_sample)[0].max()
        print(f"\nSanity check on: '{sample_text}'")
        print(f"Prediction: {'positive' if prediction == 1 else 'negative'} (confidence: {confidence:.3%})")
        return  # <-- Exit the function early, skip all the training code below



    # --- If we reach here, no saved model exists. Run the full training pipeline ---
    print("🆕 No saved model found. Training from scratch...")

    train_data = load_imdb_reviews("train")
    test_data = load_imdb_reviews("test")
    train_df = pd.DataFrame(train_data, columns=["review", "sentiment"])
    test_df = pd.DataFrame(test_data, columns=["review", "sentiment"])
    print(f"Train set: {len(train_df)} reviews, Test set: {len(test_df)} reviews")

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_df["review"])
    y_train = train_df["sentiment"]
    X_test = vectorizer.transform(test_df["review"])
    y_test = test_df["sentiment"]
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    model = LogisticRegression(
        max_iter=1000,
        C=LOGISTIC_C,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"Training accuracy: {train_acc:.2%}")
    print(f"Test accuracy:     {test_acc:.2%}")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_preds, target_names=["negative", "positive"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    print(f"Artifacts saved to '{MODEL_DIR}/'")

    # Sanity check
    loaded_model = joblib.load(MODEL_DIR / "model.pkl")
    loaded_vec = joblib.load(MODEL_DIR / "vectorizer.pkl")
    sample_text = "This horror movie was fantastic!"
    X_sample = loaded_vec.transform([sample_text])
    prediction = loaded_model.predict(X_sample)[0]
    confidence = loaded_model.predict_proba(X_sample)[0].max()
    print(f"\nSanity check on: '{sample_text}'")
    print(f"Prediction: {'positive' if prediction == 1 else 'negative'} (confidence: {confidence:.3%})")

if __name__ == "__main__":
    main()