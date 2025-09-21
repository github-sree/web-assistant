# Used for encoding/decoding serialized model bytes to store in MongoDB
import base64
# For serializing Python objects (models) more robustly than pickle
import cloudpickle
# Python logging for structured logs
import logging
# To access environment variables like MONGO_URI
import os
# For numerical operations like entropy, probabilities
import numpy as np
# For handling training data from MongoDB
import pandas as pd
# Linear classifier (efficient, supports partial_fit)
from sklearn.linear_model import SGDClassifier
# For probability calibration (turns margin scores into probabilities)
from sklearn.calibration import CalibratedClassifierCV
# Splits data + performs hyperparameter tuning
from sklearn.model_selection import train_test_split, GridSearchCV
# Metrics for evaluation
from sklearn.metrics import classification_report, confusion_matrix
# Embedding model for text → vectors
from sentence_transformers import SentenceTransformer
# MongoDB client for storage of training data, models, feedback
from pymongo import MongoClient
# Cache embeddings to avoid recomputing for the same query
from functools import lru_cache
# For versioning and timestamps
from datetime import datetime

# -------------------------
# Config & setup
# -------------------------
# The pretrained SentenceTransformer model name. Good default — strong embeddings.
DEFAULT_EMBEDDER = "all-mpnet-base-v2"
# Configure logging to print INFO+ messages to the console.
logging.basicConfig(level=logging.INFO)
# Creates a named logger used throughout the file.
logger = logging.getLogger("intent-classifier")
# MongoDB setup with fallback defaults
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
# Database name environment variable (default sre_assistant).
MONGO_DB = os.getenv("MONGO_DB", "sre_assistant")
# Create a Mongo client connected to the configured URI.
client = MongoClient(MONGO_URI)
# Reference to the Mongo database.
db = client[MONGO_DB]
intents_collection = db["intent_train_data"]  # Stores labeled queries (training data)
models_collection = db["intent_models"]  # Stores serialized trained models + metrics
# A dedicated collection for queued feedback (new labels that will be applied in batch).
feedback_collection = db["intent_feedback"]  # Stores user feedback for retraining later


# -------------------------
# Helpers
# -------------------------
def preprocess_text(text: str) -> str:
    """Clean and normalize text before embedding.
Simple normalizer: strip whitespace and lowercase. The aim is to reduce surface variation (e.g., Login vs login).
- Why: Embeddings are robust, but consistent input reduces noise and ensures identical strings map to same cache keys.
- Note: You can extend this to remove punctuation, expand abbreviations, fix common misspellings, or run domain-specific normalization.
"""
    text = text.strip().lower()  # Remove whitespace and lowercase text
    return text


def compute_entropy(probs: np.ndarray) -> float:
    """Compute entropy of probability distribution: higher entropy = more uncertain.
- Compute entropy of the probability distribution returned by the classifier. Useful as an uncertainty measure: high entropy → model is unsure (probabilities spread across classes).
"""
    return -np.sum(probs * np.log(probs + 1e-9))  # Add 1e-9 to avoid log(0)


# -------------------------
# Intent Classifier
# -------------------------
class IntentPrediction:
    def __init__(self):
        """
- self.embedder = SentenceTransformer(DEFAULT_EMBEDDER) → loads the embedding model into memory. Important: first load will download model weights (or read from local cache). It can be heavy (CPU or GPU).
- self.classifier = None → placeholder for trained/scikit-learn classifier.
- self.label_names = [] → list of known intent labels.
- self.version = None → will be set when model is trained and saved.
        """
        # Load sentence transformer embedder
        self.embedder = SentenceTransformer(DEFAULT_EMBEDDER)
        self.classifier = None  # Will hold trained classifier
        self.label_names = []  # Store list of intent labels
        self.version = None  # Track version of the trained model

    # Cache embeddings so repeated queries don’t re-embed, limits the cache memory use
    @lru_cache(maxsize=5000)
    def embed(self, text: str):
        """
        It ensures cached key consistency and the text passed to encoder is normalized.
- Calls self.embedder.encode([preprocess_text(text)], normalize_embeddings=True)[0].
- preprocess_text(text) — ensures cached key consistency and the text passed to encoder is normalized.
- encode([...]) returns an array; [0] extracts the vector for the single sentence.
- normalize_embeddings=True ensures vectors are L2-normalized (useful for cosine similarity or some classifiers).
        :param text:
        :return:
        """
        return self.embedder.encode([preprocess_text(text)], normalize_embeddings=True)[0]

    def load_data_from_mongo(self):
        """Fetch labeled training data from MongoDB."""
        docs = list(intents_collection.find({}, {"_id": 0, "query": 1, "label": 1}))
        if not docs:
            logger.warning("No training data found in MongoDB 'intents' collection.")
            return pd.DataFrame(columns=["query", "label"])
        df = pd.DataFrame(docs)
        df["query"] = df["query"].astype(str).map(preprocess_text)  # Normalize queries
        df["label"] = df["label"].astype(str)  # Ensure labels are strings
        return df

    def train(self, balance: bool = True):
        """Train classifier on intents from MongoDB."""
        df = self.load_data_from_mongo()
        if df.empty:
            raise ValueError("No training data in MongoDB to train the model.")

        # Balance=True tells the function to balance classes by under sampling majority classes to the size of the smallest class.
        if balance:
            class_counts = df["label"].value_counts()
            min_count = class_counts.min()
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(min_count, replace=True), include_groups=False)
                .reset_index(drop=True)
            )
            logger.info(f"Balanced training set with {min_count} samples per class")

        queries = df["query"].tolist()
        labels = df["label"].tolist()
        self.label_names = sorted(set(labels))

        logger.info(f"Training on {len(queries)} samples across {len(self.label_names)} intents")

        # Convert all queries into embeddings
        embeddings = self.embedder.encode(
            queries, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )

        # Split dataset into train/test
        #Reserve 20% for evaluation. random_state=42 ensures reproducible splits.
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, labels, test_size=0.2, random_state=42
        )

        # Hyperparameter tuning with grid search
        # search over learning rate regularization (alpha) and number of iterations.
        param_grid = {"alpha": [1e-4, 1e-3, 1e-2], "max_iter": [500, 1000]}

        grid = GridSearchCV(
            SGDClassifier(loss="log_loss", class_weight="balanced", n_jobs=-1),
            param_grid,
            cv=3,
            scoring="f1_macro",  # Evaluate across all classes equally
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        base_clf = grid.best_estimator_  # Best classifier after tuning
        logger.info(f"Best params: {grid.best_params_}")

        # Calibrate classifier for reliable probabilities
        self.classifier = CalibratedClassifierCV(base_clf, cv="prefit")
        self.classifier.fit(X_train, y_train)

        # Evaluate on test set
        preds = self.classifier.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        # Save trained model with version timestamp
        self.version = int(datetime.utcnow().timestamp())
        self._save_model_to_mongo(report, cm)

        logger.info("Training complete and saved to MongoDB.")
        return report

    def _save_model_to_mongo(self, report=None, cm=None):
        """Serialize model + metadata and save into MongoDB."""
        serialized_model = base64.b64encode(
            cloudpickle.dumps((self.classifier, self.label_names))
        ).decode("utf-8")
        doc = {
            "name": "intent_classifier",
            "version": self.version,
            "model": serialized_model,
            "metrics": report,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "timestamp": datetime.utcnow(),
        }
        models_collection.update_one({"name": "intent_classifier"}, {"$set": doc}, upsert=True)
        logger.info("Model saved in MongoDB 'models' collection.")

    def load(self):
        """Load latest saved model from MongoDB."""
        doc = models_collection.find_one({"name": "intent_classifier"})
        if not doc:
            logger.warning("No saved model found in MongoDB.")
            return False

        serialized_model = base64.b64decode(doc["model"])
        self.classifier, self.label_names = cloudpickle.loads(serialized_model)
        self.version = doc.get("version")
        logger.info(f"Model v{self.version} loaded from MongoDB.")
        return True

    def predict(self, query: str, prob_threshold: float = 0.3, entropy_threshold: float = 1.5):
        """Predict intent with thresholds for confidence + entropy."""
        if not self.classifier:
            raise ValueError("Model not trained or loaded.")

        emb = self.embed(query).reshape(1, -1)
        probs = self.classifier.predict_proba(emb)[0]

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        entropy = compute_entropy(probs)

        # If prediction is too uncertain, return 'unknown'
        if confidence < prob_threshold or entropy > entropy_threshold:
            return {"intent": "unknown", "confidence": confidence, "entropy": entropy}

        return {
            "intent": self.classifier.classes_[best_idx],
            "confidence": confidence,
            "entropy": entropy,
            "probs": dict(zip(self.classifier.classes_, map(float, probs))),
        }

    def queue_feedback(self, query: str, correct_label: str, notes: str = ""):
        """Queue feedback instead of immediate retrain."""
        feedback_collection.insert_one({
            "query": preprocess_text(query),
            "correct_label": correct_label,
            "notes": notes,
            "timestamp": datetime.utcnow(),
        })
        logger.info(f"Feedback queued: '{query}' → {correct_label}")
        return {"queued": True, "label": correct_label}

    def apply_feedback(self):
        """Apply queued feedback by updating training data and retraining."""
        feedbacks = list(feedback_collection.find())
        if not feedbacks:
            logger.info("No feedback to apply.")
            return

        # Insert feedback into main training dataset
        for fb in feedbacks:
            intents_collection.update_one(
                {"query": fb["query"]}, {"$set": {"label": fb["correct_label"]}}, upsert=True
            )

        # Clear feedback queue after applying
        feedback_collection.delete_many({})
        logger.info("Feedback applied, retraining...")
        return self.train()


if __name__ == "__main__":
    clf = IntentPrediction()
    if not clf.load():  # Load existing model if available, else train a new one
        clf.train()
    print(clf.predict("Application crashed after login"))  # Example prediction
