# timeframe_model.py
# ML-based timeframe predictor with MongoDB model storage
# Requirements:
# pip install sentence-transformers scikit-learn cloudpickle pymongo dateparser python-dotenv

import base64
import cloudpickle
import logging
import os
import re
from functools import lru_cache
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import dateparser
from dateparser.search import search_dates

# -------------------------
# Config & setup
# -------------------------
DEFAULT_EMBEDDER = os.getenv("DEFAULT_EMBEDDER", "all-mpnet-base-v2")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("timeframe-classifier")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "sre_assistant")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

training_collection = db["timeframe_training_data"]   # { query: str, label: str, meta: {} }
models_collection = db["timeframe_models"]       # stores serialized model
feedback_collection = db["timeframe_feedback"]   # queued feedback for retraining

# Some label choices. Extend as needed.
DEFAULT_LABELS = [
    "last_n_minutes",
    "last_n_hours",
    "last_n_days",
    "today",
    "yesterday",
    "this_week",
    "last_week",
    "absolute_range",
    "since_time",
    "default"  # fallback -> last 1 hour
]


# -------------------------
# Helpers
# -------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    return text.strip().lower()


def compute_entropy(probs: np.ndarray) -> float:
    return -np.sum(probs * np.log(probs + 1e-9))


def _extract_number_unit(query: str):
    """
    Looks for patterns like 'last 2 hours', 'past 3 days', 'last 30 minutes'
    Returns (n:int, unit:str) or (None, None)
    """
    m = re.search(r"(?:last|past|for|in the last)\s+(\d+)\s*(minute|minutes|hour|hours|day|days|week|weeks)", query,
                  flags=re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        # normalize unit
        if "minute" in unit:
            return n, "minutes"
        if "hour" in unit:
            return n, "hours"
        if "day" in unit:
            return n, "days"
        if "week" in unit:
            return n, "weeks"
    return None, None


def _absolute_from_to_regex(query: str):
    """
    Look for 'from X to Y' or 'between X and Y' patterns
    Returns tuple(datetime or None, datetime or None)
    """
    m = re.search(r"(?:from|between)\s+(.*?)\s+(?:to|and)\s+(.*)", query, flags=re.I)
    if m:
        left = m.group(1).strip()
        right = m.group(2).strip()
        left_dt = dateparser.parse(left, settings={'RELATIVE_BASE': datetime.now()})
        right_dt = dateparser.parse(right, settings={'RELATIVE_BASE': datetime.now()})
        return left_dt, right_dt
    return None, None


def _search_dates(query: str):
    """
    Use dateparser.search.search_dates to find any dates/times in the free text.
    Returns list of (text, datetime) tuples, or [].
    """
    try:
        res = search_dates(query, settings={'RELATIVE_BASE': datetime.now(), 'RETURN_AS_TIMEZONE_AWARE': False})
        return res or []
    except Exception:
        return []


def _to_iso(dt):
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


# -------------------------
# Time range conversion
# -------------------------
def convert_label_and_query_to_range(label: str, query: str, now: datetime = None):
    """
    Given a predicted label and the original query, convert to (start: datetime, end: datetime).
    This function still uses deterministic logic but driven by ML label + extracted entities.
    """
    now = now or datetime.now()
    q = query.lower()

    # 1) Patterns for numeric relative times
    n, unit = _extract_number_unit(q)
    if label in ("last_n_minutes", "last_n_hours", "last_n_days") and n and unit:
        if "minutes" in unit:
            start = now - timedelta(minutes=n)
        elif "hours" in unit:
            start = now - timedelta(hours=n)
        elif "days" in unit:
            start = now - timedelta(days=n)
        elif "weeks" in unit:
            start = now - timedelta(weeks=n)
        else:
            start = now - timedelta(hours=1)
        end = now
        return start, end

    # 2) Absolute range like "from X to Y" or "between X and Y"
    left_dt, right_dt = _absolute_from_to_regex(q)
    if label == "absolute_range" and left_dt and right_dt:
        return left_dt, right_dt

    # 3) If label is absolute_range but regex didn't match, try searching dates
    if label == "absolute_range":
        found = _search_dates(q)
        if len(found) >= 2:
            return found[0][1], found[1][1]
        # if only one date/time found, take it as start and now as end
        if len(found) == 1:
            return found[0][1], now

    # 4) since_time
    if label == "since_time":
        found = _search_dates(q)
        if found:
            return found[0][1], now
        # try keywords like 'since morning', 'since midnight'
        if "since morning" in q or "since today morning" in q:
            start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            return start, now

    # 5) explicit keywords
    if label == "today" or "today" in q:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now

    if label == "yesterday" or "yesterday" in q:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start.replace(hour=23, minute=59, second=59, microsecond=0)
        return start, end

    if label == "this_week" or "this week" in q:
        start = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now

    if label == "last_week" or "last week" in q:
        start = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        return start, end

    # 6) fallback: if search_dates finds a single mention, treat as since_time
    found = _search_dates(q)
    if found:
        if len(found) >= 2:
            return found[0][1], found[1][1]
        return found[0][1], now

    # 7) default fallback -> last 1 hour
    return now - timedelta(hours=1), now


# -------------------------
# Timeframe Classifier
# -------------------------
class TimeframePrediction:
    def __init__(self, embedder_name: str = DEFAULT_EMBEDDER):
        self.embedder = SentenceTransformer(embedder_name)
        self.classifier = None
        self.label_names = DEFAULT_LABELS.copy()
        self.version = None

    @lru_cache(maxsize=5000)
    def embed(self, text: str):
        return self.embedder.encode([preprocess_text(text)], normalize_embeddings=True)[0]

    def load_data_from_mongo(self):
        docs = list(training_collection.find({}, {"_id": 0, "query": 1, "label": 1}))
        if not docs:
            logger.warning("No timeframe training data found in MongoDB 'timeframe_training' collection.")
            return pd.DataFrame(columns=["query", "label"])
        df = pd.DataFrame(docs)
        df["query"] = df["query"].astype(str).map(preprocess_text)
        df["label"] = df["label"].astype(str)
        return df

    def train(self, balance: bool = True):
        df = self.load_data_from_mongo()
        if df.empty:
            raise ValueError("No timeframe training data in MongoDB to train the model.")

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

        logger.info(f"Training timeframe classifier on {len(queries)} samples across {len(self.label_names)} labels")

        embeddings = self.embedder.encode(queries, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

        param_grid = {"alpha": [1e-4, 1e-3, 1e-2], "max_iter": [500, 1000]}
        grid = GridSearchCV(
            SGDClassifier(loss="log_loss", class_weight="balanced", n_jobs=-1),
            param_grid,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        base_clf = grid.best_estimator_
        logger.info(f"Best params: {grid.best_params_}")

        self.classifier = CalibratedClassifierCV(base_clf, cv="prefit")
        self.classifier.fit(X_train, y_train)

        preds = self.classifier.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)

        self.version = int(datetime.utcnow().timestamp())
        self._save_model_to_mongo(report, cm)

        logger.info("Timeframe training complete and saved to MongoDB.")
        return report

    def _save_model_to_mongo(self, report=None, cm=None):
        serialized = base64.b64encode(cloudpickle.dumps((self.classifier, self.label_names))).decode("utf-8")
        doc = {
            "name": "timeframe_classifier",
            "version": self.version,
            "model": serialized,
            "metrics": report,
            "confusion_matrix": cm.tolist() if cm is not None else None,
            "timestamp": datetime.utcnow()
        }
        models_collection.update_one({"name": "timeframe_classifier"}, {"$set": doc}, upsert=True)
        logger.info("Timeframe model saved in MongoDB 'timeframe_models' collection.")

    def load(self):
        doc = models_collection.find_one({"name": "timeframe_classifier"})
        if not doc:
            logger.warning("No saved timeframe model found in MongoDB.")
            return False
        serialized = base64.b64decode(doc["model"])
        self.classifier, self.label_names = cloudpickle.loads(serialized)
        self.version = doc.get("version")
        logger.info(f"Timeframe model v{self.version} loaded from MongoDB.")
        return True

    def predict(self, query: str, prob_threshold: float = 0.25, entropy_threshold: float = 1.5):
        """
        Predict label + convert to start/end datetimes.
        Returns dict:
         - label, confidence, entropy, probs (per-class)
         - start_iso, end_iso (or None)
         - raw_start, raw_end (datetime objects)
        """
        if not self.classifier:
            raise ValueError("Model not trained or loaded.")

        emb = self.embed(query).reshape(1, -1)
        probs = self.classifier.predict_proba(emb)[0]
        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        entropy = float(compute_entropy(probs))

        # low-confidence handling
        if confidence < prob_threshold or entropy > entropy_threshold:
            pred_label = "default"
        else:
            pred_label = str(self.classifier.classes_[best_idx])

        # convert to concrete datetime range
        start_dt, end_dt = convert_label_and_query_to_range(pred_label, query, now=datetime.now())

        return {
            "label": pred_label,
            "confidence": confidence,
            "entropy": entropy,
            "probs": dict(zip(map(str, self.classifier.classes_), map(float, probs))),
            "start": start_dt,
            "end": end_dt,
            "start_iso": _to_iso(start_dt),
            "end_iso": _to_iso(end_dt)
        }

    def queue_feedback(self, query: str, correct_label: str, notes: str = ""):
        feedback_collection.insert_one({
            "query": preprocess_text(query),
            "correct_label": correct_label,
            "notes": notes,
            "timestamp": datetime.utcnow()
        })
        logger.info(f"Queued feedback: '{query}' -> {correct_label}")
        return {"queued": True, "label": correct_label}

    def apply_feedback(self):
        feedbacks = list(feedback_collection.find({}))
        if not feedbacks:
            logger.info("No feedback to apply.")
            return None

        for fb in feedbacks:
            training_collection.update_one({"query": fb["query"]}, {"$set": {"label": fb["correct_label"]}}, upsert=True)

        feedback_collection.delete_many({})
        logger.info("Applied feedback to training data. Retraining model...")
        return self.train()


# -------------------------
# CLI / Example usage
# -------------------------
if __name__ == "__main__":
    clf = TimeframePrediction()
    if not clf.load():
        # If model not found, check if there's training data; if not, create a tiny default dataset for demo

        logger.info("Training timeframe model (this may take a while)...")
        report = clf.train()
        logger.info(f"Training report: {report}")

    # Example predictions
    examples = [
        "Check service-X errors last 2 hours",
        "Show logs since yesterday evening",
        "Get logs from 2025-09-10 10:00 to 2025-09-10 14:00",
        "Check the last 30 minutes",
        "Show today's errors",
        "Investigate between Sep 5 and Sep 7",
        "Check logs this week",
        "Check logs since morning"
    ]

    for q in examples:
        out = clf.predict(q)
        logger.info(f"Query: {q}\nPrediction: {out}\n")