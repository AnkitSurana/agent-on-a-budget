"""Step 4, part 2: a small, classic ML model (the receptionist) learns from the LLM's labels.

How it works, in plain words:
1. TF-IDF turns each message into numbers. We use small pieces of words ("char n-grams"),
   so typos like "ordr" still look a lot like "order".
2. Logistic Regression learns which numbers go with which intent.
3. For a new message it gives a probability for every intent. The highest one is our "confidence".
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

from budget_agent import config


def make_model():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        LogisticRegression(C=20, max_iter=2000),
    )


def train(texts, labels):
    model = make_model()
    model.fit(list(texts), list(labels))
    return model


def predict(model, text):
    """Returns (intent, confidence) for one message. Confidence is between 0 and 1."""
    probabilities = model.predict_proba([text])[0]
    best = probabilities.argmax()
    return str(model.classes_[best]), float(probabilities[best])


def accuracy(model, texts, labels):
    return accuracy_score(list(labels), model.predict(list(texts)))


def save(model, path=config.CLASSIFIER_PATH):
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)


def load(path=config.CLASSIFIER_PATH):
    return joblib.load(path)


def load_or_train(path=None):
    """Load the receptionist. Just downloaded the project? Then it learns now (a few seconds) and is saved."""
    path = path or config.CLASSIFIER_PATH
    if not path.exists():
        import time
        from budget_agent.data import training_data
        texts, labels, source = training_data()
        print(f"No trained receptionist yet: training it now on {source} (a few seconds)...", flush=True)
        start = time.perf_counter()
        save(train(texts, labels), path)
        print(f"Receptionist trained in {time.perf_counter() - start:.1f}s and saved, so next time it loads instantly.", flush=True)
    return load(path)
