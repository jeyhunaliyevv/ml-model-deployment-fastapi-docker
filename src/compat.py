"""Thin scikit-learn compatible CatBoost wrapper for recent sklearn tags."""
from catboost import CatBoostClassifier
from sklearn.base import ClassifierMixin, BaseEstimator

class CatBoostAdapter(ClassifierMixin, BaseEstimator):
    def __init__(self, iterations=180, depth=4, learning_rate=0.05, random_seed=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed

    def fit(self, X, y):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, random_seed=self.random_seed,
            verbose=False, allow_writing_files=False, thread_count=1)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X).ravel()
