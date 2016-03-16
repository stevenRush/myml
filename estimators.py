from sklearn.base import BaseEstimator


class ProbaEstimator(BaseEstimator):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    def predict(self, X):
        return self.estimator.predict_proba(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)