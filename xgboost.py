from sklearn.base import BaseEstimator
import xgboost as xgb
import scipy
from tp import get_pool
import numpy as np


class XGBEstimator(BaseEstimator):
    def __init__(self, params):
        self.params = params

    def fit(self, X, y):
        xgtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(self.params, xgtrain, self.params['numround'])

    def predict(self, X):
        xgtest = xgb.DMatrix(X)
        return self.model.predict(xgtest)

    def predict_proba(self, X):
        return self.predict(X)

    def cv(self, X, y, n_fold=3, show_progress=False):
        xgtrain = xgb.DMatrix(X, y)
        return xgb.cv(self.params, xgtrain, self.params['numround'], nfold=n_fold, show_progress=show_progress)