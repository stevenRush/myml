from sklearn.base import BaseEstimator
import xgboost as xgb
import scipy
from xgboost import cv
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
    def cv(self, X, y, n_fold=3, show_progress=False):
        xgtrain = xgb.DMatrix(X, y)
        return xgb.cv(self.params, xgtrain, self.params['numround'], nfold=n_fold, show_progress=show_progress)   
        
        
def _xgboost_cv_score(params, X, y, n_fold):
    scipy.random.seed()
    xgtrain = xgb.DMatrix(X, y)
    return cv(params, xgtrain, params['numround'], nfold=n_fold).iloc[:, 0].values

    
def xgboost_average_cross_val_score(params, X, y, n_fold=3, n_iter=5, cores=16):
    results = []
    p = get_pool()
    for iter in range(n_iter):
        results.append(p.apply_async(_xgboost_cv_score, args=(params, X, y, n_fold)))
    return np.vstack([result.get() for result in results]).mean(axis=0)