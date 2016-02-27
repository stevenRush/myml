import numpy as np
from sklearn.base import BaseEstimator

def data_to_rgf(data, file_prefix):
    data.to_csv('{0}.x'.format(file_prefix), sep=' ', header=0, index=0)

def labels_to_rgf(labels, file_prefix):
    if np.unique(labels).shape[0] != 2:
        raise ValueError('RGF works only for binary classification')
    
    labels = np.vectorize(lambda x: '+1' if x == 1 else '-1')(labels)
    np.savetxt('{0}.y'.format(file_prefix), labels, fmt='%s')
    

class RGFEstimator(BaseEstimator):
    def __init__(self, params):
        self.params = params
    def fit(self, X, y):
        pass
    def predict(self, X):
        pass
    def predict_proba(self, X):
        pass