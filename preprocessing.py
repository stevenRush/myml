import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_validation import cross_val_predict

def calculate_statistics(train, test, by_field, fields, statistics=['mean']):
    '''Encodes statistics of field for every value in by_field'''
    
    if isinstance(fields, str):
        fields = [fields]
    if isinstance(statistics, str):
        statistics = [statistics]

    df = pd.concat([train, test]).reset_index(drop = True)
    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], df.shape[0])
    temp = df.groupby(by_field).aggregate(statistics)[fields].reset_index()
    temp.columns = [by_field] + [field + '_by_' + by_field + '_' + func for field in fields for func in statistics]
    df = df.merge(temp, on=by_field, how='left')
    train = df.ix[train_idx, :]
    test = df.ix[test_idx, :]

    return train, test


def calculate_counts(train, test, field, drop=False):
    df = pd.concat([train, test]).reset_index(drop = True)
    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], df.shape[0])

    counts = pd.DataFrame(df.groupby(field).count().iloc[:, 0])
    counts.columns = [field + '_cnt']
    train = train.merge(counts, left_on=field, right_index=1, how='left')
    test = test.merge(counts, left_on=field, right_index=1, how='left')

    if drop:
        train.drop(field, axis=1, inplace=1)
        test.drop(field, axis=1, inplace=1)

    return train, test


class _TargetMeanEncoder(BaseEstimator):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y):
        field_df = pd.DataFrame({self.field: X[self.field], 'target': y})
        self.means = field_df.groupby(self.field).mean().reset_index()
        self.target_mean = y.mean()
        return self

    def predict(self, X):
        mean_code = X[[self.field]].merge(self.means, on=self.field, how='left')['target']
        return mean_code.fillna(self.target_mean)

def calculate_target_mean(train, test, labels, field, cv):
    train_mean = cross_val_predict(_TargetMeanEncoder(field), train, labels, cv=cv)
    test_mean = _TargetMeanEncoder(field).fit(train, labels).predict(test)

    train[field + '_target_mean'] = train_mean
    test[field + '_target_mean'] = test_mean

    return train, test


def log1p_negative(array):
    negative = array < 0
    array = np.log1p(np.abs(array))
    array[negative] *= -1
    return array

def sqrt_negative(array):
    negative = array < 0
    array = np.abs(array) ** 0.5
    array[negative] *= -1
    return array