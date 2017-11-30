import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

def calculate_statistics(train, test, by_field, fields, statistics=['mean']):
    '''Encodes statistics of field for every value in by_field'''
    
    if isinstance(fields, str):
        fields = [fields]
    if isinstance(statistics, str):
        statistics = [statistics]

    fields = list(set(fields) - set(by_field))

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


class _TargetEncoder(BaseEstimator):
    def __init__(self, field, functions, default_values, smooth):
        self.field = field
        self.smooth = smooth
        self.functions = functions
        self.default_values = default_values

    def fit(self, X, y):
        field_df = pd.DataFrame({self.field: X[self.field], 'target': y})
        self.means = field_df.groupby(self.field).aggregate(['count'] + self.functions).reset_index()
        self.means.columns = [self.field] + list(self.means.columns[1:].get_level_values(1))

        if self.smooth:
            for func, default_value in zip(self.functions, self.default_values):
                self.means[func] = default_value + 2. / np.pi * np.arctan(np.log1p(self.means['count'])) * (self.means[func] - default_value)

        return self

    def predict(self, X):
        mean_code = X[[self.field]].merge(self.means, on=self.field, how='left')[self.functions]
        for func, default_value in zip(self.functions, self.default_values):
            mean_code[func] = mean_code[func].fillna(default_value)
        return mean_code

def calculate_target_mean(train, test, labels, field, cv, functions=[np.mean, ], smooth=True):
    default_values = [func(labels) for func in functions]
    function_names = [func.__name__ for func in functions]

    train_encodings = cross_val_predict(_TargetEncoder(field, function_names, default_values, smooth), train, labels, cv=cv)
    test_encodings = _TargetEncoder(field, function_names, default_values, smooth).fit(train, labels).predict(test).values

    for index, func_name in enumerate(function_names):
        train[field + '_target_' + func_name] = train_encodings[:, index]
        test[field + '_target_' + func_name] = test_encodings[:, index]

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
