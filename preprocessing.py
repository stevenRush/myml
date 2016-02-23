import pandas as pd

def calc_mean(train, test, field, by_field):
    '''Encodes mean of field for every value in by_field'''

    df = pd.concat([train, test]).reset_index(drop = True)
    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], df.shape[0])
    temp = df.groupby(by_field).mean()[[field]].reset_index()
    temp.columns = [by_field, field + '_by_' + by_field + '_mean']
    df = df.merge(temp, on=by_field, how='left')
    train = df.ix[train_idx, :]
    test = df.ix[test_idx, :]

    return train, test

