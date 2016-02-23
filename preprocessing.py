import pandas as pd

def calculate_statistics(train, test, by_field, field, statistics=['mean']):
    '''Encodes mean of field for every value in by_field'''

    df = pd.concat([train, test]).reset_index(drop = True)
    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], df.shape[0])
    temp = df.groupby(by_field).aggregate(statistics)[[field]].reset_index()
    temp.columns = [by_field, field + '_by_' + by_field + '_mean']
    df = df.merge(temp, on=by_field, how='left')
    train = df.ix[train_idx, :]
    test = df.ix[test_idx, :]

    return train, test


def calculate_counts(train, test, field):
    df = pd.concat([train, test]).reset_index(drop = True)
    train_idx = range(train.shape[0])
    test_idx = range(train.shape[0], df.shape[0])

    counts = pd.DataFrame(df.groupby(field).count().iloc[:, 0])
    counts.columns = [field + '_cnt']
    train = train.merge(counts, left_on=field, right_index=1, how='left')
    test = test.merge(counts, left_on=field, right_index=1, how='left')
    return train, test