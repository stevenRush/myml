import numpy as np
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss

from estimators import ProbaEstimator


SCORING_FUNCS = {
    'log_loss': log_loss
}

def get_classes_count(predictions):
    if predictions.size == 0:
        return 0
    return int(predictions.size * 1. / predictions.shape[0])


def stacking_whole(estimator, train, labels, test, cv, fname_prefix=None):
    '''
        Performs stacking with fitting over whole train
    '''

    predictions_train = cross_val_predict(estimator, train, labels, cv=cv)
    predictions_test = estimator.fit(train, labels).predict(test)

    if fname_prefix is not None:
        np.savetxt('{0}_train.csv'.format(fname_prefix), predictions_train)
        np.savetxt('{0}_test_whole.csv'.format(fname_prefix), predictions_test)
    else:
        return predictions_train, predictions_test


def stacking_average_byfolds(estimator, train, labels, test, cv, fname_prefix=None):
    '''
        Performs stacking averaging predictions for test over all folds
    '''

    preds_train_ = []
    preds_test_ = []
    
    for train_idx, test_idx in cv:
        estimator.fit(train[train_idx], labels[train_idx])
        preds_train_.append(estimator.predict(train[test_idx]))
        preds_test_.append(estimator.predict(test))
        
    classes_count = get_classes_count(preds_train_[0])
    predictions_train = np.zeros(shape=(train.shape[0], classes_count))

    for index, (train_idx, test_idx) in enumerate(cv):
        predictions_train[test_idx] = preds_train_[index]

    predictions_test = np.hstack(preds_test_).mean(axis=1)

    if fname_prefix is not None:
        np.savetxt('{0}_train.csv'.format(fname_prefix), predictions_train)
        np.savetxt('{0}_test_average.csv'.format(fname_prefix), predictions_test)
    else:
        return predictions_train, predictions_test


def stacking_both(estimator, train, labels, test, cv, fname_prefix=None):
    '''
        Performs stacking using both methods
    '''

    preds_train_ = []
    preds_test_ = []

    for train_idx, test_idx in cv:
        estimator.fit(train[train_idx], labels[train_idx])
        preds_train_.append(estimator.predict(train[test_idx]))
        preds_test_.append(estimator.predict(test))

    classes_count = get_classes_count(preds_train_[0])
    predictions_train = np.zeros(shape=(train.shape[0], classes_count))

    for index, (train_idx, test_idx) in enumerate(cv):
        predictions_train[test_idx] = preds_train_[index]

    predictions_test = np.hstack(preds_test_).mean(axis=1)
    predictions_test_whole =  estimator.fit(train, labels).predict(test)

    if fname_prefix is not None:
        np.savetxt('{0}_train.csv'.format(fname_prefix), predictions_train)
        np.savetxt('{0}_test_average.csv'.format(fname_prefix), predictions_test)
        np.savetxt('{0}_test_whole.csv'.format(fname_prefix), predictions_test_whole)
    else:
        return predictions_train, predictions_test, predictions_test_whole

    
def blend_models(estimator1, estimator2, X1, y1, X2=None, y2=None, n_folds=5, random_state=42, scoring=None):
    if isinstance(scoring, str):
        if scoring not in SCORING_FUNCS:
            raise KeyError('Unknown scoring function')
        scoring = SCORING_FUNCS[scoring]

    need_argmax = scoring in ['accuracy']

    weights = np.linspace(0, 1, 100)
    scores = np.zeros_like(weights)
    
    if X2 is None:
        X2 = X1
        y2 = y1
    
    cv = StratifiedKFold(y1, n_folds=n_folds, random_state=random_state)
    preds1 = cross_val_predict(ProbaEstimator(estimator1), X1, y1, cv=cv)
    preds2 = cross_val_predict(ProbaEstimator(estimator2), X2, y2, cv=cv)
    
    for train_idx, test_idx in cv:
        pr1 = preds1[test_idx]
        pr2 = preds2[test_idx]
        truth = y1[test_idx]

        for index, alpha in enumerate(weights):
            if need_argmax:
                preds = np.argmax(alpha * pr1 + (1-alpha) * pr2, axis=1)
            else:
                preds = alpha * pr1 + (1-alpha) * pr2
            scores[index] += scoring(truth, preds)
    return scores / 5