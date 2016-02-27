import numpy as np
from sklearn.cross_validation import StratifiedKFold, cross_val_predict
from sklearn.metrics import log_loss

from estimators import ProbaEstimator


SCORING_FUNCS = {
    'log_loss': log_loss
}


def stacking_whole(estimator, train, labels, test, cv):
    '''
        Performs stacking with fitting over whole train
    '''

    predictions_train = cross_val_predict(estimator, train, labels, cv=cv)
    predictions_test = estimator.fit(train, labels).predict(test)
    
    return predictions_train, predictions_test


def stacking_average_byfolds(estimator, train, labels, test, cv):
    '''
        Performs stacking averaging predictions for test over all folds
    '''

    preds_train_ = []
    preds_test_ = []
    
    for train_idx, test_idx in cv:
        estimator.fit(train[train_idx], labels[train_idx])
        preds_train_.append(estimator.predict(train[test_idx]))
        preds_test_.append(estimator.predict(test))
        
    predictions_count = len(preds_train_[0][0])
    preds_train = np.zeros(train.shape[0], predictions_count)
    
    
    
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