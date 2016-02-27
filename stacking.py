def stacking_whole(estimator, train, labels, test, cv):
    preds_train = cross_val_predict(estimator, train, labels, cv=cv)
    preds_test = estimator.fit(train, labels).predict(test)
    
    return preds_train, preds_test

def stacking_whole(estimator, train, labels, test, cv):
    preds_train_ = []
    preds_test_ = []
    
    for train_idx, test_idx in cv:
        estimator.fit(train[train_idx], labels[train_idx])
        preds_train_.append(estimator.predict(train[test_idx]))
        preds_test_.append(estimator.predict(test))
        
    predictions_count = len(preds_train_[0][0])
    preds_train = np.zeros(train.shape[0], predictions_count)
    
    
    
def blend_models(estimator1, estimator2, X1, y1, X2=None, y2=None, n_folds=5, random_state=42):
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
            preds = np.argmax(alpha * pr1 + (1-alpha) * pr2, axis=1)
            scores[index] += accuracy_score(truth, preds)
    return scores / 5