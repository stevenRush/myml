def _cv_score(estimator, X, y, scoring=None, cv=None):
    import scipy
    from sklearn.cross_validation import cross_val_score
    
    scipy.random.seed()
    return cross_val_score(estimator, X, y, scoring=scoring, cv=cv)

    
def average_cross_val_score(estimator, X, y, scoring=None, cv=None, n_iter=5, cores=16):
    from multiprocessing import Pool
    import numpy as np
    
    results = []
    p = Pool(processes=cores)
    for iter in range(n_iter):
        results.append(p.apply_async(_cv_score, args=(estimator, X, y), kwds={'scoring':scoring, 'cv':cv}))
    return np.vstack([result.get() for result in results])
    
    
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

def average_blend(estimator1, estimator2, X1, y1, X2=None, y2=None, n_iter=16, n_folds=5):
    from multiprocessing import Pool
    
    p = Pool(processes=16)
    results = []
    for iter in range(n_iter):
        results.append(p.apply_async(blend_models, (estimator1, estimator2, X1, y1, X2, y2), 
                                     {'n_folds':n_folds, 'random_state':iter}))
    
    return np.vstack([result.get() for result in results]).mean(axis=0)