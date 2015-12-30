def _cv_score(estimator, X, y, scoring=None, cv=None):
    import scipy
    from sklearn.cross_validation import cross_val_score
    
    scipy.random.seed()
    return cross_val_score(estimator, X, y, scoring=scoring, cv=cv)

def average_cross_val_score(estimator, X, y, scoring=None, cv=None, n_iter=5, cores=16):
    from multiprocessing import Pool
    
    
    results = []
    p = Pool(processes=cores)
    for iter in range(n_iter):
        results.append(p.apply_async(_cv_score, args=(estimator, X, y), kwds={'scoring':scoring, 'cv':cv}))
    return np.vstack([result.get() for result in results])