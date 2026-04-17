def f_squared(clf, X, y):
    import sklearn.metrics as metrics

    n = X.shape[0]
    p = X.shape[1]
    r_squared = metrics.r2_score(y, clf.predict(X))
    return r_squared  / (1 - r_squared)

def f_stat(clf, X, y):
    """Calculate summary F-statistic for beta coefficients.

    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].

    Returns
    -------
    float
        The F-statistic value.
    """
    import sklearn.metrics as metrics

    n = X.shape[0]
    p = X.shape[1]
    r_squared = metrics.r2_score(y, clf.predict(X))
    return (r_squared / p) / ((1 - r_squared) / (n - p - 1))


def f_stat_pvalue(clf, X, y):
    """Calculate summary F-statistic p value for beta coefficients.

    Parameters
    ----------
    clf : sklearn.linear_model
        A scikit-learn linear model classifier with a `predict()` method.
    X : numpy.ndarray
        Training data used to fit the classifier.
    y : numpy.ndarray
        Target training values, of shape = [n_samples].

    Returns
    -------
    float
        The F-statistic p value.
    """
    import sklearn.metrics as metrics
    import numpy as np
    import scipy

    n = X.shape[0] # Esto se extrae par los grados de libertad del numeador y el denomindor (no. predictores, no. sujetos - no. predictores-1)
    p = X.shape[1]
    r_squared = metrics.r2_score(y, clf.predict(X))
    
    return np.round(scipy.stats.f.sf(f_stat(clf, X, y), n, (n - p - 1)), 15)

def coef_se(clf, X, y):
    import sklearn.metrics as metrics
    import numpy as np
    import scipy
    n = X.shape[0]
    X1 = np.hstack((np.ones((n, 1)), np.matrix(X)))
    se_matrix = scipy.linalg.sqrtm(
        metrics.mean_squared_error(y, clf.predict(X)) *
        np.linalg.inv(X1.T * X1))
    return np.diagonal(se_matrix)

def coef_tval_XGB_tree(clf, X, y):
    import numpy as np
    a = np.nan
    b = np.array( (clf.feature_importances_/ np.sum(clf.feature_importances_))  / coef_se(clf, X, y)[1:])
    return np.append(a, b)

def coef_pval_XGB_tree(clf, X, y):
    import numpy as np
    import scipy
    n = X.shape[0]
    t = coef_tval_XGB_tree(clf, X, y)
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 1))
    return p

def coef_tval_RF(clf, X, y):
    import numpy as np
    a = np.nan
    b = np.array(clf.feature_importances_ / coef_se(clf, X, y)[1:])
    return np.append(a, b)

def coef_pval_RF(clf, X, y):
    import numpy as np
    import scipy
    n = X.shape[0]
    t = coef_tval_RF(clf, X, y)
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 1))
    return p

def coef_tval(clf, X, y):
    import numpy as np
    a = np.array(clf.intercept_ / coef_se(clf, X, y)[0])
    b = np.array(clf.coef_ / coef_se(clf, X, y)[1:])
    return np.append(a, b)


def coef_pval(clf, X, y):
    import numpy as np
    import scipy
    n = X.shape[0]
    t = coef_tval(clf, X, y)
    p = 2 * (1 - scipy.stats.t.cdf(abs(t), n - 1))
    return p

#################################### Pseudo R²

#####Efron
def efron_rsquare(y, y_pred_proba):
    import numpy as np
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred_proba, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)

def skopt_efron_rsquare(estimator, X, y):
    import numpy as np
    
    y_pred_proba = estimator.predict_proba(X)[:,1]
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred_proba, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)

##### McFadden
def full_log_likelihood(w, X, y):
    import numpy as np
    score = np.dot(X, w).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def null_log_likelihood(w, X, y):
    import numpy as np
    z = np.array([w if i == 0 else 0.0 for i, w in enumerate(w.reshape(1, X.shape[1])[0])]).reshape(X.shape[1], 1)
    score = np.dot(X, z).reshape(1, X.shape[0])
    return np.sum(-np.log(1 + np.exp(score))) + np.sum(y * score)

def mcfadden_rsquare(w, X, y):
    return 1.0 - (full_log_likelihood(w, X, y) / null_log_likelihood(w, X, y))

def mcfadden_adjusted_rsquare(w, X, y):
    k = float(X.shape[1])
    return 1.0 - ((full_log_likelihood(w, X, y) - k) / null_log_likelihood(w, X, y))


