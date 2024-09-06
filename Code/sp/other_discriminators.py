import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

#from translation import logroypdf
from roy import logroypdf

def logistic_loss(X1, X2):
    """Function "loss" from main_roy.m"""
    n, m = X1.shape[1], X2.shape[1]
    labels = np.concatenate([np.ones(n), 2 * np.ones(m)])
    
    reg1 = np.column_stack([X1.T, X1[0, :]**2, X1[2, :]**2])
    reg2 = np.column_stack([X2.T, X2[0, :]**2, X2[2, :]**2])
    
    X = np.vstack([reg1, reg2])
    
    # Fit logistic regression model
    lr_model = LogisticRegression(fit_intercept=True)
    lr_model.fit(X, labels)
    
    # Extract coefficients (lambda in MATLAB code)
    lambda_coef = np.concatenate([lr_model.intercept_, lr_model.coef_[0]])
    
    # Calculate loss
    v1 = np.mean(logistic.logcdf(np.dot(np.column_stack([np.ones((n, 1)), reg1]), lambda_coef)))
    v2 = np.mean(logistic.logcdf(-np.dot(np.column_stack([np.ones((m, 1)), reg2]), lambda_coef)))
    v = v1 + v2
    
    return v, lambda_coef

def OracleD(x, y, th_x, th_y):
    """OracleD.m"""
    logpxx = logroypdf(x, th_x)
    logpxy = logroypdf(x, th_y)
    logpyx = logroypdf(y, th_x)
    logpyy = logroypdf(y, th_y)
    
    v = np.mean(logpxx - np.logaddexp(logpxx, logpxy)) + \
        np.mean(logpyy - np.logaddexp(logpyx, logpyy))
    
    return v