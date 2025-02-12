import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

from roy import logroypdf, royinv

def logistic_loss(X_1, X_2):
    """
    Caclulate a logistic regression loss between two equally-dimensional Roy model samples X_1 and X_2.

    Translation of the functions "loss" from main_roy.m and "loss1" from main_case.m.
    
    Parameters
    ----------
    X_1 : array-like, shape (n, 4)
        First sample.
    X_2 : array-like, shape (m, 4)
        Second sample.

    Returns
    -------
    v : float
        Loss value.
    coefficients : array-like, shape (7,)
        Coefficients of the logistic regression model.    
    """
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2
    n = len(d_1_1)
    m = len(d_1_2)
    assert n == m, "n == m required for logistic regression loss"

    Y = np.concatenate([np.ones(n), 0 * np.ones(m)])

    moments_1 = np.column_stack([log_w_1_1, d_1_1, log_w_2_1, d_2_1, log_w_1_1**2, log_w_2_1**2])
    moments_2 = np.column_stack([log_w_1_2, d_1_2, log_w_2_2, d_2_2, log_w_1_2**2, log_w_2_2**2])

    # Fit logistic regression model
    lr_model = LogisticRegression(fit_intercept=True)
    lr_model.fit(np.vstack([moments_1, moments_2]), Y)

    # Extract coefficients (lambda in MATLAB code)
    coefficients = np.concatenate([lr_model.intercept_, lr_model.coef_[0]])

    # Calculate loss
    v = np.mean(logistic.logcdf(np.dot(np.column_stack([np.ones(n), moments_1]), coefficients))) + \
        np.mean(logistic.logcdf(-np.dot(np.column_stack([np.ones(m), moments_2]), coefficients)))
    
    return v, coefficients

def logistic_loss_2(X_1, X_2):
    """
    Calculate a logistic regression loss between two equally-dimensional Roy model samples X_1 and X_2. 
    
    Translation of the function "loss2" from main_case.m.
    
    Parameters
    ----------
    X_1 : array-like, shape (n, 4)
        First sample.   
    X_2 : array-like, shape (m, 4)
        Second sample.

    Returns
    -------
    v : float
        Loss value.
    coefficients : array-like, shape (7,)
        Coefficients of the logistic regression model.
    """
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2
    n = len(d_1_1)
    m = len(d_1_2)
    assert n == m, "n == m required for logistic regression loss"

    Y = np.concatenate([np.ones(n), 0 * np.ones(m)])

    moments_1 = np.column_stack([log_w_1_1, d_1_1, log_w_2_1, d_2_1, log_w_1_1**2, log_w_2_1**2, log_w_1_1 * log_w_2_1])
    moments_2 = np.column_stack([log_w_1_2, d_1_2, log_w_2_2, d_2_2, log_w_1_2**2, log_w_2_2**2, log_w_1_2 * log_w_2_2])

    # Fit logistic regression model
    lr_model = LogisticRegression(fit_intercept=True)
    lr_model.fit(np.vstack([moments_1, moments_2]), Y)

    # Extract coefficients (lambda in MATLAB code)
    coefficients = np.concatenate([lr_model.intercept_, lr_model.coef_[0]])

    # Calculate loss
    v = np.mean(logistic.logcdf(np.dot(np.column_stack([np.ones(n), moments_1]), coefficients))) + \
        np.mean(logistic.logcdf(-np.dot(np.column_stack([np.ones(m), moments_2]), coefficients)))
                
    return v, coefficients

def OracleD(x, y, th_x, th_y):
    """
    Calculate the value of the oracle discriminator between two simulated Roy model samples.
    
    Translation of OracleD.m.
    
    Parameters
    ----------
    x : array-like, shape (n, 4)
        First sample.
    y : array-like, shape (m, 4)
        Second sample.
    th_x : array-like, shape (7,)
        Economic parameters of the Roy model for the first sample.
    th_y : array-like, shape (7,)
        Economic parameters of the Roy model for the second sample.

    Returns
    -------
    v : float
        Oracle discriminator value.    
    """
    logpxx = logroypdf(x, th_x)
    logpxy = logroypdf(x, th_y)
    logpyx = logroypdf(y, th_x)
    logpyy = logroypdf(y, th_y)
    
    v = np.mean(logpxx - np.logaddexp(logpxx, logpxy)) + \
        np.mean(logpyy - np.logaddexp(logpyx, logpyy))
    
    return v

"""
# Test

u_1 = np.random.rand(100, 4)
u_2 = np.random.rand(100, 4)
theta_1 = np.array([1.8, 2, 0.5, 0, 1, 1, 0.5])
theta_2 = np.array([1.9, 2.1, 0.6, 0.1, 1.1, 1.1, 0.6])
X_1 = royinv(u_1, theta_1)
X_2 = royinv(u_2, theta_2)
print(logistic_loss(X_1, X_2))
"""