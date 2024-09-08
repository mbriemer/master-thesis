import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

#from translation import logroypdf
from roy import logroypdf, royinv

# Helper functions

def loglogisticcdf(x, mu = 0, sigma = 1):
    """loglogisticcdf.m"""
    

# Losses

def logistic_loss(X_1, X_2):
    """Function "loss" from main_roy.m"""
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2
    n = len(d_1_1)
    m = len(d_1_2)
    assert n == m, "n == m required for logistic regression loss"

    Y = np.concatenate([np.ones(n), 2 * np.ones(m)])

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

def OracleD(x, y, th_x, th_y):
    """OracleD.m"""
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