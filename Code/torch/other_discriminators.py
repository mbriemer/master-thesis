#from tqdm import tqdm
import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression

def logistic_loss3(X_1, X_2):
    """Function "loss2" from main_case.m"""
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1[:, 0], X_1[:, 1], X_1[:, 2], X_1[:, 3]
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2[:, 0], X_2[:, 1], X_2[:, 2], X_2[:, 3]
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