import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

def warn_info(message, category, filename, lineno, file=None, line=None):
    print(f"Convergence Warning Details:")
    print(f"Message: {message}")
    print(f"Category: {category}")
    print(f"Filename: {filename}")
    print(f"Line number: {lineno}")
    
    # You can add more custom information here
    print("Additional Info: Check your model parameters and data.")

def quadratic_loss(true_theta, theta):
    noise = 0.01 * np.random.randn(7)
    return np.sum(((true_theta - theta) + noise)**2)