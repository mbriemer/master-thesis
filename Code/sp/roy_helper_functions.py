# Helper functions for the simulation of the Roy model

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import logsumexp

def logEexpmax(mu1, mu2, sig1, sig2, rho):
    """logEexpmax.m"""
    theta = np.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = norm(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + np.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + np.log(cdf2)    
    e = np.logaddexp(e1, e2)
    return e

def mvn_inverse_cdf(u, mu, sigma):
    """mvinv.m"""
    L = np.linalg.cholesky(sigma)
    z = norm.ppf(u)
    return mu + np.matmul(z, L.T)

def smooth(x, lambda_val): # TODO Check
    """Smoothing code in royinv.m"""
    if lambda_val == 0:
        return x
    else:
        diffs = np.diff(x).float()
        std_diffs = np.std(diffs)
        smoothed = 1 + np.special.ndtr(diffs / (lambda_val * std_diffs))
        return smoothed.squeeze()
    
def logexpnmaxpdf(z, a, b, mu1, mu2, sig1, sig2, rho):
    """From logroypdf.m"""
    p = np.full_like(z, -np.inf)
    i = z > np.maximum(a, b)
    logzb = np.log(z[i] - b)
    logza = np.log(z[i] - a)
    r = np.sqrt(1 - rho**2)
    
    p1 = (-logzb + norm.logpdf(logzb, mu2, sig2) + 
          norm.logcdf(logza, mu1 + rho * sig1 / sig2 * (logzb - mu2), r * sig1))
    p2 = (-logza + norm.logpdf(logza, mu1, sig1) + 
          norm.logcdf(logzb, mu2 + rho * sig2 / sig1 * (logza - mu1), r * sig2))
    
    p[i] = logsumexp([p1, p2], axis=0)
    return p

def lognmaxpdf(x, mu1, mu2, sig1, sig2, rho):
    "From logroypdf.m"
    if rho == 1:
        if norm.cdf(x, mu1, sig1) < norm.cdf(x, mu2, sig2):
            return norm.logpdf(x, mu1, sig1)
        else:
            return norm.logpdf(x, mu2, sig2)
    elif rho == -1:
        if norm.cdf(x, mu1, sig1) >= norm.sf(x, mu2, sig2):
            return logsumexp([norm.logpdf(x, mu1, sig1), norm.logpdf(x, mu2, sig2)])
        else:
            return -np.inf
    else:
        r = np.sqrt(1 - rho**2)
        x1 = (x - mu1) / (sig1 * r)
        x2 = (x - mu2) / (sig2 * r)
        p1 = norm.logpdf(x, mu1, sig1) + norm.logcdf(x2 - rho * x1)
        p2 = norm.logpdf(x, mu2, sig2) + norm.logcdf(x1 - rho * x2)
        return logsumexp([p1, p2])
    

import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp

def logroypdf(y, theta):
    """
    Calculate log probability density for the Roy model.
    
    Parameters:
    y: array-like, shape (n_samples, 4)
        Data array where each row represents (log_wage1, sector1, log_wage2, sector2)
    theta: array-like, shape (9,)
        Parameter vector (mu1, mu2, gamma1, gamma2, sigma1, sigma2, rho_s, rho_t, beta)
    
    Returns:
    p: array-like, shape (n_samples,)
        Log probability density for each sample
    """
    mu1, mu2, gamma1, gamma2, sigma1, sigma2, rho_s, rho_t, beta = theta
    
    # Transpose y if it's not in the expected shape
    if y.shape[1] != 4:
        y = y.T
    
    log_wage1, sector1, log_wage2, sector2 = y.T
    
    # Helper function for log of expected max
    def log_expected_max(mu_a, mu_b, sigma_a, sigma_b, rho):
        theta = np.sqrt((sigma_a - sigma_b)**2 + 2 * (1 - rho) * sigma_a * sigma_b)
        z = (mu_a - mu_b + sigma_a**2 - rho * sigma_a * sigma_b) / theta
        return mu_a + sigma_a**2/2 + np.log(norm.cdf(z)) + \
               mu_b + sigma_b**2/2 + np.log(norm.cdf(-z))
    
    # Calculate log of expected future wages
    log_expected_future = np.log(beta) + np.where(
        sector1 == 0,
        log_expected_max(mu1 + gamma1, mu2, sigma1, sigma2, rho_s),
        log_expected_max(mu1, mu2 + gamma2, sigma1, sigma2, rho_s)
    )
    
    # Calculate log of value function
    log_value = logsumexp([log_wage1, log_expected_future], axis=0)
    
    # Calculate log probability of sector choice in period 1
    log_prob_sector1 = np.where(
        sector1 == 0,
        norm.logcdf((log_wage1 - mu2) / sigma2),
        norm.logcdf((log_wage1 - mu1) / sigma1)
    )
    
    # Calculate log probability of wage in period 1
    log_prob_wage1 = np.where(
        sector1 == 0,
        norm.logpdf(log_wage1, mu1, sigma1),
        norm.logpdf(log_wage1, mu2, sigma2)
    )
    
    # Calculate log probability of wage in period 2
    mu2_cond = np.where(
        sector1 == 0,
        mu1 + gamma1 + rho_t * sigma1 / sigma2 * (log_wage1 - mu1),
        mu2 + gamma2 + rho_t * sigma2 / sigma1 * (log_wage1 - mu2)
    )
    sigma2_cond = np.sqrt(1 - rho_t**2) * np.where(sector1 == 0, sigma1, sigma2)
    log_prob_wage2 = norm.logpdf(log_wage2, mu2_cond, sigma2_cond)
    
    # Calculate log probability of sector choice in period 2
    log_prob_sector2 = np.where(
        sector2 == 0,
        norm.logcdf((log_wage2 - mu2_cond) / sigma2_cond),
        norm.logcdf(-(log_wage2 - mu2_cond) / sigma2_cond)
    )
    
    # Combine all log probabilities
    return log_prob_sector1 + log_prob_wage1 + log_prob_wage2 + log_prob_sector2

def royinv(noise, theta, lambda_val, num_samples):
    """royinv.m"""
    
    #try:
    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    rho_t = 0
    beta = 0.9

    eps_mu = np.zeros(4)
    eps_sigma = np.array([[sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2],
                            [rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2],
                            [rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2],
                            [rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2]])

    # Check if the covariance matrix is positive definite
    #if not np.all(np.linalg.eigvals(eps_sigma) > 0):
    #    print(f"Covariance matrix is not positive definite for theta: {theta}")
    #    return None

    eps = mvn_inverse_cdf(noise, eps_mu, eps_sigma)
    
    # Log wages at t = 1 for each sector

    logw1m = np.column_stack((mu_1 + eps[:, 0],
                                mu_2 + eps[:, 1]))

    # Value function at t = 1 for each sector

    logv1m = np.column_stack((
        np.logaddexp(logw1m[:, 0], np.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s)),
        np.logaddexp(logw1m[:, 1], np.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
    ))

    # Sector choices at t = 1

    d1 = np.argmax(logv1m, axis=1)

    # Observed log wages at t = 1

    logw1 = logv1m[np.arange(num_samples), d1]

    # Log wages at t == 2

    logw2m = np.column_stack((
        mu_1 + gamma_1 * (d1 == 0) + eps[:, 2],
        mu_2 + gamma_2 * (d1 == 1) + eps[:, 3]
    ))

    # % Observed log wages and sector choices at t = 2

    logw2 = np.max(logw2m, axis=1)
    d2 = np.argmax(logw2m, axis=1)

    #d1_smooth = smooth(x = logv1m, lambda_val = lambda_val)
    #d2_smooth = smooth(x = logw2m, lambda_val = lambda_val)

    return np.stack([logw1, d1, logw2, d2], axis=-1)
    #except Exception as e:
    #    print(f"Error in royinv for theta {theta}: {str(e)}")
    #    return None

def plot_results(results):
    all_mu_values, all_sigma_values, all_discriminator_losses, all_generator_losses, iteration_numbers = results
    num_repetitions = len(all_mu_values)
    
    plt.figure(figsize=(12, 5))

    # Plot mu
    plt.subplot(2, 2, 1)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_mu_values[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('μ over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('μ')
    
    # Plot sigma
    plt.subplot(2, 2, 2)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_sigma_values[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('σ over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('σ')
    
    # Plot discriminator loss
    plt.subplot(2, 2, 3)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_discriminator_losses[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('Discriminator loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot generator loss
    plt.subplot(2, 2, 4)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_generator_losses[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('Generator loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
