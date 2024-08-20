# Helper functions for the simulation of the Roy model

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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

def royinv(noise, theta, lambda_val, num_samples):
    """royinv.m"""
    
    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t, beta = theta

    eps_mu = np.zeros(4)
    eps_sigma = np.array([[sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2],
                                [rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2],
                                [rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2],
                                [rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2]])
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

    return np.stack([logw1, d1, logw2, d2], axis = -1)

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
