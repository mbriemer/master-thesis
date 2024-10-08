import torch

import numpy as np
from scipy.stats import norm#, lognorm
from scipy.linalg import sqrtm

def mvn_inverse_cdf(u, mu, sigma):
    # Compute the square root of sigma using eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    L = torch.matmul(eigenvectors, torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0))))
    
    # Ensure L is real
    L = L.real
    
    # Compute the inverse CDF (percent point function)
    z = torch.special.ndtri(u)  # equivalent to norm.ppf in SciPy
    
    return mu + torch.matmul(z, L.T)

def logEexpmax(mu1, mu2, sig1, sig2, rho):
    """logEexpmax.m"""
    theta = torch.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = torch.distributions.Normal(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + torch.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + torch.log(cdf2)
    e = torch.logaddexp(e1, e2)
    return e

def royinv(noise, theta, lambda_ = 0):
    """royinv.m"""
    
    if len(theta) == 7:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
        rho_t = torch.tensor(0., device=theta.device)
        beta = torch.tensor(0.9, device=theta.device)
    elif len(theta) == 8:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t = theta
        beta = torch.tensor(0.9, device=theta.device)
    elif len(theta) == 9:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t, beta = theta

    # Covariance matrix
    Sigma = torch.stack([
        torch.stack([sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2]),
        torch.stack([rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2]),
        torch.stack([rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2]),
        torch.stack([rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2])
    ], dim=0)

    # Shocks
    epsilons = mvn_inverse_cdf(noise, torch.zeros(4, device=theta.device), Sigma)
    eps_1_1 = epsilons[:,0]
    eps_1_2 = epsilons[:,1]
    eps_2_1 = epsilons[:,2]
    eps_2_2 = epsilons[:,3]

    # Log wages at t = 1 for each sector
    log_w_1_1 = mu_1 + eps_1_1
    log_w_1_2 = mu_2 + eps_1_2

    # Log value functions at t = 1 for each sector
    log_v_1_1 = torch.logaddexp(log_w_1_1,
                             torch.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s))
    log_v_1_2 = torch.logaddexp(log_w_1_2,
                             torch.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
    
    # Sector choices at t = 1
    d_1 = torch.where(log_v_1_1 > log_v_1_2, 1., 2.)

    # Observed log wages at t = 1
    log_w_1 = torch.where(d_1 == 1, log_w_1_1, log_w_1_2)

    # Log wages at t = 2 for each sector
    log_w_2_1 = torch.where(d_1 == 1,
                        mu_1 + gamma_1 + eps_2_1,
                        mu_1 + eps_2_1) 
    log_w_2_2 = torch.where(d_1 == 2,
                        mu_2 + gamma_2 + eps_2_2,
                        mu_2 + eps_2_2)

    # Sector choices at t = 2
    d_2 = torch.where(log_w_2_1 > log_w_2_2, 1., 2.)

    # Observed log wages at t = 2
    log_w_2 = torch.where(d_2 == 1, log_w_2_1, log_w_2_2)

    if lambda_ > 0:
        d_1 = 1 + torch.distributions.Normal(0,
                                             lambda_ * torch.std(log_v_1_1 - log_v_1_2)).cdf(log_v_1_1 - log_v_1_2)
        d_2 = 1 + torch.distributions.Normal(0,
                                             lambda_ * torch.std(log_w_2_1 - log_w_2_2)).cdf(log_w_2_1 - log_w_2_2)

    return torch.stack([log_w_1, d_1, log_w_2, d_2], dim = 1)

def soft_royinv(noise, theta):
    if len(theta) == 7:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
        rho_t = torch.tensor(0., device=theta.device)
        beta = torch.tensor(0.9, device=theta.device)
    elif len(theta) == 8:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t = theta
        beta = torch.tensor(0.9, device=theta.device)
    elif len(theta) == 9:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t, beta = theta

    # Covariance matrix
    Sigma = torch.stack([
        torch.stack([sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2]),
        torch.stack([rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2]),
        torch.stack([rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2]),
        torch.stack([rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2])
    ], dim=0)

    # Shocks
    epsilons = mvn_inverse_cdf(noise, torch.zeros(4, device=theta.device), Sigma)
    eps_1_1 = epsilons[:,0]
    eps_1_2 = epsilons[:,1]
    eps_2_1 = epsilons[:,2]
    eps_2_2 = epsilons[:,3]

    # Log wages at t = 1 for each sector
    log_w_1_1 = mu_1 + eps_1_1
    log_w_1_2 = mu_2 + eps_1_2

    # Log value functions at t = 1 for each sector
    log_v_1_1 = torch.logaddexp(log_w_1_1,
                             torch.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s))
    
    log_v_1_2 = torch.logaddexp(log_w_1_2,
                              torch.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
    
    # Sector choices at t = 1
    d_1 = torch.sigmoid(log_v_1_1 - log_v_1_2)

    # Observed log wages at t = 1
    log_w_1 = d_1 * log_w_1_1 + (1 - d_1) * log_w_1_2

    # Log wages at t = 2 for each sector
    log_w_2_1 = d_1 * (mu_1 + gamma_1 + eps_2_1) + (1 - d_1) * (mu_1 + eps_2_1)
    log_w_2_2 = d_1 * (mu_2 + eps_2_2) + (1 - d_1) * (mu_2 + gamma_2 + eps_2_2)

    # Sector choices at t = 2
    d_2 = torch.sigmoid(log_w_2_1 - log_w_2_2)

    # Observed log wages at t = 2
    log_w_2 = d_2 * log_w_2_1 + (1 - d_2) * log_w_2_2

    return torch.stack([log_w_1, d_1, log_w_2, d_2], dim = 1)


""" # Testing 
u = torch.rand(300, 4)
theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5])

true_values = royinv(u, theta)
print(true_values) """


### sp

def mvn_inverse_cdf_sp(u, mu, sigma):
    """mvinv.m"""
    L = np.real(sqrtm(sigma))
    z = norm.ppf(u)
    return mu + np.matmul(z, L.T)

def logEexpmax_sp(mu1, mu2, sig1, sig2, rho):
    """logEexpmax.m"""
    theta = np.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = norm(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + np.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + np.log(cdf2)
    e = np.logaddexp(e1, e2)
    return e

def royinv_sp(noise, theta, lambda_ = 0):
    """royinv.m"""
    
    if len(theta) == 7:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
        rho_t = 0
    else:
        mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s, rho_t = theta
    beta = 0.9

    # Covariance matrix
    Sigma = np.array([[sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**1, rho_s * rho_t * sigma_1 * sigma_2],
                      [rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2],
                      [rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2],
                      [rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2]])
    
    # Shocks
    epsilons = mvn_inverse_cdf_sp(noise, np.zeros(4), Sigma)
    eps_1_1 = epsilons[:,0]
    eps_1_2 = epsilons[:,1]
    eps_2_1 = epsilons[:,2]
    eps_2_2 = epsilons[:,3]

    # Log wages at t = 1 for each sector
    log_w_1_1 = mu_1 + eps_1_1
    log_w_1_2 = mu_2 + eps_1_2

    # Log value functions at t = 1 for each sector
    log_v_1_1 = np.logaddexp(log_w_1_1,
                             np.log(beta) + logEexpmax_sp(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s))
    log_v_1_2 = np.logaddexp(log_w_1_2,
                             np.log(beta) + logEexpmax_sp(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
    
    # Sector choices at t = 1
    d_1 = np.where(log_v_1_1 > log_v_1_2, 1, 2)

    # Observed log wages at t = 1
    log_w_1 = np.where(d_1 == 1, log_w_1_1, log_w_1_2)

    # Log wages at t = 2 for each sector
    log_w_2_1 = np.where(d_1 == 1,
                        mu_1 + gamma_1 + eps_2_1,
                        mu_1 + eps_2_1) 
    log_w_2_2 = np.where(d_1 == 2,
                        mu_2 + gamma_2 + eps_2_2,
                        mu_2 + eps_2_2)

    # Sector choices at t = 2
    d_2 = np.where(log_w_2_1 > log_w_2_2, 1, 2)

    # Observed log wages at t = 2
    log_w_2 = np.where(d_2 == 1, log_w_2_1, log_w_2_2)

    if lambda_ > 0:
        d_1 = 1 + norm.cdf(log_v_1_1 - log_v_1_2, 
                           0,
                           lambda_ * np.std(log_v_1_1 - log_v_1_2))
        d_2 = 1 + norm.cdf(log_w_2_1 - log_w_2_2,
                            0,
                            lambda_ * np.std(log_w_2_1 - log_w_2_2))

    return log_w_1, d_1, log_w_2, d_2