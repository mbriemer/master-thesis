"""Functions for generating observations from the Roy model."""
import torch

def mvn_inverse_cdf(u, mu, sigma):
    """
    Generate samples from a multivariate normal distribution using the inverse CDF method with eigendecomposition.

    Similar to mvninv.m.

    Parameters
    ----------
    u : torch.Tensor
        A tensor of shape (n, d) containing the quantiles of the standard normal distribution.
    mu : torch.Tensor
        A tensor of shape (d,) containing the mean of the multivariate normal distribution.
    sigma : torch.Tensor
        A tensor of shape (d, d) containing the covariance matrix of the multivariate normal distribution.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, d) containing the samples from the multivariate normal distribution.

    """
    # Compute the square root of sigma using eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma)
    L = torch.matmul(eigenvectors, torch.diag(torch.sqrt(torch.clamp(eigenvalues, min=0))))
    
    # Ensure L is real
    L = L.real
    
    # Compute the inverse CDF (percent point function)
    z = torch.special.ndtri(u)  # equivalent to norm.ppf in SciPy
    
    return mu + torch.matmul(z, L.T)

def logEexpmax(mu1, mu2, sig1, sig2, rho):
    """
    Compute the logarithm of the expected maximum of two jointly normal random variables.

    Translation of logEexpmax.m.
    
    Parameters
    ----------
    mu1 : torch.Tensor
        Mean of the first Gaussian random variable.
    mu2 : torch.Tensor
        Mean of the second Gaussian random variable.
    sig1 : torch.Tensor
        Standard deviation of the first Gaussian random variable (must be positive).
    sig2 : torch.Tensor
        Standard deviation of the second Gaussian random variable (must be positive).
    rho : torch.Tensor
        Correlation coefficient between the two Gaussian random variables (must be in [-1, 1]).

    Returns
    -------
    torch.Tensor
        The logarithm of the expected maximum of the two Gaussian random variables.
    """
    theta = torch.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = torch.distributions.Normal(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + torch.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + torch.log(cdf2)
    e = torch.logaddexp(e1, e2)
    return e

def royinv(noise, theta, lambda_ = 0):
    """
    Generate observations from the Roy model (log wages and sector choices).
    
    Translation of royinv.m.
    
    Parameters
    ----------
    noise : torch.Tensor
        Noise vector used for sampling shocks from a multivariate normal distribution.
    theta : torch.Tensor of shape (7,) or (8,) or (9,)
        A vector of economic parameters of the Roy model as defined in Section 3.2 of Kaji, Manresa and Pouliot (2023).
        If the length of theta is 7, then beta is set to 0.9 and rho_t is set to 0.
        If the length of theta is 8, then beta is set to 0.9.
    lambda_ : float, optional
        A parameter for smoothing sector choices (default is 0).

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, 4) containing the log wages and sector choices for each individual in the sample.
        The columns are: log wage at t = 1, sector choice at t = 1, log wage at t = 2, sector choice at t = 2.
    """
    
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

""" # Testing 
u = torch.rand(300, 4)
theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5])

true_values = royinv(u, theta)
print(true_values) """