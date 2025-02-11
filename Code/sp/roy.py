import numpy as np
from scipy.stats import norm#, lognorm
from scipy.linalg import sqrtm

def mvn_inverse_cdf(u, mu, sigma):
    """
    Generate samples from a multivariate normal distribution using the inverse CDF method.
    
    Translation of mvinv.m.

    Parameters
    ----------
    u : np.ndarray
        An array of shape (n, d) containing the quantiles of the standard normal distribution.
    mu : np.ndarray
        An array of shape (d,) containing the mean of the multivariate normal distribution.
    sigma : np.ndarray
        An array of shape (d, d) containing the covariance matrix of the multivariate normal distribution.

    Returns
    -------
    np.ndarray
        An array of shape (n, d) containing the samples from the multivariate normal distribution.
    """
    L = np.real(sqrtm(sigma))
    z = norm.ppf(u)
    return mu + np.matmul(z, L.T)

def logEexpmax(mu1, mu2, sig1, sig2, rho):
    """
    Compute the logarithm of the expected maximum of two jointly normal random variables.

    Translation of logEexpmax.m.
    
    Parameters
    ----------
    mu1 : float
        Mean of the first Gaussian random variable.
    mu2 : float
        Mean of the second Gaussian random variable.
    sig1 : float
        Standard deviation of the first Gaussian random variable (must be positive).
    sig2 : float
        Standard deviation of the second Gaussian random variable (must be positive).
    rho : float
        Correlation coefficient between the two Gaussian random variables (must be in [-1, 1]).

    Returns
    -------
    float
        The logarithm of the expected maximum of the two Gaussian random variables.    
    """
    theta = np.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = norm(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + np.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + np.log(cdf2)
    e = np.logaddexp(e1, e2)
    return e

def royinv(noise, theta, lambda_ = 0):
    """
    Generate observations from the Roy model (log wages and sector choices).
    
    Translation of royinv.m.
    
    Parameters
    ----------
    noise : np.ndarray
        An array of shape (n, 4) containing the quantiles of the standard normal distribution.
    theta : np.ndarray
        A vector of economic parameters of the Roy model as defined in Section 3.2 of Kaji, Manresa and Pouliot (2023).
        If the length of theta is 7, then beta is set to 0.9 and rho_t is set to 0.
        If the length of theta is 8, then beta is set to 0.9.
    lambda_ : float, optional
        A scalar representing the strength of the perturbation (default is 0).
    
    Returns
    -------
    np.ndarray
        An array of shape (n,) containing the log wages at t = 1.
    np.ndarray
        An array of shape (n,) containing the sector choices at t = 1.
    np.ndarray
        An array of shape (n,) containing the log wages at t = 2.
    np.ndarray
        An array of shape (n,) containing the sector choices at t = 2.    
    """
    
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
    epsilons = mvn_inverse_cdf(noise, np.zeros(4), Sigma)
    eps_1_1 = epsilons[:,0]
    eps_1_2 = epsilons[:,1]
    eps_2_1 = epsilons[:,2]
    eps_2_2 = epsilons[:,3]

    # Log wages at t = 1 for each sector
    log_w_1_1 = mu_1 + eps_1_1
    log_w_1_2 = mu_2 + eps_1_2

    # Log value functions at t = 1 for each sector
    log_v_1_1 = np.logaddexp(log_w_1_1,
                             np.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s))
    log_v_1_2 = np.logaddexp(log_w_1_2,
                             np.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
    
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

def lognmaxpdf(x,mu_1,mu_2,sig_1,sig_2,rho):
    """
    Calculate the logarithm of the pdf of the maximum of two normal random variables.
    
    Translation of lognmaxpdf (line 129-160) from logroypdf.m.

    Parameters
    ----------
    x : float
        The value at which to evaluate the pdf. 
    mu_1 : float
        Mean of the first Gaussian random variable.
    mu_2 : float
        Mean of the second Gaussian random variable.
    sig_1 : float
        Standard deviation of the first Gaussian random variable (must be positive).
    sig_2 : float
        Standard deviation of the second Gaussian random variable (must be positive).
    rho : float
        Correlation coefficient between the two Gaussian random variables (must be in [-1, 1]).

    Returns
    -------
    float
        The logarithm of the pdf of the maximum of the two Gaussian random variables.
    """
    if rho == 1:
        if norm.logcdf(x, mu_1, sig_1) < norm.logcdf(x, mu_2, sig_2):
            return norm.lopdf(x, mu_1, sig_1)
        else:
            return norm.lopdf(x, mu_2, sig_2)
        
    elif rho == -1:
        if norm.logcdf(x, mu_1, sig_1) >= (1 - norm.logcdf(x, mu_2, sig_2)):
            return np.logaddexp(norm.lopdf(x, mu_1, sig_1), norm.lopdf(x, mu_2, sig_2))
        else:
            return -np.inf
        
    else:
        # Nadarajah and Kotz (2008, eq. (1--2))
        r = np.sqrt(1 - rho**2)
        x_1 = (x - mu_1) / sig_1 / r
        x_2 = (x - mu_2) / sig_2 / r
        p_1 = norm.lopdf(x, mu_1, sig_1) + norm.logcdf(x_2 - rho * x_1)
        p_2 = norm.lopdf(x, mu_2, sig_2) + norm.logcdf(x_1 - rho * x_2)
        if np.all(np.isnan(np.logaddexp(p_1, p_2))):
            pass
        return np.logaddexp(p_1, p_2)

def logexpnmaxpdf(z,a,b,mu1,mu2,sig1,sig2,rho):
    """
    Calculate the logarithm of the pdf of the maximum of two shifted lognormal random variables.
    
    Translation of logexpnmaxpdf (lines 115-125) from logroypdf.m.
    Computes the log likelihood of Z = max(a+exp(X), b+exp(Y)) where X and Y follow a bivariate normal distribution:
    [X,Y] ~ N([mu1,mu2], [[sig1^2, rho*sig1*sig2], [rho*sig1*sig2, sig2^2]])

    Parameters
    ----------
    z : float
        The value at which to evaluate the pdf.
    a : float
        First shift parameter.
    b : float
        Second shift parameter.
    mu1 : float
        Mean of the first Gaussian random variable.
    mu2 : float
        Mean of the second Gaussian random variable.
    sig1 : float
        Standard deviation of the first Gaussian random variable (must be positive).
    sig2 : float
        Standard deviation of the second Gaussian random variable (must be positive).
    rho : float
        Correlation coefficient between the two Gaussian random variables (must be in [-1, 1]).

    Returns
    -------
    p : ndarray
        Log probability density function evaluated at the specified points.
    """
    p = np.full_like(z, -np.inf)
    j = np.argwhere(z > np.maximum(a,b))#, 1, 0)
    logzb = np.log(z[j] - b)
    logza = np.log(z[j] - a)
    r = np.sqrt(1 - rho**2)

    p1 = -logzb + norm.lopdf(logzb, mu2, sig2) + \
                  norm.logcdf(logza, mu1 + rho * sig1/sig2 * (logzb - mu2), r * sig1)
    
    p2 = -logza + norm.lopdf(logza, mu1, sig1) + \
                  norm.logcdf(logzb, mu2 + rho * sig2/sig1 * (logza - mu1), r * sig2)
    
    p[j] = np.logaddexp(p1, p2)
    return p

def logroypdf(y, theta):
    """
    Calculate the logarithm of the pdf of obversations from the Roy model.

    Translation of (most of) logroypdf.m.
    Comments carried over from there.

    Parameters
    ----------
    y : np.ndarray
        An array of shape (n, 4) containing the observations from the Roy model.
    theta : np.ndarray
        A vector of economic parameters of the Roy model, with rho_t set to 0.

    Returns
    -------
    np.ndarray
        An array of shape (n,) containing the logarithm of the pdf of the observations.    
    """
    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    beta = 0.9

    r = np.sqrt(1-rho_s**2)
    log_w_1, d_1, log_w_2, d_2 = y

    # logbE(1)=log(beta*E[w_2|d_1=1]) 
    # logbE(2)=log(beta*E[w_2|d_1=2])
    logbE_1 = np.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s)
    logbE_2 = np.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s)

    # log(v1) = log of observed value in period 1
    log_v_1 = np.logaddexp(log_w_1,
                            np.where(d_1 == 1, logbE_1, logbE_2))
    
    
    ### PERIOD 1 ###

    # pseudo-wage for the counterfactual sector in period 1 that yields the
    # same value as the observed one

    a = np.exp(log_v_1) - np.exp(np.where(d_1 == 2, logbE_1, logbE_2))
    i = np.argwhere(a > 0)

    # Marginal log pdf of v1 (not of log(v1) nor of log(w1))
    p11 = np.full_like(log_v_1, -np.inf)
    p11[i] = logexpnmaxpdf(np.exp(log_v_1[i]), np.exp(logbE_1), np.exp(logbE_2), mu_1, mu_2, sigma_1, sigma_2, rho_s)

    # Conditional log pdf of d1 given v1
    p1d = norm.lopdf(log_w_1[i],
                                np.where(d_1 == 1, mu_1, mu_2)[i],
                                np.where(d_1 == 1, sigma_1, sigma_2)[i]) + \
                   norm.logcdf(np.log(a[i]),
                                 np.where(d_1 == 2, mu_1, mu_2)[i] + rho_s * \
                                 np.where(d_1 == 2, sigma_1, sigma_2)[i] / np.where(d_1 == 1, sigma_1, sigma_2)[i] * \
                                 (log_w_1[i] - np.where(d_1 == 1, mu_1, mu_2)[i]),
                                 r * np.where(d_1 == 2, sigma_1, sigma_2)[i])
    p1e = norm.lopdf(np.log(a[i]),
                                 np.where(d_1 == 2, mu_1, mu_2)[i],
                                 np.where(d_1 == 2, sigma_1, sigma_2)[i]) + \
                   norm.logcdf(log_w_1[i],
                                np.where(d_1 == 1, mu_1, mu_2)[i] + rho_s * \
                                np.where(d_1 == 1, sigma_1, sigma_2)[i] / np.where(d_1 == 2, sigma_1, sigma_2)[i] * \
                                (np.log(a[i]) - np.where(d_1 == 2, mu_1, mu_2)[i]),
                                r * np.where(d_1 == 1, sigma_1, sigma_2)[i])
    
    p12 = np.full_like(p11, -np.inf)
    p12[i] = p1d - np.logaddexp(p1d, p1e) 

    ### PERIOD 2 ###

    # Conditional log pdf of log(w2) given (w1,d1)
    p21 = lognmaxpdf(log_w_2,
                    mu_1 + np.where(d_1 == 1, gamma_1, 0),
                    mu_2 + np.where(d_1 == 2, gamma_2, 0),
                    sigma_1, sigma_2, rho_s)
    
    # Conditional log pdf of d2 given (w1,d1)

    p2d = norm.lopdf(log_w_2,
                  np.where(d_2 == 1, mu_1, mu_2) + 
                  np.where(d_2 == 1, gamma_1, gamma_2) * (d_1 == d_2),
                  np.where(d_2 == 1, sigma_1, sigma_2)) + \
          norm.logcdf(log_w_2,
                  np.where(d_2 == 2, mu_1, mu_2) + 
                  np.where(d_2 == 2, gamma_1, gamma_2) * (d_1 != d_2) + 
                  rho_s * np.where(d_2 == 2, sigma_1, sigma_2) / np.where(d_2 == 1, sigma_1, sigma_2) * 
                  (log_w_2 - np.where(d_2 == 1, mu_1, mu_2) - 
                   np.where(d_2 == 1, gamma_1, gamma_2) * (d_1 == d_2)),
                  r * np.where(d_2 == 2, sigma_1, sigma_2))
    
    p2e = norm.lopdf(log_w_2,
                  np.where(d_2 == 2, mu_1, mu_2) + 
                  np.where(d_2 == 2, gamma_1, gamma_2) * (d_1 != d_2),
                  np.where(d_2 == 2, sigma_1, sigma_2)) + \
          norm.logcdf(log_w_2,
                  np.where(d_2 == 1, mu_1, mu_2) + 
                  np.where(d_2 == 1, gamma_1, gamma_2) * (d_1 == d_2) + 
                  rho_s * np.where(d_2 == 1, sigma_1, sigma_2) / np.where(d_2 == 2, sigma_1, sigma_2) * 
                  (log_w_2 - np.where(d_2 == 2, mu_1, mu_2) - 
                   np.where(d_2 == 2, gamma_1, gamma_2) * (d_1 != d_2)),
                  r * np.where(d_2 == 1, sigma_1, sigma_2))
    
    p22 = np.logaddexp(p2d, p2e)
    #print(p11[:10], p12[:10], p21[:10], p22[:10])

    return p11 + p12 + p21 + p22

def roysupp(y, theta):
    """
    Check if points are outside the support of the Roy model.
    
    Evaluates whether given points lie outside the support of logroypdf with 
    specified parameters. Returns positive values for points outside the support.
    Translation of roysupp.m.

    Parameters
    ----------
    y : np.ndarray
        An array of shape (n, 4) containing the observations from the Roy model.
    theta : np.ndarray
        A vector of economic parameters of the Roy model, with rho_t set to 0.
    
    Returns
    -------
    np.ndarray
        Indicator values. Positive values indicate points outside the support of the Roy model with the given parameters.    
    """

    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    beta = 0.9

    log_w_1, d_1, log_w_2, d_2 = y

    # logbE(1)=log(beta*E[w_2|d_1=1]) 
    # logbE(2)=log(beta*E[w_2|d_1=2])
    logbE_1 = np.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s)
    logbE_2 = np.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s)

    # log(v1) = log of observed value in period 1
    log_v_1 = np.logaddexp(log_w_1,
                            np.where(d_1 == 1, logbE_1, logbE_2))
    
    a = np.exp(log_v_1) - np.exp(np.where(d_1 == 2, logbE_1, logbE_2))
    c = - np.min(a)

    return c

def perturb(X, true_theta, lower_bounds, upper_bounds, rng):
    """
    Perturb the true parameter vector by adding normal noise, while ensuring it is within the support of the Roy model.
    
    Translation of lines 320-328 of main_roy.m.
    
    Parameters
    ----------
    X : np.ndarray
        An array of shape (n, 4) containing the observations from the Roy model.
    true_theta : np.ndarray
        A vector of economic parameters of the Roy model, with rho_t set to 0.
    lower_bounds : np.ndarray
        A vector of lower bounds for the economic parameters.
    upper_bounds : np.ndarray
        A vector of upper bounds for the economic parameters.
    rng : np.random.Generator
        A random number generator.
    
    Returns
    -------
    np.ndarray
        A perturbed parameter vector that lies within the support of the Roy model.
    """
    while True:
        theta_perturbed = true_theta + rng.normal(0, 0.2, len(true_theta))
        theta_perturbed = np.clip(theta_perturbed, lower_bounds, upper_bounds)
        if roysupp(X, theta_perturbed) <= 0:
            return theta_perturbed

def perturb_uniform(X, lower_bounds, upper_bounds, rng):
    """
    Sample a parameter vector from a uniform distribution, while ensuring it is within the support of the Roy model.

    Parameters
    ----------
    X : np.ndarray
        An array of shape (n, 4) containing the observations from the Roy model.
    lower_bounds : np.ndarray
        A vector of lower bounds for the economic parameters.
    upper_bounds : np.ndarray
        A vector of upper bounds for the economic parameters.
    rng : np.random.Generator
        A random number generator.

    Returns
    -------
    np.ndarray
        A parameter vector that lies within the support of the Roy model.
    """
    while True:
        theta_perturbed = rng.uniform(low=lower_bounds, high=upper_bounds)
        if roysupp(X, theta_perturbed) <= 0:
            return theta_perturbed

"""       
# Testing 
u = np.random.rand(300, 4)
theta = np.array([1.8, 2, 0.5, 0, 1, 1, 0.5])

true_values = royinv(u, theta)
lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1])
upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1])

perturbed_theta = perturb(true_values, theta, lower_bounds, upper_bounds)
print(perturbed_theta)
"""