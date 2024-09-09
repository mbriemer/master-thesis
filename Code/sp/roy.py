import numpy as np
from scipy.stats import norm#, lognorm

def mvn_inverse_cdf(u, mu, sigma):
    """mvinv.m"""
    L = np.linalg.cholesky(sigma)
    z = norm.ppf(u)
    return mu + np.matmul(z, L.T)

def lognormpdf_ml(x, mu = 0, sigma = 1):
    """lognormpdf with Matlab-like arguments"""
    return norm.logpdf(x, mu, sigma)

def lognormcdf_ml(x, mu = 0, sigma = 1):
    """lognormcdf with Matlab-like arguments"""
    return norm.logcdf(x, mu, sigma)

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

def royinv(noise, theta, lambda_ = 0):
    """royinv.m"""
    
    # skip smoothing code for now

    size = np.array(noise.shape)
    size = size[1:]

    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    rho_t = 0
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
    """lognmaxpdf from logroypdf.m"""
    if rho == 1:
        if lognormcdf_ml(x, mu_1, sig_1) < lognormcdf_ml(x, mu_2, sig_2):
            return lognormpdf_ml(x, mu_1, sig_1)
        else:
            return lognormpdf_ml(x, mu_2, sig_2)
        
    elif rho == -1:
        if lognormcdf_ml(x, mu_1, sig_1) >= (1 - lognormcdf_ml(x, mu_2, sig_2)):
            return np.logaddexp(lognormpdf_ml(x, mu_1, sig_1), lognormpdf_ml(x, mu_2, sig_2))
        else:
            return -np.inf
        
    else:
        # Nadarajah and Kotz (2008, eq. (1--2))
        r = np.sqrt(1 - rho**2)
        x_1 = (x - mu_1) / sig_1 / r
        x_2 = (x - mu_2) / sig_2 / r
        p_1 = lognormpdf_ml(x, mu_1, sig_1) + lognormcdf_ml(x_2 - rho * x_1)
        p_2 = lognormpdf_ml(x, mu_2, sig_2) + lognormcdf_ml(x_1 - rho * x_2)
        if np.all(np.isnan(np.logaddexp(p_1, p_2))):
            pass
        return np.logaddexp(p_1, p_2)

def logexpnmaxpdf(z,a,b,mu1,mu2,sig1,sig2,rho):
    """logexpnmaxpdf from logroypdf.m"""
    p = np.full_like(z, -np.inf)
    j = np.argwhere(z > np.maximum(a,b))#, 1, 0)
    logzb = np.log(z[j] - b)
    logza = np.log(z[j] - a)
    r = np.sqrt(1 - rho**2)

    p1 = -logzb + lognormpdf_ml(logzb, mu2, sig2) + \
                  lognormcdf_ml(logza, mu1 + rho * sig1/sig2 * (logzb - mu2), r * sig1)
    
    p2 = -logza + lognormpdf_ml(logza, mu1, sig1) + \
                  lognormcdf_ml(logzb, mu2 + rho * sig2/sig1 * (logza - mu1), r * sig2)
    
    p[j] = np.logaddexp(p1, p2)
    return p

def logroypdf(y, theta):
    """logroypdf.m"""
    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    rho_t = 0
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
    p1d = lognormpdf_ml(log_w_1[i],
                                np.where(d_1 == 1, mu_1, mu_2)[i],
                                np.where(d_1 == 1, sigma_1, sigma_2)[i]) + \
                   lognormcdf_ml(np.log(a[i]),
                                 np.where(d_1 == 2, mu_1, mu_2)[i] + rho_s * \
                                 np.where(d_1 == 2, sigma_1, sigma_2)[i] / np.where(d_1 == 1, sigma_1, sigma_2)[i] * \
                                 (log_w_1[i] - np.where(d_1 == 1, mu_1, mu_2)[i]),
                                 r * np.where(d_1 == 2, sigma_1, sigma_2)[i])
    p1e = lognormpdf_ml(np.log(a[i]),
                                 np.where(d_1 == 2, mu_1, mu_2)[i],
                                 np.where(d_1 == 2, sigma_1, sigma_2)[i]) + \
                   lognormcdf_ml(log_w_1[i],
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

    p2d = lognormpdf_ml(log_w_2,
                  np.where(d_2 == 1, mu_1, mu_2) + 
                  np.where(d_2 == 1, gamma_1, gamma_2) * (d_1 == d_2),
                  np.where(d_2 == 1, sigma_1, sigma_2)) + \
          lognormcdf_ml(log_w_2,
                  np.where(d_2 == 2, mu_1, mu_2) + 
                  np.where(d_2 == 2, gamma_1, gamma_2) * (d_1 != d_2) + 
                  rho_s * np.where(d_2 == 2, sigma_1, sigma_2) / np.where(d_2 == 1, sigma_1, sigma_2) * 
                  (log_w_2 - np.where(d_2 == 1, mu_1, mu_2) - 
                   np.where(d_2 == 1, gamma_1, gamma_2) * (d_1 == d_2)),
                  r * np.where(d_2 == 2, sigma_1, sigma_2))
    
    p2e = lognormpdf_ml(log_w_2,
                  np.where(d_2 == 2, mu_1, mu_2) + 
                  np.where(d_2 == 2, gamma_1, gamma_2) * (d_1 != d_2),
                  np.where(d_2 == 2, sigma_1, sigma_2)) + \
          lognormcdf_ml(log_w_2,
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
    """roysupp.m"""

    mu_1, mu_2, gamma_1, gamma_2, sigma_1, sigma_2, rho_s = theta
    rho_t = 0
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
    """Lines 320-328 of main_roy.m"""
    while True:
        theta_perturbed = true_theta + rng.normal(0, 0.2, len(true_theta))
        theta_perturbed = np.clip(theta_perturbed, lower_bounds, upper_bounds)
        if roysupp(X, theta_perturbed) <= 0:
            return theta_perturbed

def perturb_uniform(X, lower_bounds, upper_bounds, rng):
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

