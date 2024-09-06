import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp


def logroypdf(y, theta):
    """
    Log of pdf of the Roy model.
    
    The logarithm of the pdf of the observations from a Roy model. It gives
    the pdf of (w1,d1,log(w2),d2), not (log(w1),d1,log(w2),d2), whereas the
    input y is (log(w1),d1,log(w2),d2). This is not an issue for MLE. Also,
    it is easy to fix using the change-of-variables formula if necessary.
    
    A closed-form expression seems to be available only when rho_t = 0.
    
    Input:
        y       Each column consisting of (log(w1),d1,log(w2),d2)'
        theta   Vector of (mu1,mu2,gam1,gam2,sig1,sig2,rhos)
    
    Output:
        p       Row vector of the log likelihood
    """
    
    y = y.T
    assert y.shape[0] == 4, 'First dimension of the data must be 4.'
    assert len(theta) == 7, 'Not enough number of parameters.'
    
    # Relabel parameters
    mu = theta[:2]
    gam = theta[2:4]
    sig = theta[4:6]
    rhot = 0  # Fixed to 0
    rhos = theta[6]
    beta = 0.9  # Fixed value
    r = np.sqrt(1 - rhos**2)
    
    assert rhot == 0, 'Nonzero rho_t is not supported.'
    
    # log(beta*E[w_2|d_1])
    # logbE[0]=log(beta*E[w_2|d_1=1]), logbE[1]=log(beta*E[w_2|d_1=2])
    logbE = np.log(beta) + logEexpmax(
        np.array([[mu[0]+gam[0], mu[0]], [mu[1], mu[1]+gam[1]]]),
        sig[0], sig[1], rhos
    )
    
    # log(v1) = log of observed value in period 1
    logv1 = np.logaddexp(y[0], logbE[y[1].astype(int) - 1])
    
    # PERIOD 1
    
    # pseudo-wage for the counterfactual sector in period 1 that yields the
    # same value as the observed one
    a = np.exp(logv1) - np.exp(logbE[1 - (y[1].astype(int) - 1)])  # Fixed indexing
    i = a > 0
    
    # Marginal log pdf of v1 (not of log(v1) nor of log(w1))
    p11 = np.full_like(logv1, -np.inf)
    p11[i] = logexpnmaxpdf(
        np.exp(logv1[i]), np.exp(logbE[0]), np.exp(logbE[1]),
        mu[0], mu[1], sig[0], sig[1], rhos
    )
    
    # Conditional log pdf of d1 given v1
    p1d = norm.logpdf(y[0, i], mu[y[1, i].astype(int) - 1], sig[y[1, i].astype(int) - 1]) + \
          norm.logcdf(
              np.log(a[i]),
              mu[2 - y[1, i].astype(int)] + rhos * sig[2 - y[1, i].astype(int)] / sig[y[1, i].astype(int) - 1] * (y[0, i] - mu[y[1, i].astype(int) - 1]),
              r * sig[2 - y[1, i].astype(int)]
          )
    p1e = norm.logpdf(np.log(a[i]), mu[2 - y[1, i].astype(int)], sig[2 - y[1, i].astype(int)]) + \
          norm.logcdf(
              y[0, i],
              mu[y[1, i].astype(int) - 1] + rhos * sig[y[1, i].astype(int) - 1] / sig[2 - y[1, i].astype(int)] * (np.log(a[i]) - mu[2 - y[1, i].astype(int)]),
              r * sig[y[1, i].astype(int) - 1]
          )
    p12 = np.full_like(p11, -np.inf)
    p12[i] = p1d - np.logaddexp(p1d, p1e)
    
    # PERIOD 2
    
    # Conditional log pdf of log(w2) given (w1,d1)
    p21 = lognmaxpdf(
        y[2],
        mu[0] + gam[0] * (y[1] == 1),
        mu[1] + gam[1] * (y[1] == 2),
        sig[0], sig[1], rhos
    )
    
    # Conditional log pdf of d2 given (w1,d1,w2)
    p2d = norm.logpdf(
        y[2],
        mu[y[3].astype(int) - 1] + gam[y[3].astype(int) - 1] * (y[1] == y[3]),
        sig[y[3].astype(int) - 1]
    ) + norm.logcdf(
        y[2],
        mu[2 - y[3].astype(int)] + gam[2 - y[3].astype(int)] * (y[1] == 3 - y[3]) + \
        rhos * sig[2 - y[3].astype(int)] / sig[y[3].astype(int) - 1] * \
        (y[2] - mu[y[3].astype(int) - 1] - gam[y[3].astype(int) - 1] * (y[1] == y[3])),
        r * sig[2 - y[3].astype(int)]
    )
    p2e = norm.logpdf(
        y[2],
        mu[2 - y[3].astype(int)] + gam[2 - y[3].astype(int)] * (y[1] == 3 - y[3]),
        sig[2 - y[3].astype(int)]
    ) + norm.logcdf(
        y[2],
        mu[y[3].astype(int) - 1] + gam[y[3].astype(int) - 1] * (y[1] == y[3]) + \
        rhos * sig[y[3].astype(int) - 1] / sig[2 - y[3].astype(int)] * \
        (y[2] - mu[2 - y[3].astype(int)] - gam[2 - y[3].astype(int)] * (y[1] == 3 - y[3])),
        r * sig[y[3].astype(int) - 1]
    )
    p22 = p2d - np.logaddexp(p2d, p2e)
    
    # AGGREGATE
    p = p11 + p12 + p21 + p22
    
    return p

def logexpnmaxpdf(z, a, b, mu1, mu2, sig1, sig2, rho):
    """
    Log of pdf of maximum of two shifted lognormal.
    Let [X,Y] ~ N([mu1,mu2],[sig1^2 rho*sig1*sig2; rho*sig1*sig2 sig2^2]).
    This function computes the log likelihood of Z = max(a+exp(X),b+exp(Y)).
    """
    p = np.full_like(z, -np.inf)
    i = z > np.maximum(a, b)
    logzb = np.log(z[i] - b)
    logza = np.log(z[i] - a)
    r = np.sqrt(1 - rho**2)
    p1 = -logzb + norm.logpdf(logzb, mu2, sig2) + \
         norm.logcdf(logza, mu1 + rho * sig1 / sig2 * (logzb - mu2), r * sig1)
    p2 = -logza + norm.logpdf(logza, mu1, sig1) + \
         norm.logcdf(logzb, mu2 + rho * sig2 / sig1 * (logza - mu1), r * sig2)
    p[i] = np.logaddexp(p1, p2)
    return p

def lognmaxpdf(x, mu1, mu2, sig1, sig2, rho):
    """Log of pdf of maximum of two normal."""
    if rho == 1:
        return np.where(
            norm.cdf(x, mu1, sig1) < norm.cdf(x, mu2, sig2),
            norm.logpdf(x, mu1, sig1),
            norm.logpdf(x, mu2, sig2)
        )
    elif rho == -1:
        return np.where(
            norm.cdf(x, mu1, sig1) >= norm.sf(x, mu2, sig2),
            np.logaddexp(norm.logpdf(x, mu1, sig1), norm.logpdf(x, mu2, sig2)),
            -np.inf
        )
    else:
        r = np.sqrt(1 - rho**2)
        x1 = (x - mu1) / sig1 / r
        x2 = (x - mu2) / sig2 / r
        p1 = norm.logpdf(x, mu1, sig1) + norm.logcdf(x2 - rho * x1)
        p2 = norm.logpdf(x, mu2, sig2) + norm.logcdf(x1 - rho * x2)
        return np.logaddexp(p1, p2)

def logEexpmax(mu, sig1, sig2, rho):
    """Log of E[exp(max(X,Y))] where [X,Y] ~ N(mu, Sigma)."""
    mu1, mu2 = mu
    theta = np.sqrt((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2)
    normal_dist = norm(0, 1)

    cdf1 = normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta)
    cdf2 = normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta)
    
    e1 = mu1 + sig1**2 / 2 + np.log(cdf1)
    e2 = mu2 + sig2**2 / 2 + np.log(cdf2)    
    e = np.logaddexp(e1, e2)
    return e