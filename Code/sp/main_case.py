"""main_case.m"""
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm

from roy import royinv
from other_discriminators import logistic_loss_2
from NND_sp import generator_loss

if __name__ == "__main__":

    n = m = 300 # Real and fake sample size
    S = 500 # Number of simulations
    g = 30 # Number of neural nets to average over

    true_theta = np.array([1.8, 2, 0.5, 0, 1, 1, 0.5, 0]) # switched rho_s and rho_t
    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, -1])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 1])

    # Observations
    Z_X = np.random.rand(n, 4)
    X = royinv(Z_X, true_theta)
    Z = np.random.rand(m, 4)

    # Perturb
    theta_initial_guess = true_theta + np.random.normal(0, 0.2, len(true_theta))
    theta_initial_guess = np.clip(theta_initial_guess, lower_bounds, upper_bounds)

    """ # Pre-estimate with logistic regression
    AdvL = opt.minimize(lambda theta : logistic_loss_2(X, royinv(Z, theta))[0],
                        x0 = theta_initial_guess,
                        method='Nelder-Mead',
                        bounds = list(zip(lower_bounds, upper_bounds)),
                        options={'return_all' : True, 'disp' : True, 'adaptive' : True})
    print(theta_initial_guess)
    print(AdvL.x) """
    
    # Other code that I skip

    # Loss plots

    K = 30 # Number of grid points
    param_names = [
        r'$\mu_1$',
        r'$\mu_2$',
        r'$\gamma_1$',
        r'$\gamma_2$',
        r'$\sigma_1$',
        r'$\sigma_2$',
        r'$\rho_s$',
        r'$\rho_t$'
    ]


    param_grid = np.linspace(lower_bounds, upper_bounds, K).T
    cD_grid = np.zeros_like(param_grid)
    NND_grid = np.zeros_like(param_grid)

    for i in tqdm(range(len(true_theta))):
        for k in tqdm(range(K)):
            theta = true_theta.copy()
            theta[i] = param_grid[i, k]
            cD_grid[i, k] = logistic_loss_2(X, royinv(Z, theta))[0]
            #NND_grid[i, k] = generator_loss(X, royinv(Z, theta), num_hidden=10, num_models=g)

    fig, axs = plt.subplots(4, 2)
    axs = axs.flatten()  # Flatten the 2D array of axes to make indexing easier

    for i in range(len(true_theta)):
        ax = axs[i]

        #ax.plot(param_grid[i, :], NND_grid[i, :], linewidth=1.5, color='blue', label='$\\mathbf{M}_\\theta(\\hat{D}_\\theta)$')
        ax.plot(param_grid[i, :], cD_grid[i, :], linewidth=1.5, color='red', label='$\\mathbf{L}_\\theta$')

        ax.axvline(x=true_theta[i], color='r', linestyle='--', label=f'True {param_names[i]}')

        ax.set_xlim(lower_bounds[i], upper_bounds[i])
        ax.set_ylim(-1.4, -1.25)

        ax.legend(loc='best', frameon=False)

    plt.tight_layout()
    plt.show()