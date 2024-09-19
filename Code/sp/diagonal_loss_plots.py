import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from roy import logroypdf, royinv
from NND_sp import generator_loss as NND
from other_discriminators import logistic_loss_2 

# Parameter grid
# Grid limits from main_roy.m
lower_lims = np.array([1.4, 1.4, 0.1, -0.4, 0.8, 0.7, -0.2])
upper_lims = np.array([2.3, 2.5, 0.9, 0.4, 1.3, 1.3, 0.9])
lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, -1])
upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 1])
wide_lower_bounds = np.array([-20, -20, -10, -10, 0, 0, -1, -1])
wide_upper_bounds = np.array([20, 20, 10, 10, 10, 10, 1, 1])

K = 30
param_grid = np.linspace(wide_lower_bounds, wide_upper_bounds, K).T

cD_grid = np.zeros_like(param_grid[:2,:])
NND_grid = np.zeros_like(param_grid[:2,:])

theta = [1.8, 2, 0.5, 0, 1, 1, 0.5, 0]

n = 300
m = 300
lambda_ = 0
g = 30

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

diag_names = [
    r'($\mu_1, \mu_2)$',
    r'$(\mu_1, - \mu_2)$'
]

Z = np.random.rand(m, 4)  # Latent variables for synthetic observations
X = royinv(Z, theta)

for k in tqdm(range(K)):
    th = theta.copy()
    th[:2] = param_grid[:2, k]
    NND_grid[0,k] = NND(X, royinv(Z, th), num_hidden=10, num_models=g)
    cD_grid[0,k] = logistic_loss_2(X, royinv(Z, th))[0]

for k in tqdm(range(K)):
    th = theta.copy()
    th[0] = param_grid[0, k]
    th[1] = param_grid[1, -k-1]
    NND_grid[1,k] = NND(X, royinv(Z, th), num_hidden=10, num_models=g)
    cD_grid[1,k] = logistic_loss_2(X, royinv(Z, th))[0]

fig, axs = plt.subplots(1, 2)

for i in range(2):
    ax = axs[i]

    ax.plot(param_grid[i, :], NND_grid[i, :], linewidth=1.5, color='blue', label='NN')
    ax.plot(param_grid[i, :], cD_grid[i, :], linewidth=1.5, color='red', label='Logistic')

    for j in range(2):
        ax.axvline(x=theta[j], color='black', linestyle='--', label=f'True {param_names[j]}')


    max = np.max([np.max(NND_grid[i, :]), np.max(cD_grid[i, :])]) + 0.1
    min = np.min([np.min(NND_grid[i, :]), np.min(cD_grid[i, :])]) - 0.1
    ax.set_xlim(wide_lower_bounds[i], wide_upper_bounds[i])
    ax.set_ylim(min, max)

    ax.legend(loc='best', frameon=False)

plt.tight_layout()
plt.show()