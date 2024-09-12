import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from roy import logroypdf, royinv
from NND_sp import generator_loss as NND
from other_discriminators import OracleD

# Parameter grid
K = 30
param_grid = np.linspace(np.array([1.4, 1.4, 0.1, -0.4, 0.8, 0.7, -0.2]),
                         np.array([2.3, 2.5, 0.9, 0.4, 1.3, 1.3, 0.9]),
                         K)

cD_grid = np.zeros_like(param_grid)
NND_grid = np.zeros_like(param_grid)
OracleD_grid = np.zeros_like(param_grid)
LL_grid = np.zeros_like(param_grid)

# Assume these variables are defined elsewhere
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

Z = np.random.rand(m, 4)  # Latent variables for synthetic observations
X = royinv(Z, theta)

for i in tqdm(range(len(theta))):
    for k in tqdm(range(K)):
        #print(f"Parameter {i+1}/{len(theta)}, iteration {k+1}/{K}")
        th = theta.copy()
        th[i] = param_grid[i, k]
        #cD_grid[i, k] = loss(X, royinv(Z, th, smooth=lambda_))
        NND_grid[i, k] = NND(X, royinv(Z, th), num_hidden=10, num_models=g)
        OracleD_grid[i, k] = OracleD(X, royinv(Z, th), theta, th)
        LL_grid[i, k] = -np.mean(logroypdf(X, th)) / 2

# Plotting
fig, axs = plt.subplots(4, 2)  # Increased height
axs = axs.flatten()  # Flatten the 2D array of axes to make indexing easier

for i in range(len(theta)):
    ax = axs[i]
    axL = ax.twinx()
    
    # Plot all curves on the same y-axis
    ax.plot(param_grid[i, :], NND_grid[i, :], linewidth=1.5, color='blue', label='$\\mathbf{M}_\\theta(\\hat{D}_\\theta)$')
    ax.plot(param_grid[i, :], OracleD_grid[i, :], linewidth=1.5, color='green', label='$\\mathbf{M}_\\theta(D_\\theta)$')
    axL.plot(param_grid[i, :], (LL_grid[i, :] / 2), linewidth=1.5, color='red', label='$\\mathbf{L}_\\theta$')

    # Add vertical line for true parameter value
    ax.axvline(x=theta[i], color='r', linestyle='--', label=f'True {param_names[i]}')
    
    # Set axes limits
    ax.set_xlim(param_grid[i, 0], param_grid[i, -1])
    ax.set_ylim(-1.4, -1.25)
    axL.set_ylim((LL_grid[i, :].min())/2 - 0.01, (LL_grid[i, np.isfinite(LL_grid[i,:])].max())/2 + 0.01)
  
    # Add legend
    ax.legend(loc='best', frameon=False)
    
    # Add title and labels
    #ax.set_ylabel(f'Parameter: {param_names[i]}', fontsize=12)  # Increased padding
    #ax.set_xlabel(param_names[i], fontsize=12)
    #ax.set_ylabel('Loss', fontsize=12)

# Remove the last (empty) subplot
fig.delaxes(axs[-1])

plt.tight_layout()
plt.show()