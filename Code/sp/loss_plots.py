import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm


#from roy_helper_functions import royinv, logroypdf_b
from roy import logroypdf, royinv
#from translation import logroypdf
from NND_sp import generator_loss as NND
from other_discriminators import OracleD

#from custom_functions import royrnd, royinv, logroypdf, NND, OracleD, conv

#def linspacev(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
#    return a + np.linspace(0, 1, n) * (b - a)[:, np.newaxis]

def linspacev(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    return a[:, np.newaxis] + np.linspace(0, 1, n) * (b - a)[:, np.newaxis]

def loss(X, Y):
    return np.mean((X - Y)**2)

K = 30
param_grid = linspacev(
    np.array([1.4, 1.4, 0.1, -0.4, 0.8, 0.7, -0.2]),
    np.array([2.3, 2.5, 0.9, 0.4, 1.3, 1.3, 0.9]),
    K
)

#param_grid = linspacev(
#    np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]),
#    np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
#    K
#)

#lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1])#, 0, 0.9])
#upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1])#, 0, 0.9])
#param_grid = np.clip(param_grid, lower_bounds, upper_bounds)

cD_grid = np.zeros_like(param_grid)
NND_grid = np.zeros_like(param_grid)
OracleD_grid = np.zeros_like(param_grid)
LL_grid = np.zeros_like(param_grid)

# Assume these variables are defined elsewhere
theta = [1.8, 2, 0.5, 0, 1, 1, 0.5]
#theta2: np.ndarray
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
        r'$\rho_s$'
    ]

#X = royrnd(theta, n)  # Real observations
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
        #LL_grid_finite[i, k] = np.minimum(LL_grid[i, k], -1)



# Plotting
fig, axs = plt.subplots(4, 2)  # Increased height
axs = axs.flatten()  # Flatten the 2D array of axes to make indexing easier

for i in range(len(theta)):
    ax = axs[i]
    axL = ax.twinx()

    #max_diff = np.absolute(OracleD_grid[i, :].max() - LL_grid[i, np.isfinite(LL_grid[i,:])].max())
    
    # Plot all curves on the same y-axis
    ax.plot(param_grid[i, :], NND_grid[i, :], linewidth=1.5, color='blue', label='$\\mathbf{M}_\\theta(\\hat{D}_\\theta)$')
    ax.plot(param_grid[i, :], OracleD_grid[i, :], linewidth=1.5, color='green', label='$\\mathbf{M}_\\theta(D_\\theta)$')
    axL.plot(param_grid[i, :], (LL_grid[i, :] / 2), linewidth=1.5, color='red', label='$\\mathbf{L}_\\theta$')
    
    #Plot points
    #ax.scatter(param_grid[i, :], NND_grid[i, :], s=20, color='blue', label='$\\mathbf{M}_\\theta(\\hat{D}_\\theta)$')
    #ax.scatter(param_grid[i, :], OracleD_grid[i, :], s=5, color='green', label='$\\mathbf{M}_\\theta(D_\\theta)$')
    #ax.scatter(param_grid[i, :], (LL_grid[i, :] / 2)-2.3, s=5, color='red', label='$\\mathbf{L}_\\theta$')

    # Add vertical line for true parameter value
    ax.axvline(x=theta[i], color='r', linestyle='--', label=f'True {param_names[i]}')
    
    # Set axes limits
    ax.set_xlim(param_grid[i, 0], param_grid[i, -1])
    ax.set_ylim(-1.4, -1.25)
    #y_min = np.minimum(NND_grid[i, :].min(), OracleD_grid[i, :].min())
    #y_max = np.maximum(NND_grid[i, :].max(), OracleD_grid[i, :].max())
    #ax.set_ylim(y_min, y_max)
    axL.set_ylim((LL_grid[i, :].min())/2 - 0.01, (LL_grid[i, np.isfinite(LL_grid[i,:])].max())/2 + 0.01)
    #axL.set_ylim(0.2, 0)
    #y_max = np.maximum(OracleD_grid[i, :].max(), (LL_grid[i, np.isfinite(LL_grid[i,:])].max() - max_diff))
    #y_min = np.minimum(OracleD_grid[i, :].min(), LL_grid[i, :].min())
    #y_min = np.minimum(OracleD_grid[i, np.argwhere(np.isneginf(OracleD_grid[i, :]) == False)].min(), 
    #                   LL_grid[i, np.argwhere(np.isneginf(LL_grid[i, :]) == False)].min())
    #ax.set_ylim(y_max, y_min)
    
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


