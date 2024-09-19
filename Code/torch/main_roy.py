import torch
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
#from tqdm import tqdm

from roy import royinv
from NND import Discriminator_paper, generator_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Simulation hyperarameters
n = m = 300
lambda_ = 0
g = 30
S = 1

# Neural net hyperparameters
n_discriminator = 60
#criterion = torch.nn.BCELoss()
#criterion = SamplesLoss("sinkhorn", p=1, blur=0.01) # Approximate Wasserstein-1 distance

# True parameter values
true_theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0], device=device)
lower_bounds = torch.tensor([1, 1, -.5, -1, 0, 0, -1, -1], device=device)
upper_bounds = torch.tensor([3, 3, 1.5, 1, 2, 2, 1, 1], device=device)
wide_lower_bounds = torch.tensor([-10, -10, -10, -10, 0, 0, -1, -1], device=device)
wide_upper_bounds = torch.tensor([10, 10, 10, 10, 10, 10, 1, 1], device=device)

### Loss plots
K = 30
param_ranges = [torch.linspace(start, end, K, device=device) for start, end in zip(wide_lower_bounds, wide_upper_bounds)]

""" cD_grid = torch.zeros((len(true_theta), K), device=device)
NND_grid = torch.zeros((len(true_theta), K), device=device)
OracleD_grid = torch.zeros((len(true_theta), K), device=device)
LL_grid = torch.zeros((len(true_theta), K), device=device)

cd_grid_diag = torch.zeros((2, K), device=device)
NND_grid_diag = torch.zeros((2, K), device=device)

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


# Latent variables for synthetic observations
Z = torch.rand(m, 4).to(device)
# Synthetic observations
X = royinv(Z, true_theta)

for i in range(len(true_theta)):
    for k in range(K):
        print(f'Parameter {i+1}/{len(true_theta)}, iteration {k+1}/{K}')
        theta = true_theta.clone()
        theta[i] = param_ranges[i][k]  
        #cD_grid[i, k] = loss(X, royinv(Z, th, smooth=lambda_))
        NND_grid[i, k] = generator_loss(X, royinv(Z, theta), Discriminator_paper, criterion, n_discriminator, g)
        #OracleD_grid[i, k] = OracleD(X, royinv(Z, th), theta, th)
        #LL_grid[i, k] = -np.mean(logroypdf(X, th)) / 2

# Plotting

NND_grid = NND_grid.detach().cpu().numpy()
true_theta = true_theta.cpu().numpy()
param_ranges = [param_range.cpu().numpy() for param_range in param_ranges]
lower_bounds = lower_bounds.cpu().numpy()
upper_bounds = upper_bounds.cpu().numpy()
wide_lower_bounds = wide_lower_bounds.cpu().numpy()
wide_upper_bounds = wide_upper_bounds.cpu().numpy()


fig, axs = plt.subplots(4, 2)
axs = axs.flatten()

for i in range(len(true_theta)):
    ax = axs[i]

    ax.plot(param_ranges[i], NND_grid[i, :], linewidth=1.5, color='blue', label='NN')
    #ax.plot(param_grid[i, :], cD_grid[i, :], linewidth=1.5, color='red', label='Logistic')

    ax.axvline(x=true_theta[i], color='r', linestyle='--', label=f'True {param_names[i]}')

    #max = torch.max([torch.max(NND_grid[i, :]), torch.max(cD_grid[i, :])])
    #min = torch.min([torch.min(NND_grid[i, :]), torch.min(cD_grid[i, :])])
    ax.set_xlim(wide_lower_bounds[i], wide_upper_bounds[i])
    #ax.set_ylim(min, max)

    ax.legend(loc='best', frameon=False)

plt.tight_layout()
#plt.show()
plt.savefig('./simres/loss_plots.png')
#np.savez('./simres/loss_plots.npz', param_grid=param_grid, cD_grid=cD_grid, NND_grid=NND_grid) """

### Diagonal loss plots

Z = torch.rand(m, 4).to(device)
X = royinv(Z, true_theta)

n_diags = 4
cd_grid_diag = torch.zeros((n_diags, K), device=device)
NND_grid_diag = torch.zeros((n_diags, K), device=device)

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

criterion = torch.nn.BCELoss()

for k in range(K):
    theta = true_theta.clone()
    theta[0] = param_ranges[0][k]
    theta[1] = param_ranges[1][k]
    NND_grid_diag[0, k] = generator_loss(X, royinv(Z, theta), Discriminator_paper, criterion, n_discriminator, g)
    #cD_grid_diag[0, k] = logistic_loss_2(X, royinv(Z, theta))[0]

for k in range(K):
    theta = true_theta.clone()
    theta[0] = param_ranges[0][k]
    theta[1] = -param_ranges[1][-k-1]
    NND_grid_diag[1, k] = generator_loss(X, royinv(Z, theta), Discriminator_paper, criterion, n_discriminator, g)
    #cD_grid_diag[1, k] = logistic_loss_2(X, royinv(Z, theta))[0]

criterion = SamplesLoss("sinkhorn", p=1, blur=0.01) # Approximately Wasserstein-1 distance

for k in range(K):
    theta = true_theta.clone()
    theta[0] = param_ranges[0][k]
    theta[1] = -param_ranges[1][-k-1]
    NND_grid_diag[2, k] = generator_loss(X, royinv(Z, theta), Discriminator_paper, criterion, n_discriminator, g)
    #cD_grid_diag[1, k] = logistic_loss_2(X, royinv(Z, theta))[0]

for k in range(K):
    theta = true_theta.clone()
    theta[0] = param_ranges[0][k]
    theta[1] = -param_ranges[1][-k-1]
    NND_grid_diag[3, k] = generator_loss(X, royinv(Z, theta), Discriminator_paper, criterion, n_discriminator, g)
    #cD_grid_diag[1, k] = logistic_loss_2(X, royinv(Z, theta))[0]

NND_grid_diag = NND_grid_diag.detach().cpu().numpy()
#cD_grid_diag = cD_grid_diag.detach().cpu().numpy()

fig, axs = plt.subplots(2, 2)
axs = axs.flatten()

for i in range(4):
    ax = axs[i]

    ax.plot(param_ranges[0], NND_grid_diag[i, :], linewidth=1.5, color='blue', label='NN')
    #ax.plot(param_ranges[0], cD_grid_diag[i, :], linewidth=1.5, color='red', label='Logistic')

    for j in range(2):
        ax.axvline(x=true_theta[j], color='black', linestyle='--', label=f'True {param_names[j]}')

    max = torch.max([torch.max(NND_grid_diag[i, :])]) + 0.1#, torch.max(cD_grid_diag[i, :])])
    min = torch.min([torch.min(NND_grid_diag[i, :])]) - 0.1#, torch.min(cD_grid_diag[i, :])])
    ax.set_xlim(wide_lower_bounds[0], wide_upper_bounds[0])
    ax.set_ylim(min, max)

    ax.legend(loc='best', frameon=False)

plt.tight_layout()
plt.savefig('./simres/diagonal_loss_plots.png')