"""Estimation simulation with Wasserstein"""
import torch
from geomloss import SamplesLoss
import matplotlib.pyplot as plt
#from tqdm import tqdm

from roy import royinv
#from NND import Discriminator_paper, generator_loss

class Generator(torch.nn.Module):
    def __init__(self, intial_guess, lambda_=0):
        super(Generator, self).__init__()
        self.theta = torch.nn.Parameter(intial_guess)
        self.lambda_ = lambda_

    def forward(self, noise):
        return royinv(noise, self.theta, lambda_=self.lambda_)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    #raise Exception("No GPU available")
print(device)

# Simulation hyperarameters
n = m = 300
lambda_ = 0
g = 1
S = 10

# Neural net hyperparameters
n_discriminator = 1000
n_generator = 5000
criterion = torch.nn.BCELoss()
wasserstein = SamplesLoss("sinkhorn", p=1, blur=0.01) # Approximately Wasserstein-p distance

# True parameter values and bounds
true_theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0], device=device)
lower_bounds = torch.tensor([1, 1, -.5, -1, 0, 0, -1, -1], device=device)
upper_bounds = torch.tensor([3, 3, 1.5, 1, 2, 2, 1, 1], device=device)
wide_lower_bounds = torch.tensor([-10, -10, -10, -10, 0, 0, -1, -1], device=device)
wide_upper_bounds = torch.tensor([10, 10, 10, 10, 10, 10, 1, 1], device=device)

# Initialize tensors to store parameter values and losses
all_params = torch.empty(S, 8, n_generator, device=device)
all_losses = torch.empty(S, n_generator, device=device)


# Simulation loop
for s in range(S):
    Z = torch.rand(m, 4).to(device)
    X = royinv(Z, true_theta).detach()

    # Initial guess uniform within the bounds
    intial_guess = wide_lower_bounds + (wide_upper_bounds - wide_lower_bounds) * torch.rand(8, device=device)
    #intial_guess = torch.tensor([5., -5., -5., 5., 5., 5., -.5, -.5], device=device)
    #intial_guess = torch.tensor([2.5, 0.5, 0, -0.5, 0.5, 1.5, -0.9, 0.9], device=device)
    generator = Generator(intial_guess, lambda_).to(device)
    optimizerG = torch.optim.Adam(generator.parameters())
   
    for i in range(n_generator):
        optimizerG.zero_grad()
        fake_samples = generator.forward(Z.detach())  # Detach Z instead of fake_samples
        generator_loss = wasserstein(X.detach(), fake_samples)
        all_params[s, :, i] = generator.theta.detach()
        all_losses[s, i] = generator_loss.item()  # Use .item() instead of .detach()
        generator_loss.backward()
        #print(generator.theta.grad)
        optimizerG.step()

# Save results
torch.save(all_params, 'simres/wide_all_params.pt')
torch.save(all_losses, 'simres/wide_all_losses.pt')

# Plot parameters
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

all_params = all_params.detach().cpu().numpy()
all_losses = all_losses.detach().cpu().numpy()
true_theta = true_theta.cpu().numpy()

fig, axs = plt.subplots(4, 2)
ax = axs.flatten()

for i in range(8):
    for s in range(S):
        ax[i].plot(all_params[s, i, :], color='blue', alpha=0.7, linewidth=0.7)
    #ax[i].set_title(param_names[i])
    ax[i].hlines(true_theta[i], 0, n_generator, color='black', linestyle='--', linewidth=0.7, label=f'True {param_names[i]}')

    ax[i].legend(loc='best', frameon=False)

plt.tight_layout()
#plt.show()
plt.savefig('./simres/wide_all_params.png')
