"""Estimation simulation with Wasserstein"""
import torch
from torch.utils.data import RandomSampler
from geomloss import SamplesLoss
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm

from roy import royinv, soft_royinv, royinv_sp
from other_discriminators import logistic_loss3
#from NND import Discriminator_paper, generator_loss

class Generator(torch.nn.Module):
    def __init__(self, intial_guess, lambda_=0):
        super(Generator, self).__init__()
        self.theta = torch.nn.Parameter(intial_guess)
        self.lambda_ = lambda_

    def forward(self, noise):
        return royinv(noise, self.theta, self.lambda_)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    #raise Exception("No GPU available")
print(device)

# Simulation hyperarameters
n = m = 300
lambda_ = torch.tensor(0.3, device=device)
g = 1
S = 40

# Neural net hyperparameters
#n_discriminator = 1000
n_generator = 6000
#criterion = torch.nn.BCELoss()
wasserstein = SamplesLoss("sinkhorn", p=1, blur=0.01) # Approximately Wasserstein-p distance

# True parameter values and bounds
true_theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0], device=device)
lower_bounds = torch.tensor([1, 1, -.5, -1, 0, 0, -1, -1], device=device)
upper_bounds = torch.tensor([3, 3, 1.5, 1, 2, 2, 1, 1], device=device)
wide_lower_bounds = torch.tensor([-10, -10, -10, -10, 0, 0, -1, -1], device=device)
wide_upper_bounds = torch.tensor([10, 10, 10, 10, 10, 10, 1, 1], device=device)

# Observed data and latent noise
Z = torch.rand(m, 4).to(device)
X = royinv(Z, true_theta, lambda_=lambda_).detach()

#Perturb the true parameter values
theta_0 = true_theta + torch.randn(8, device=device) * 0.1
theta_0 = torch.clamp(theta_0, lower_bounds, upper_bounds)
print(f"Inital guess: {theta_0}")

# Pre-estimate with logistic regression
Z_sp = Z.cpu().numpy()
X_sp = royinv_sp(Z_sp, true_theta.cpu().numpy())
AdvL0 = opt.minimize(lambda theta : logistic_loss3(X_sp, royinv_sp(Z_sp, theta))[0],
                    x0 = theta_0.cpu().numpy(),
                    method='Nelder-Mead',
                    bounds = list(zip(lower_bounds.cpu().numpy(), upper_bounds.cpu().numpy())),
                    options={'return_all' : True, 'disp' : True, 'adaptive' : True})
#AdvL = torch.tensor(AdvL.x, device=device)
AdvL0 = torch.tensor(AdvL0.x, device=device, dtype=true_theta.dtype)
print(f"Pre-estimation: {AdvL0}")


""" #Pre-estimation with the logistic discriminator
generator = Generator(theta_0, lambda_).to(device)
optimizerG = torch.optim.Adam(generator.parameters())

for i in range(n_generator):
    optimizerG.zero_grad()
    fake_samples = generator.forward(Z.detach())
    generator_loss = logistic_loss2(X.detach(), fake_samples)[0]
    generator_loss.backward()
    optimizerG.step()

AdvL = generator.theta.detach()
print(f"Initial guess: {theta_0}")
print(f"Pre-estimation: {AdvL}")
 """

# Initialize tensors to store parameter values and losses
all_params = torch.empty(S, 8, n_generator, device=device)
all_losses = torch.empty(S, n_generator, device=device)

# Simulation loop
for s in range(S):
    try:
        # Draw bootstrap samples
        biU = torch.tensor(list(RandomSampler(Z, replacement=True, num_samples=m)), device=device)
        biX = torch.tensor(list(RandomSampler(X, replacement=True, num_samples=n)), device=device)   

        bU = Z[biU]
        bX = X[biX]
        #print(bU)
        #print(bX)
        """"
        bU_sp = bU.cpu().numpy()
        bX_sp = bX.cpu().numpy()
        # Pre-estimate with logistic regression
        AdvL = opt.minimize(lambda theta : logistic_loss3(bX_sp, royinv_sp(bU_sp, theta))[0],
                            x0 = AdvL.cpu().numpy(),
                            method='Nelder-Mead',
                            bounds = list(zip(lower_bounds.cpu().numpy(), upper_bounds.cpu().numpy())),
                            options={'return_all' : True, 'disp' : True, 'adaptive' : True})
        AdvL = torch.tensor(AdvL.x, device=device, dtype=true_theta.dtype)
        print(f"Pre-estimatin in repetition {s}: {AdvL}")    
        """

        # Initial guess uniform within the bounds
        #intial_guess = wide_lower_bounds + (wide_upper_bounds - wide_lower_bounds) * torch.rand(8, device=device)
        #intial_guess = torch.tensor([5., -5., -5., 5., 5., 5., -.5, -.5], device=device)
        #intial_guess = torch.tensor([2.5, 0.5, 0, -0.5, 0.5, 1.5, -0.9, 0.9], device=device)
        initial_guess = AdvL0.clone().detach()
        lambda_ = lambda_.clone().detach()
        #print(f"Initial guess in repetition {s}: {initial_guess}")
        generator = Generator(initial_guess, lambda_).to(device)
        optimizerG = torch.optim.Adam(generator.parameters(), lr=0.0002)

        for i in tqdm(range(n_generator)):
            try:
                Z = torch.rand(m, 4).to(device)
                X = royinv(Z, true_theta, lambda_=lambda_).detach()
                optimizerG.zero_grad()
                fake_samples = generator.forward(Z.detach())
                generator_loss = wasserstein(X.detach(), fake_samples)
                all_params[s, :, i] = generator.theta.detach()
                all_losses[s, i] = generator_loss.item()
                generator_loss.backward()
                #print(generator.theta.grad)
                optimizerG.step()
            except Exception as e:
                print(f"Error in repetition {s} in generator step {i}: {e}")

    except Exception as e:
        print(f"Error in repetition {s}: {e}")
        continue

# Save results
torch.save(all_params, 'simres/all_params.pt')
torch.save(all_losses, 'simres/all_losses.pt')

""" # Plot parameters
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
 """