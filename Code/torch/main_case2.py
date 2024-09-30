"""Estimation simulation with Wasserstein"""
import torch
from torch.utils.data import RandomSampler
from geomloss import SamplesLoss

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
#from tqdm import tqdm

from roy import royinv, soft_royinv, royinv_sp
from other_discriminators import logistic_loss3
from NND import Discriminator_paper, generator_loss

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    #device = torch.device("cpu")
    raise Exception("No GPU available")
print(device)

wasserstein1 = SamplesLoss("sinkhorn", p=1, blur=0.01) # Approximately Wasserstein-p distance
wasserstein2 = SamplesLoss("sinkhorn", p=2, blur=0.01) # Approximately Wasserstein-p distance

class Generator(torch.nn.Module):
    def __init__(self, intial_guess, lambda_=0):
        super(Generator, self).__init__()
        self.theta = torch.nn.Parameter(intial_guess)
        self.lambda_ = lambda_

    def forward(self, noise):
        return royinv(noise, self.theta, self.lambda_)

def wasserstein_loss_1(X, U, theta, lambda_, device=device):
    X = torch.tensor(X, device=device, dtype=torch.float32)
    U = torch.tensor(U, device=device, dtype=torch.float32)
    theta = torch.tensor(theta, device=device, dtype=torch.float32) 
    lambda_ = torch.tensor(lambda_, device=device, dtype=torch.float32)
    loss = wasserstein1(X, royinv(U, theta, lambda_))
    loss = loss.cpu().numpy()
    return loss

def wasserstein_loss_2(X, U, theta, lambda_, device=device):
    X = torch.tensor(X, device=device, dtype=torch.float32)
    U = torch.tensor(U, device=device, dtype=torch.float32)
    theta = torch.tensor(theta, device=device, dtype=torch.float32) 
    lambda_ = torch.tensor(lambda_, device=device, dtype=torch.float32)
    loss = wasserstein2(X, royinv(U, theta, lambda_))
    loss = loss.cpu().numpy()
    return loss

def jensen_shannon_loss(X, U, theta, lambda_, device=device):
    X = torch.tensor(X, device=device, dtype=torch.float32).detach()
    U = torch.tensor(U, device=device, dtype=torch.float32).detach()
    theta = torch.tensor(theta, device=device, dtype=torch.float32).detach()
    lambda_ = torch.tensor(lambda_, device=device, dtype=torch.float32).detach()
    criterion = torch.nn.BCELoss()
    generator = Generator(theta, lambda_).to(device)
    discriminator = Discriminator_paper().to(device)
    optimizerD = torch.optim.Adam(discriminator.parameters())
    true_samples = X
    fake_samples = generator.forward(U).detach()
    for i in range(1000):
        optimizerD.zero_grad()
        true_output = discriminator.forward(true_samples)
        fake_output = discriminator.forward(fake_samples)
        lossD = criterion(true_output, torch.ones_like(true_output)) + criterion(fake_output, torch.zeros_like(fake_output))
        lossD.backward()
        optimizerD.step()

    lossD = lossD.detach().cpu().numpy()
    return lossD

# Simulation hyperarameters
n = m = 300
lambda_ = 0#torch.tensor(0.1, device=device)
g = 1
S = 10

# Neural net hyperparameters
#n_jscriminator = 1000
#n_generator = 1200
#criterion = torch.nn.BCELoss()

# True parameter values and bounds
true_theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0], device=device)
lower_bounds = torch.tensor([1, 1, -.5, -1, 0, 0, -1, -1], device=device)
upper_bounds = torch.tensor([3, 3, 1.5, 1, 2, 2, 1, 1], device=device)
wide_lower_bounds = torch.tensor([-10, -10, -10, -10, 0, 0, -1, -1], device=device)
wide_upper_bounds = torch.tensor([10, 10, 10, 10, 10, 10, 1, 1], device=device)

# Observed data and latent noise
Z = torch.rand(m, 4).to(device)
X = royinv(Z, true_theta).detach()

#Perturb the true parameter values
#theta_0 = true_theta + torch.randn(8, device=device) * 0.1
#theta_0 = torch.clamp(theta_0, lower_bounds, upper_bounds)
#print(f"Inital guess: {theta_0}")

""" # Pre-estimate with logistic regression
Z_sp = Z.cpu().numpy()
X_sp = royinv_sp(Z_sp, true_theta.cpu().numpy())
AdvL0 = opt.minimize(lambda theta : logistic_loss3(X_sp, royinv_sp(Z_sp, theta))[0],
                    x0 = theta_0.cpu().numpy(),
                    method='Nelder-Mead',
                    bounds = list(zip(lower_bounds.cpu().numpy(), upper_bounds.cpu().numpy())),
                    options={'return_all' : True, 'disp' : True, 'adaptive' : True})
print(f"Pre-estimation: {AdvL0.x}") """

# Simulation loop
for s in range(S):
    try:
        # Draw bootstrap samples
        """ biU = torch.tensor(list(RandomSampler(Z, replacement=True, num_samples=m)), device=device)
        biX = torch.tensor(list(RandomSampler(X, replacement=True, num_samples=n)), device=device)   

        bU = Z[biU]
        bX = X[biX]
        #print(bU)
        #print(bX)
        
        bU_sp = bU.cpu().numpy()
        bX_sp = bX.cpu().numpy()
        bX_sp = [bX_sp[:,0], bX_sp[:,1], bX_sp[:,2], bX_sp[:,3]] """

        # Pre-estimate with logistic regression
        """ AdvL = opt.minimize(lambda theta : logistic_loss3(bX_sp, royinv_sp(bU_sp, theta))[0],
                            x0 = AdvL0.x,
                            method='Nelder-Mead',
                            bounds = list(zip(lower_bounds.cpu().numpy(), upper_bounds.cpu().numpy())),
                            options={'return_all' : True, 'disp' : True, 'adaptive' : True})
        print(f"Pre-estimation in repetition {s}: {AdvL.x}") """ 
        theta0 = np.random.uniform(low=wide_lower_bounds.cpu().numpy(), high=wide_upper_bounds.cpu().numpy())

        AdvN = opt.minimize(lambda theta : jensen_shannon_loss(X, Z, theta, lambda_),
                            x0 = theta0,
                            method='Nelder-Mead',
                            bounds = list(zip(wide_lower_bounds.cpu().numpy(), wide_upper_bounds.cpu().numpy())),
                            options={'return_all' : True, 'disp' : True, 'adaptive' : True})
        print(f"Estimation in repetition {s}: {AdvN.x}")

        offset = 190
        id = offset + s

        np.savez(f"simres/estimation_js_wide_uniform_{id}.npz", AdvN)

    except Exception as e:
        print(f"Error in repetition {s}: {e}")
        continue