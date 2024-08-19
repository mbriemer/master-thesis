# Training loops and functions for estimation

import torch
from scipy.optimize import minimize
from tqdm import tqdm

import roy_helper_functions as rhf
import NND as nnd

def train_kpm(generator_function, DiscriminatorClass, criterion, theta, num_samples=300, num_repetitions=10, n_discriminator=1, g=10):
    """Training loop that is close to the original code"""

    results = [[] for _ in range(num_repetitions)]

    theta_initial_guess = torch.ones_like(theta, requires_grad=True) * 0.5
    bounds = [(1, 3),
             (1, 3),
             (-0.5, 1.5),
             (-1, 1), 
             (0, 2),
             (0, 2),
             (0, 0), # rho_t is fixed in simple case
             (-1, 1),
             (0.9, 0.9)] # beta is fixed
    
    optimizerD = torch.optim.Adam(DiscriminatorClass().parameters())

    for rep in tqdm(range(num_repetitions)):
        u = torch.rand(num_samples, 4)
        true_values = rhf.royinv(u, theta, 0, num_samples)

        def loss_function(theta):
            fake_values = generator_function(u, theta, num_samples)
            return nnd.NDD_loss(true_values, fake_values, DiscriminatorClass, optimizerD, criterion, n_discriminator, g)
        
        result = minimize(loss_function, theta_initial_guess, method='Nelder-Mead', bounds=bounds, options={'return_all': True})
        results[rep] = result

    return results
        

