# Training loops and functions for estimation

import numpy as np
import estimagic as em
import scipy.optimize as opt
from tqdm import tqdm

import roy_helper_functions as rhf
import NND_sp as nnd

def callback(xk):
    print(f"Current solution: {xk}")

def train_kpm(generator_function, true_theta, num_hidden=10, g=10, num_samples=300, num_repetitions=10):
    """Training loop that is close to the original code"""

    results = [[] for _ in range(num_repetitions)]
    theta_initial_guess = [2, 2, 0, 0, 1, 1, 0, 0, 0.9]

    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, -0.1, 0.89])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 0.1, 0.91])
    sp_bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    print(sp_bounds)
    
    for rep in range(num_repetitions):
        print(f"Starting repetition {rep}")
        u = np.random.rand(num_samples, 4)
        true_values = generator_function(u, true_theta, 0, num_samples)

        def loss_function(theta):
            fake_values = rhf.royinv(u, theta, 0, num_samples)
            return - nnd.generator_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)
        '''
        result = em.minimize(criterion=loss_function,
                             params=theta_initial_guess,
                             algorithm="scipy_neldermead",
                             #lower_bounds=lower_bounds,
                             #upper_bounds=upper_bounds,
                             logging="my_magic_log.db" 
                             )
        '''
        
        result = opt.minimize(fun = loss_function, 
                              x0 = theta_initial_guess, 
                              method='Nelder-Mead', 
                              #bounds=sp_bounds,
                              callback=callback,
                              options={'disp' : True})

        results[rep] = result

    return results
        

