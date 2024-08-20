# Training loops and functions for estimation

import numpy as np
import estimagic as em

import roy_helper_functions as rhf
import NND_sp as nnd

def train_kpm(generator_function, true_theta, num_hidden=10, g=10, num_samples=300, num_repetitions=10):
    """Training loop that is close to the original code"""

    results = [[] for _ in range(num_repetitions)]
    theta_initial_guess = np.ones_like(true_theta) * 0.5
    theta_initial_guess[6] = 0
    theta_initial_guess[8] = 0.9

    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, 0, -1, 0.9])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 0, 1, 0.9])
    
    for rep in range(num_repetitions):
        u = np.random.rand(num_samples, 4)
        true_values = generator_function(u, true_theta, 0, num_samples)

        def loss_function(theta):
            fake_values = generator_function(u, theta, 0, num_samples)
            return nnd.NDD_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)
        
        result = em.minimize(criterion=loss_function,
                             params=theta_initial_guess,
                             algorithm="scipy_neldermead",
                             lower_bounds=lower_bounds,
                             upper_bounds=upper_bounds  
                             )
        results[rep] = result

    return results
        

