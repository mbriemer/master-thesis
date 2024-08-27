# Training loops and functions for estimation

import numpy as np
#import estimagic as em
import scipy.optimize as opt
#from tqdm import tqdm
import multiprocessing as mp

from roy_helper_functions import royinv
from NND_sp import generator_loss
from other_discriminators import OracleD as oracle

def callback(xk):
    print(f"Current solution: {xk}")

def callback(xk):
    print(f"Current solution: {xk}")

def loss_function(theta, u, true_values, num_hidden, g):
    print(f"Current theta: {theta}")
    fake_values = royinv(u, theta, 0, len(u))
    return -generator_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)

def run_single_repetition(args):
    rep, generator_function, true_theta, num_hidden, g, num_samples, theta_initial_guess, sp_bounds = args
    
    print(f"Starting repetition {rep}")
    u = np.random.rand(num_samples, 4)
    true_values = generator_function(u, true_theta, 0, num_samples)
    
    result = opt.minimize(
        fun=lambda theta: loss_function(theta, u, true_values, num_hidden, g),
        x0=theta_initial_guess,
        method='Powell',
        bounds=sp_bounds,
        callback=callback,
        options={'return_all': True, 'disp': True, 'maxiter': 10}
    )
    
    return result

def train_kpm_parallel(generator_function, true_theta, num_hidden=10, g=10, num_samples=300, num_repetitions=10):
    theta_initial_guess = [2, 2, 0, 0, 1, 1, 0, 0, 0.9]
    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, 0, 0.9])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 0, 0.9])
    sp_bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    print(sp_bounds)

    # Prepare arguments for each repetition
    args_list = [(rep, generator_function, true_theta, num_hidden, g, num_samples, theta_initial_guess, sp_bounds) 
                 for rep in range(num_repetitions)]

    # Use all available CPU cores
    num_processes = mp.cpu_count()
    print(f"Number of cores: {num_processes}")

    # Run parallel computations
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_single_repetition, args_list)

    return results

def train_kpm(generator_function, true_theta, num_hidden=10, g=10, num_samples=300, num_repetitions=10):
    """Training loop that is close to the original code"""

    results = [[] for _ in range(num_repetitions)]
    results_oracle = [[] for _ in range(num_repetitions)]
    theta_initial_guess = [2, 2, 0, 0, 1, 1, 0, 0, 0.9]

    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, 0, 0.9])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 0, 0.9])
    sp_bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    print(sp_bounds)
    
    for rep in range(num_repetitions):
        print(f"Starting repetition {rep}")
        u = np.random.rand(num_samples, 4)
        true_values = generator_function(u, true_theta, 0, num_samples)
        '''
        def oracle_loss_function(theta):
            fake_values = royinv(u, theta, 0, num_samples)
            return oracle(true_values, fake_values, true_theta, theta)
        '''

        def loss_function(theta):
            print(f"Current theta: {theta}")
            fake_values = royinv(u, theta, 0, num_samples)
            return - generator_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)
        '''
        result = em.minimize(criterion=loss_function,
                             params=theta_initial_guess,
                             algorithm="scipy_neldermead",
                             #lower_bounds=lower_bounds,
                             #upper_bounds=upper_bounds,
                             logging="my_magic_log.db" 
                             )
        '''
        '''
        result_oracle = opt.minimize(fun = oracle_loss_function,
                                     x0 = theta_initial_guess, 
                                     method='Powell', 
                                     bounds=sp_bounds,
                                     callback=callback,
                                     options={'return_all' : True, 'disp' : True, 'maxiter' : 1000})
        
        print(f"Oracle result: {result_oracle.x}")
        '''

        result = opt.minimize(fun = loss_function, 
                              x0 = theta_initial_guess, 
                              method='Powell', 
                              bounds=sp_bounds,
                              callback=callback,
                              options={'return_all' : True, 'disp' : True, 'maxiter' : 10})

        results[rep] = result
        #results_oracle[rep] = result_oracle

    return results#, results_oracle
