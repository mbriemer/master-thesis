# Training loops and functions for estimation

import numpy as np
import scipy.optimize as opt
import multiprocessing as mp

from roy_helper_functions import royinv
from NND_sp import generator_loss

def callback(xk):
    print(f"Current theta: {xk}")

def loss_function(theta, u, true_values, num_hidden, g):
    #try:
    fake_values = royinv(u, theta, 0, len(u))
    #    if fake_values is None:
    #        print(f"royinv returned None for theta: {theta}")
    #         return np.inf
    loss = generator_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)
    #    if loss is None:
    #        print(f"generator_loss returned None for theta: {theta}")
    #        return np.inf
    #print(f"Loss for theta {theta}: {loss}")
    return -loss
    #except Exception as e:
    #    print(f"Error in loss_function for theta {theta}: {str(e)}")
    #    return np.inf
    
def run_single_repetition(args):
    rep, seed, generator_function, true_theta, num_hidden, g, num_samples = args
    print(f"Starting repetition {rep}")
    
    try:
        rng = np.random.default_rng(seed)
        u = rng.random((num_samples, 4))
        true_values = generator_function(u, true_theta, 0, num_samples)
        
        if true_values is None:
            raise ValueError("generator_function returned None")
        
        theta_initial_guess = rng.uniform(low=[1, 1, -0.5, -1, 0, 0, -0.99],
                                          high=[3, 3, 1.5, 1, 2, 2, 0.99])
        #print(f"Initial guess for repetition {rep}: {theta_initial_guess}")
        
        true_theta =         [1.8, 2,  0.5, 0,  1, 1, 0.5]#, 0, 0.9]
        #theta_initial_guess = [1.9, 2.1, 0.4, -0.1, 0.9, 1.1, 0.6]#, 0, 0.9]

        lower_bounds = [1, 1, -0.5, -1, 0, 0, -0.99]#, 0, 0.9]
        upper_bounds = [3, 3, 1.5, 1, 2, 2, 1, 0.99]#, 0.9]
        sp_bounds = list(zip(lower_bounds, upper_bounds))

        result = opt.minimize(
            fun=lambda theta: loss_function(theta, u, true_values, num_hidden, g),
            x0=theta_initial_guess,
            method='Nelder-Mead',
            bounds=sp_bounds,
            callback=callback,
            options={'disp': True, 'maxiter': 400, 'return_all': True, 'adaptive': True}
        )
        
        if not result.success:
            print(f"Optimization failed for repetition {rep}: {result.message}")
        
        return rep, theta_initial_guess, result, None
    except Exception as e:
        print(f"Error in repetition {rep}: {str(e)}")
        return rep, None, None, str(e)
    
def train_kpm_parallel(generator_function, true_theta, num_hidden=10, g=30, num_samples=300, num_repetitions=10):
    num_processes = mp.cpu_count()

    ss = np.random.SeedSequence()
    child_seeds = ss.spawn(num_repetitions)
    
    args_list = [(rep, seed, generator_function, true_theta, num_hidden, g, num_samples) 
                 for rep, seed in enumerate(child_seeds)]

    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in pool.imap_unordered(run_single_repetition, args_list):
            results.append(result)
            rep, initial_value, final_result, error = result
            print(result)
            if error is None:
                np.savez(f'simres/intermediate_result_{rep}.npz', 
                         initial_value=initial_value, 
                         final_value=final_result.x, 
                         all_values=final_result.allvecs,
                         fun_values=final_result.allvecs)
            else:
                with open(f'error_log_{rep}.txt', 'w') as f:
                    f.write(f"Error in repetition {rep}: {error}")

    #results.sort(key=lambda x: x[0])
    return results

'''
def train_kpm(generator_function, true_theta, num_hidden=10, g=10, num_samples=300, num_repetitions=10):
    """Training loop that is close to the original code"""

    results = [[] for _ in range(num_repetitions)]
    results_oracle = [[] for _ in range(num_repetitions)]
    theta_initial_guess = [2, 2, 0, 0, 1, 1, 0, 0, 0.9]

    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, 0, 0.9])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 0, 0.9])
    sp_bounds = list(zip(lower_bounds.tolist(), upper_bounds.tolist()))
    
    for rep in range(num_repetitions):
        print(f"Starting repetition {rep}")
        u = np.random.rand(num_samples, 4)
        true_values = generator_function(u, true_theta, 0, num_samples)

        #def oracle_loss_function(theta):
        #    fake_values = royinv(u, theta, 0, num_samples)
        #    return oracle(true_values, fake_values, true_theta, theta)


        def loss_function(theta):
            print(f"Current theta: {theta}")
            fake_values = royinv(u, theta, 0, num_samples)
            return - generator_loss(true_values, fake_values, num_hidden=num_hidden, num_models=g)
        
        result = em.minimize(criterion=loss_function,
                             params=theta_initial_guess,
                             algorithm="scipy_neldermead",
                             #lower_bounds=lower_bounds,
                             #upper_bounds=upper_bounds,
                             logging="my_magic_log.db" 
                             )

        result_oracle = opt.minimize(fun = oracle_loss_function,
                                     x0 = theta_initial_guess, 
                                     method='Powell', 
                                     bounds=sp_bounds,
                                     callback=callback,
                                     options={'return_all' : True, 'disp' : True, 'maxiter' : 1000})
        
        print(f"Oracle result: {result_oracle.x}")
        

        result = opt.minimize(fun = loss_function, 
                              x0 = theta_initial_guess, 
                              method='Powell', 
                              bounds=sp_bounds,
                              callback=callback,
                              options={'return_all' : True, 'disp' : True, 'maxiter' : 10})

        results[rep] = result
        #results_oracle[rep] = result_oracle

    return results#, results_oracle
'''
