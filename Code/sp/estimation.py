# Training loops and functions for estimation

import numpy as np
import scipy.optimize as opt
import multiprocessing as mp

from roy import royinv, logroypdf, perturb, perturb_uniform, roysupp
from NND_sp import generator_loss
from other_discriminators import OracleD, logistic_loss

def create_optimization_function(u, true_values, true_theta):
    def objective_function(theta):
        return loss_function(theta, u, true_values, 'MLE', None)
    
    iteration = [0]  # Use a list to store mutable state
    losses = []
    gradients = []

    def callback(xk):
        loss = objective_function(xk)
        grad = opt.approx_fprime(xk, objective_function)
        
        losses.append(loss)
        gradients.append(grad)
        
        print(f"Iteration {iteration[0]}:")
        print(f"Current theta: {xk}")
        print(f"Current loss: {loss}")
        print(f"Current gradient: {grad}")
        print("--------------------")
        
        iteration[0] += 1

    return objective_function, callback, losses, gradients

def callback(xk):
    print(f"Current theta: {xk}")


def loss_function(theta, u, true_values, loss, loss_options):
    fake_values = royinv(u, theta)

    if loss == 'MLE':
        return - np.sum(logroypdf(true_values, theta))
    
    elif loss == 'oracle':
        return OracleD(true_values, fake_values, loss_options['true_theta'], theta)

    elif loss == 'logistic':
        loss, _ = logistic_loss(true_values, fake_values)
        return loss
    
    elif loss == 'NN':
        return generator_loss(true_values, fake_values, num_hidden=loss_options['num_hidden'], num_models=loss_options['num_models'])
    
def run_single_repetition(args):
    rep, seed, generator_function, true_theta, num_hidden, g, num_samples = args
    print(f"Starting repetition {rep}")
    
    #try:
    rng = np.random.default_rng(seed)
    u = rng.random((num_samples, 4))
    true_values = generator_function(u, true_theta)
    
    if true_values is None:
        raise ValueError("generator_function returned None")
    
    #theta_initial_guess = rng.uniform(low=[1, 1, -0.5, -1, 0, 0, -0.99],
    #                                  high=[3, 3, 1.5, 1, 2, 2, 0.99])
    #print(f"Initial guess for repetition {rep}: {theta_initial_guess}")
    lower_bounds = [1, 1, -0.5, -1, 0, 0, -0.99]#, 0, 0.9]
    upper_bounds = [3, 3, 1.5, 1, 2, 2, 0.99]#, 0.9]
    true_theta = [1.8, 2,  0.5, 0,  1, 1, 0.5]#, 0, 0.9]
    #theta_initial_guess = [1, 2.5,  0.5, 1, 0.5,1.5, 0.7]#, 0, 0.9]
    #theta_initial_guess = [2, 1.5, 1, 1, 0.5, 0.5, -0.5]
    #theta_initial_guess = perturb(true_values, true_theta, lower_bounds, upper_bounds, rng)
    theta_initial_guess = perturb_uniform(true_values, lower_bounds, upper_bounds, rng)

    sp_bounds = list(zip(lower_bounds, upper_bounds))
    obj_func, callback, losses, gradients = create_optimization_function(u, true_values, true_theta)

    # MLE
    print(f"Starting MLE in repetition {rep} with initial guess {theta_initial_guess}")       
    result_MLE = opt.minimize(
        fun=obj_func,#lambda theta: loss_function(theta, u, true_values, 'MLE', None),
        x0=theta_initial_guess,
        method='Nelder-Mead',
        bounds=sp_bounds,
        constraints=opt.NonlinearConstraint(roysupp, -np.inf, 0),
        #callback=callback,
        options={'disp': True, 'return_all': True, 'adaptive': False}#, 'maxiter': 600}
    )

    if not result_MLE.success:
        print(f"Optimization failed for repetition {rep} with MLE: {result_MLE.message}")
    
     # Adversarial estimator with the oracle discriminator
    print(f"Starting oracle in repetition {rep} with initial guess {result_MLE.x}")
    result_oracle = opt.minimize(
        fun=lambda theta: loss_function(theta, u, true_values, 'oracle', {'true_theta': true_theta}),
        x0=result_MLE.x,
        method='Nelder-Mead',
        bounds=sp_bounds,
        constraints=opt.NonlinearConstraint(roysupp, -np.inf, 0),
        #callback=callback,
        options={'disp': True, 'return_all': True, 'adaptive': False}#, 'maxiter': }
    ) 

    if not result_oracle.success:
        print(f"Optimization failed for repetition {rep} with oracle discriminator: {result_oracle.message}")
    
    
    """ # Adversarial estimator with a logistic discriminator
    print(f"Starting logistic in repetition {rep}")
    result_logistic = opt.minimize(
        fun=lambda theta: loss_function(theta, u, true_values, 'logistic', None),
        x0=result_oracle.x,
        method='Nelder-Mead',
        bounds=sp_bounds,
        #callback=callback,
        options={'disp': True, 'return_all': True, 'adaptive': True}#, 'maxiter': maxiter}
    )

    if not result_logistic.success:
        print(f"Optimization failed for repetition {rep} with logistic discriminator: {result_logistic.message}") """
    
    
    # Adversarial estimator with a neural network discriminator
    print(f"Starting NN in repetition {rep} with initial guess {theta_initial_guess}")
    result_NN = opt.minimize(
        fun=lambda theta: loss_function(theta, u, true_values, 'NN', {'num_hidden': num_hidden, 'num_models': g}),
        x0=theta_initial_guess,
        method='Nelder-Mead',
        bounds=sp_bounds,
        #callback=callback,
        options={'disp': True, 'return_all': True, 'adaptive': False}
    )
    
    if not result_NN.success:
        print(f"Optimization failed for repetition {rep} with NN discriminator: {result_NN.message}")

    #result = {result_MLE, result_oracle, result_logistic, result_NN}
    return rep, theta_initial_guess, result_NN, None
    #except Exception as e:
    #    print(f"Error in repetition {rep}: {str(e)}")
    #    return rep, None, None, str(e)
    
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