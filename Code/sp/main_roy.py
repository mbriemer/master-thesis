# Implementation of the Roy model simulation that is very close to the original code
#import estimagic as em
import numpy as np
np.random.seed(83209)


#from NND import Discriminator_paper
from roy_helper_functions import royinv
from estimation import train_kpm_parallel #train_kpm
from plotting_functions import plot_theta_values, plot_loss_function

if __name__ == "__main__":
    # Parameters
    true_theta = [1.8, 2, 0.5, 0, 1, 1, 0.5]#, 0, 0.9]
    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1])#, 0, 0.9])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1])#, 0, 0.9])

    num_samples = 300
    num_repetitions = 4
    #n_discriminator = 100

    results = train_kpm_parallel(royinv, true_theta, num_hidden=10, g=10, num_samples=num_samples, num_repetitions=num_repetitions)

    # Process and save results
    for rep, initial_value, result, error in results:
        if error is None:
            np.savez(f'simres/estimation_result_{rep}.npz', 
                     initial_value=initial_value, 
                     final_value=result.x, 
                     all_values=result.allvecs,
                     fun_values=result.allvecs)
            print(f"Repetition {rep} completed successfully.")
        else:
            print(f"Error in repetition {rep}: {error}")
    '''
    print("NND")
    #print(results)
    plot_theta_values(results, true_theta, lower_bounds, upper_bounds)
    plot_loss_function(results)

    with open("results.txt", "w") as f:
        f.write(str(results))
    '''
    '''
    plt_loss = em.criterion_plot(results)
    plt_loss.show

    plt_params = em.params_plot(results)
    plt_params.show()

    '''