# Implementation of the Roy model simulation that is very close to the original code
#import estimagic as em
import numpy as np
np.random.seed(83209)


#from NND import Discriminator_paper
from roy_helper_functions import royinv
from estimation import train_kpm, train_kpm_parallel
from plotting_functions import plot_theta_values, plot_loss_function

if __name__ == "__main__":

    # Parameters
    true_theta = [1.8, 2, 0.5, 0, 1, 1, 0.5, 0, 0.9]
    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1, -0.1, 0.89])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1, 0.1, 0.91])

    num_samples = 300
    num_repetitions = 10
    n_discriminator = 100

    results = train_kpm_parallel(royinv, true_theta, num_hidden=10, g=10, num_samples=num_samples, num_repetitions=num_repetitions)
    #print("Oracle")
    #print(results_oracle)
    #plot_theta_values(results_oracle, true_theta, lower_bounds, upper_bounds)
    #plot_loss_function(results_oracle)

    # Save results
    for rep, initial_value, result in results:
        np.savez(f'estimation_result_{rep}.npz', 
                 initial_value=initial_value, 
                 final_value=result.x, 
                 all_values=result.allvecs,
                 fun_values=result.allvecs)

    print(f"Estimation complete. Results saved in 'estimation_result_[0-{num_repetitions-1}].npz' files.")

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