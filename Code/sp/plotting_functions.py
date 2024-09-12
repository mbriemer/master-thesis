import matplotlib.pyplot as plt
import numpy as np

def plot_theta_values(results, true_theta, lower_bounds, upper_bounds, filename='theta_values.png'):
    num_repetitions = len(results)
    num_params = len(true_theta)

    fig, axs = plt.subplots(num_params, 1, figsize=(12, 4*num_params), sharex=True)
    fig.suptitle('Theta Values During Optimization')

    for i in range(num_params):
        for rep in range(num_repetitions):
            all_x = results[rep].allvecs
            axs[i].plot(range(len(all_x)), [x[i] for x in all_x], alpha=0.5, label=f'Rep {rep+1}' if i == 0 else '')
        
        axs[i].axhline(y=true_theta[i], color='r', linestyle='--', label='True value' if i == 0 else '')
        axs[i].axhline(y=lower_bounds[i], color='g', linestyle=':', label='Lower bound' if i == 0 else '')
        axs[i].axhline(y=upper_bounds[i], color='g', linestyle=':', label='Upper bound' if i == 0 else '')
        axs[i].set_ylabel(f'Theta {i+1}')
        axs[i].grid(True)

    axs[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    axs[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_loss_function(results, filename='loss_function.png'):
    num_repetitions = len(results)

    plt.figure(figsize=(12, 6))
    for rep in range(num_repetitions):
        all_x = results[rep].allvecs
        loss_values = [results[rep].fun] * len(all_x)  # Assuming the loss is stored in 'fun'
        plt.plot(range(len(all_x)), loss_values, label=f'Rep {rep+1}')

    plt.title('Loss Function Values During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_results(results):
    """moved from roy_helper_functions.py"""
    all_mu_values, all_sigma_values, all_discriminator_losses, all_generator_losses, iteration_numbers = results
    num_repetitions = len(all_mu_values)
    
    plt.figure(figsize=(12, 5))

    # Plot mu
    plt.subplot(2, 2, 1)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_mu_values[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('μ over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('μ')
    
    # Plot sigma
    plt.subplot(2, 2, 2)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_sigma_values[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('σ over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('σ')
    
    # Plot discriminator loss
    plt.subplot(2, 2, 3)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_discriminator_losses[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('Discriminator loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot generator loss
    plt.subplot(2, 2, 4)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_generator_losses[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('Generator loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
