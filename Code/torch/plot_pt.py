import torch
import matplotlib.pyplot as plt
import numpy as np

param_names = [
        r'$\mu_1$',
        r'$\mu_2$',
        r'$\gamma_1$',
        r'$\gamma_2$',
        r'$\sigma_1$',
        r'$\sigma_2$',
        r'$\rho_s$',
        r'$\rho_t$'
]
true_theta = [1.8, 2, 0.5, 0, 1, 1, 0.5, 0]
n = 300

# Load the data
all_params = torch.load('simres/all_params.pt', map_location='cpu')
all_losses = torch.load('simres/all_losses.pt', map_location='cpu')

# Get dimensions
S, num_params, n_generator = all_params.shape
#_, n_generator_loss = all_losses.shape

# Plot parameter values
plt.figure(figsize=(12, 8))
for i in range(num_params):
    plt.subplot(2, 4, i+1)
    for j in range(S):
        plt.plot(range(n_generator), all_params[j, i, :].cpu().numpy(), alpha=0.5, linewidth=0.5)
    plt.axhline(true_theta[i], color='black', linewidth=1)
    plt.title(f'{param_names[i]}')
    plt.xlabel('Generator Steps')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()
#plt.save('parameter_plots.png')
plt.close()

# Plot losses
plt.figure(figsize=(10, 6))
for i in range(S):
    plt.plot(range(n_generator), all_losses[i, :].cpu().numpy(), alpha=0.5, linewidth=0.5)
plt.title('Losses')
plt.xlabel('Generator Steps')
plt.ylabel('Loss')
plt.show()
#plt.save('loss_plot.png')
plt.close()

""" # Plot histograms of final parameter estimates
first_params = all_params[:, :, 0].cpu().numpy()
final_params = all_params[:, :, -1].cpu().numpy()

intial_means = np.mean(first_params, axis=0)
estimated_means = np.mean(final_params, axis=0)
estimated_stds = np.std(final_params, axis=0) * np.sqrt(n)

plt.figure(figsize=(12, 8))
for i in range(num_params):
    plt.subplot(2, 4, i+1)
    plt.axvline(true_theta[i], color='black', linewidth=1)
    plt.axvline(estimated_means[i], color='blue', linewidth=1)
    plt.axvline(intial_means[i], color='red', linewidth=1)
    plt.hist(first_params[:, i], bins=20, edgecolor='black')
    plt.title(f'{param_names[i]} (se: {estimated_stds[i]:.2f})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()
plt.close() """