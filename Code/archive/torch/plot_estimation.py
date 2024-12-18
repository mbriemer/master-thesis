import numpy as np
import matplotlib.pyplot as plt
import glob
import os

#Simulation parameters
true_theta = np.array([1.8, 2, 0.5, 0, 1, 1, 0.5, 0])
param_names = [
        r'$\mu_1$',
        r'$\mu_2$',
        r'$\gamma_1$',
        r'$\gamma_2$',
        r'$\sigma_1$',
        r'$\sigma_2$',
        r'$\rho_s$',
        r'$\rho_t$',
        r'$\beta$'
]

# Specify the folder containing the files
folder_path = './simres/js_wide_uniform/'  # Replace this with your actual folder path

# Find all files matching the pattern in the specified folder
file_pattern = os.path.join(folder_path, 'estimation_js_wide_uniform_*.npz')
files = glob.glob(file_pattern)

final_parameters = [[] for _ in range(8)]  # List to store final values for each parameter
all_parameter_evolution = [[] for _ in range(8)]  # List to store evolution of each parameter

# Load data from each file
for file in files:
    data = np.load(file, allow_pickle=True)
    result = data['arr_0'].item()  # Assuming the result object is stored in arr_0
    
    # Extract final parameter values
    for i, param_value in enumerate(result.x):
        final_parameters[i].append(param_value)

    # Print the number of optimization steps (iterations)
    print(f"File '{os.path.basename(file)}': Converged: {result.success}")
    
"""     # Extract parameter evolution
    for i, param_values in enumerate(result.allvecs.T):
        all_parameter_evolution[i].append(param_values) """

# Calculate mean and standard deviation for each parameter
param_stats = []
for param_values in final_parameters:
    mean = np.mean(param_values)
    std = np.std(param_values)
    param_stats.append((mean, std))

print('Final parameter statistics:')
for i, (mean, std) in enumerate(param_stats):
    print(f'{param_names[i]}: {mean:.2f} ± {std:.2f}')

# Plot histograms of final parameter values
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
fig.suptitle('Histograms of Final Parameter Values')

for i, param_values in enumerate(final_parameters):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    ax.hist(param_values, bins=20, edgecolor='black')
    ax.axvline(true_theta[i], color='r', linestyle='dashed', linewidth=2)
    ax.set_title(f'{param_names[i]}')
    ax.set_xlabel('Final Value')
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'main_case_histograms.png'))
plt.close()

# Plot line charts for parameter evolution
fig, axes = plt.subplots(4, 2, figsize=(15, 20))
fig.suptitle('Parameter Evolution During Optimization')

for i, param_evolution in enumerate(all_parameter_evolution):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    for run in param_evolution:
        ax.plot(run, alpha=0.5)
    
    ax.set_title(f'Parameter {i+1}')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Parameter Value')

plt.tight_layout()
plt.savefig(os.path.join(folder_path, 'parameter_evolution.png'))
plt.close()

print(f"Analysis complete. Check '{folder_path}' for 'final_parameters_histograms.png' and 'parameter_evolution.png'.")