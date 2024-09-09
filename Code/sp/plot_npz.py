import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
from datetime import datetime

def load_data(directory):
    data = []
    for file_path in glob.glob(os.path.join(directory, 'estimation_result_*.npz')):
        with np.load(file_path) as npz:
            data.append({
                'initial_value': npz['initial_value'],
                'final_value': npz['final_value'],
                'all_values': npz['all_values']
            })
    return data

def plot_histograms(data, param_names, output_dir):
    num_params = len(param_names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    for i, param_name in enumerate(param_names):
        final_values = [d['final_value'][i] for d in data]
        axes[i].hist(final_values, bins=30, edgecolor='black')
        axes[i].set_title(f'{param_name} Final Values')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_histograms.png'))
    plt.close()

def plot_line_charts(data, param_names, true_theta, lower_bounds, upper_bounds, output_dir):
    num_params = len(param_names)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # Use a colormap to assign different colors to each run
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, len(data))]
    
    for i, param_name in enumerate(param_names):
        for j, (d, color) in enumerate(zip(data, colors)):
            iterations = range(len(d['all_values']))
            values = d['all_values'][:, i]
            axes[i].plot(iterations, values, color=color, linewidth=0.5, alpha=0.5)
        
        axes[i].axhline(y=true_theta[i], color='r', linestyle='--', linewidth=2, label='True Value')
        axes[i].axhline(y=lower_bounds[i], color='g', linestyle=':', linewidth=2, label='Lower Bound')
        axes[i].axhline(y=upper_bounds[i], color='g', linestyle=':', linewidth=2, label='Upper Bound')
        axes[i].set_title(f'{param_name} Evolution')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Value')
        
        # Only add legend for the first subplot to avoid clutter
        if i == 0:
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_evolution.png'))
    plt.close()

def main(date_str, rhot = False):
    if date_str == 'local':
        data_dir = './simres'
        data = load_data(data_dir)
        current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        output_dir = f'../simres/simres_local_{current_datetime}'
    else:
        output_dir = f'~/simres/simres_{date_str}'
        data = load_data(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    param_names = [
        r'$\mu_1$',
        r'$\mu_2$',
        r'$\gamma_1$',
        r'$\gamma_2$',
        r'$\sigma_1$',
        r'$\sigma_2$',
        r'$\rho_s$',
    ]
   
    true_theta = [1.8, 2, 0.5, 0, 1, 1, 0.5]
    lower_bounds = np.array([1, 1, -.5, -1, 0, 0, -1])
    upper_bounds = np.array([3, 3, 1.5, 1, 2, 2, 1])

    if rhot == True:
        param_names.append(r'$\rho_t$')
        true_theta = np.append(true_theta, 0)
        lower_bounds = np.append(lower_bounds, -1)
        upper_bounds = np.append(upper_bounds, 1) 
    
    plot_histograms(data, param_names, output_dir)
    plot_line_charts(data, param_names, true_theta, lower_bounds, upper_bounds, output_dir)
    
    print(f"Plots have been saved in the directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate parameter plots with date-specific output.")
    parser.add_argument("date", help="Date string for the output directory (format: YYYYMMDDHHMMSS)")
    parser.add_argument("rhot", help="Include rho_t in the plots", type=bool)
    args = parser.parse_args()
    main(args.date, args.rhot)