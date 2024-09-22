import torch
import matplotlib.pyplot as plt
import numpy as np

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
        plt.plot(range(n_generator), all_params[j, i, :].cpu().numpy())
    plt.title(f'Parameter {i+1}')
    plt.xlabel('Generator Steps')
    plt.ylabel('Value')
plt.tight_layout()
plt.show()
#plt.save('parameter_plots.png')
plt.close()

# Plot losses
plt.figure(figsize=(10, 6))
for i in range(S):
    plt.plot(range(n_generator), all_losses[i, :].cpu().numpy())
plt.title('Losses')
plt.xlabel('Generator Steps')
plt.ylabel('Loss')
plt.show()
#plt.save('loss_plot.png')
plt.close()

#print("Plots have been saved as 'parameter_plots.png' and 'loss_plot.png'.")