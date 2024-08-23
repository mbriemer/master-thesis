import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def loss(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

result = minimize(loss, [0, 0], method='Nelder-Mead', options={'return_all': True})

# Create plot
fig, ax = plt.subplots()

# Extract all points from the optimization path
all_points = np.array(result.allvecs)

# Plot the optimization path
ax.plot(all_points[:, 0], all_points[:, 1], 'b.-', alpha=0.5)

# Mark start and end points
ax.plot(all_points[0, 0], all_points[0, 1], 'go', label='Start')
ax.plot(all_points[-1, 0], all_points[-1, 1], 'ro', label='End')

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Optimization Path')
ax.legend()

# Add contour plot of the loss function
x = np.linspace(-1, 3, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
Z = loss([X, Y])
ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.5)

# Save the plot
plt.savefig('optimization_plot.png')

# Store output
with open('output.txt', 'w') as f:
    f.write(str(result))

print("Optimization complete. Results saved in 'output.txt' and plot saved as 'optimization_plot.png'.")