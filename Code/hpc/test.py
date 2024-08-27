import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import multiprocessing as mp

def loss(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def single_repetition(args):
    rep, seed = args
    rng = np.random.default_rng(seed)
    
    print(f"Starting run {rep}")
    initial_value = rng.uniform(0,3,2)
    result = minimize(loss, initial_value, method='Nelder-Mead', options={'return_all': True})
    return rep, initial_value, result.x

def parallelized(num_runs):
    num_processes = mp.cpu_count()
    print(f"Number of cores: {num_processes}")

    # Create a SeedSequence
    ss = np.random.SeedSequence()
    # Generate child seeds for each run
    child_seeds = ss.spawn(num_runs)
    
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(single_repetition, enumerate(child_seeds))
    
    # Sort results by run number to ensure order
    results.sort(key=lambda x: x[0])
    
    # Print results in order
    for rep, initial_value, optimum in results:
        print(f"Run {rep} complete. Started at {initial_value}, found optimum: {optimum}.")
    
    return results

if __name__ == "__main__":
    num_runs = 100
    results = parallelized(num_runs=num_runs)
    print(f"Total runs completed: {len(results)}")

    """
    for r in range(len(results)):
        rep = results[r]
        # Create plot
        fig, ax = plt.subplots()

        # Extract all points from the optimization path
        all_points = np.array(rep.allvecs)

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
        plt.savefig(f'optimization_plot_{r}.png')

        # Store output
        with open(f'output_{r}.txt', 'w') as f:
            f.write(str(rep))

    print("Optimization complete. Results saved in 'output.txt' and plot saved as 'optimization_plot.png'.")
    """