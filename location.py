import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

class Generator_location(torch.nn.Module):
    def __init__(self):
        super(Generator_location, self).__init__()
        self.theta = torch.nn.Parameter(torch.tensor([2., 3.]))

    def forward(self, u):
        mu, sigma = self.theta
        return torch.distributions.Normal(mu, sigma).icdf(u)
    
class Discriminator_location(torch.nn.Module):
    def __init__(self):
        super(Discriminator_location, self).__init__()

        self.stack = torch.nn.Sequential(
            torch.nn.Linear(1, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.stack(x.unsqueeze(1)).squeeze(1)
    
def train_location(Discriminator_object, Generator_object, criterion, u, theta, num_repetitions = 10, num_iterations = 5000, num_samples = 1000, n_discriminator = 1, n_generator = 1):

    all_mu_values = [[] for _ in range(num_repetitions)]
    all_sigma_values = [[] for _ in range(num_repetitions)]
    all_discriminator_losses = [[] for _ in range(num_repetitions)]
    all_generator_losses = [[] for _ in range(num_repetitions)]
    iteration_numbers = []

    for rep in tqdm(range(num_repetitions)):

        discriminator = Discriminator_object()
        generator = Generator_object()

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
        optimizer_generator = torch.optim.Adam(generator.parameters())

        true_generator = Generator_object()
        true_generator.theta.data = torch.tensor(theta)
        true_samples = true_generator.forward(u).detach()

        for it in tqdm(range(num_iterations)):
            # Train discriminator
            for _ in range(n_discriminator):
                optimizer_discriminator.zero_grad()
                fake_samples = generator.forward(u)

                true_logits = discriminator(true_samples)
                fake_logits = discriminator(fake_samples)
                discriminator_loss = criterion(true_logits, torch.ones_like(true_logits)) + \
                       criterion(fake_logits, torch.zeros_like(fake_logits))
                discriminator_loss.backward()
                optimizer_discriminator.step()
            
            # Train generator
            for _ in range(n_generator):
                optimizer_generator.zero_grad()
                fake_samples = generator.forward(u)
                generator_loss = criterion(discriminator(fake_samples), torch.ones_like(discriminator(fake_samples)))
                generator_loss.backward()
                optimizer_generator.step()  

            all_mu_values[rep].append(generator.theta[0].item())
            all_sigma_values[rep].append(generator.theta[1].item())
            all_discriminator_losses[rep].append(discriminator_loss.item())
            all_generator_losses[rep].append(generator_loss.item())

            if rep == 0:
                iteration_numbers.append(it)

    return all_mu_values, all_sigma_values, all_discriminator_losses, all_generator_losses, iteration_numbers

def plot_results(results):
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
    

'''
n = 1000
u = torch.rand(n)
theta = (0, 1)
location = Generator_location()
x = location(u, theta)

plt.hist(u, density=True)
plt.hist(x, density=True)
plt.show()
'''

num_samples = 1000
u = torch.rand(num_samples)

results = train_location(Discriminator_location, Generator_location, torch.nn.BCEWithLogitsLoss(), u, (0., 1.), num_repetitions=5, num_samples=num_samples)
plot_results(results)