import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def mvn_inverse_cdf(u, mu, sigma):
    
    # Compute the Cholesky decomposition of the covariance matrix
    L = torch.linalg.cholesky(sigma)
    
    # Compute the inverse CDF of a standard normal distribution
    normal = torch.distributions.Normal(0, 1)
    z = normal.icdf(u)
    
    # Transform the standard normal quantiles to multivariate normal quantiles
    return mu + torch.matmul(z, L.T)

class Generator_location(torch.nn.Module):
    def __init__(self, d):
        super(Generator_location, self).__init__()
        #self.theta = torch.nn.Parameter(torch.tensor([2., 3.]))
        self.theta = torch.nn.ParameterList([2 * torch.nn.Parameter(torch.ones(d)), 3 * torch.nn.Parameter(torch.eye(d))])

    def forward(self, u):
        mu, sigma = self.theta
        #return torch.distributions.Normal(mu, sigma).icdf(u)
        return mvn_inverse_cdf(u, mu, sigma)
    
class Discriminator_location(torch.nn.Module):
    def __init__(self, d):
        super(Discriminator_location, self).__init__()

        self.stack = torch.nn.Sequential(
            torch.nn.Linear(d, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.stack(x.unsqueeze(1)).squeeze(1)
    
def train_location(Discriminator_object, Generator_object, criterion, d, true_theta, num_repetitions = 10, num_iterations = 5000, num_samples = 1000, n_discriminator = 1, n_generator = 1):

    all_mu_values = [[] for _ in range(num_repetitions)]
    all_sigma_values = [[] for _ in range(num_repetitions)]
    all_discriminator_losses = [[] for _ in range(num_repetitions)]
    all_generator_losses = [[] for _ in range(num_repetitions)]
    iteration_numbers = range(num_iterations)

    for rep in tqdm(range(num_repetitions)):

        u = torch.distributions.Uniform(torch.zeros(d), torch.ones(d)).sample((num_samples,))

        discriminator = Discriminator_object(d)
        generator = Generator_object(d)

        optimizer_discriminator = torch.optim.Adam(discriminator.parameters())
        optimizer_generator = torch.optim.Adam(generator.parameters())

        true_generator = Generator_object(d)
        true_generator.theta[0].data = torch.tensor(true_theta[0])
        true_generator.theta[1].data = torch.tensor(true_theta[1])
        true_samples = true_generator.forward(u).detach()

        for _ in tqdm(range(num_iterations)):

            #u = torch.rand(num_samples)

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

            all_mu_values[rep].append(generator.theta[0][0].item())
            all_sigma_values[rep].append(generator.theta[1][0,0].item())
            all_discriminator_losses[rep].append(discriminator_loss.item())
            all_generator_losses[rep].append(generator_loss.item())

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

num_samples = 300
d = 3
#u = torch.distributions.Uniform(torch.zeros(d), torch.ones(d)).sample((num_samples,))
true_mu = torch.zeros(d)
true_sigma = torch.eye(d)
true_theta = [true_mu, true_sigma]

results = train_location(Discriminator_location, Generator_location, torch.nn.BCEWithLogitsLoss(), d, true_theta, num_repetitions=5, num_iterations = 3000, num_samples=num_samples)
plot_results(results)