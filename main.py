import torch
#torch.manual_seed(521)
torch.manual_seed(52891824)
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal

import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import torch.nn as nn

def logEexpmax(mu1, mu2, sig1, sig2, rho):
    theta = torch.sqrt(torch.tensor((sig1 - sig2)**2 + 2 * (1 - rho) * sig1 * sig2))
    normal_dist = torch.distributions.Normal(0, 1)
    e1 = mu1 + sig1**2 / 2 + torch.log(normal_dist.cdf((mu1 - mu2 + sig1**2 - rho * sig1 * sig2) / theta))
    e2 = mu2 + sig2**2 / 2 + torch.log(normal_dist.cdf((mu2 - mu1 + sig2**2 - rho * sig1 * sig2) / theta))
    e = torch.logaddexp(e1, e2)
    return e

def smoothing_function(x, lambda_val):
    diffs = torch.diff(x).float()
    std_diffs = torch.std(diffs)
    smoothed = 1 + torch.special.ndtr(diffs / (lambda_val * std_diffs))

    return smoothed.squeeze()

def transform_sigmoid(x, lower=1, upper=3):
    return lower + (upper - lower) * torch.sigmoid(x)

def transform_tanh(x, lower, upper):
    return (lower + upper) / 2 + (upper - lower) / 2 * torch.tanh(x)

def inverse_transform_sigmoid(y, lower=1, upper=3):
    x = (y - lower) / (upper - lower)
    return math.log(x / (1 - x))

def inverse_transform_tanh(y, lower=1, upper=3):
    x = (2 * y - (lower + upper)) / (upper - lower)
    return math.atanh(x)


class PolakRibiereCG(torch.optim.Optimizer):
    def __init__(self, params, max_iter=50, tol=1e-5):
        defaults = dict(max_iter=max_iter, tol=tol)
        super(PolakRibiereCG, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        loss = None
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['dir'] = torch.zeros_like(p.data)
                    state['prev_grad'] = torch.zeros_like(p.data)

                grad = p.grad
                prev_grad = state['prev_grad']
                dir = state['dir']

                # Polak-Ribière formula
                beta = torch.sum((grad - prev_grad) * grad) / torch.sum(prev_grad * prev_grad)
                beta = torch.clamp(beta, min=0)  # Ensure beta is non-negative

                # Update direction
                dir = -grad + beta * dir

                # Line search
                alpha = 1.0
                init_loss = closure()
                for _ in range(group['max_iter']):
                    p.data.add_(alpha * dir)
                    loss = closure()
                    if loss < init_loss:
                        break
                    p.data.add_(-alpha * dir)  # Revert step
                    alpha *= 0.5

                # Update state
                state['prev_grad'] = grad.clone()
                state['dir'] = dir
                state['step'] += 1

        return loss

class Generator_net(torch.nn.Module):

    def __init__(self, in_features = 1, out_features = 1):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta = Parameter(torch.Tensor(9))
        self.reset_parameters() 
        self.stack = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def reset_parameters(self):
        self.theta.data = torch.zeros_like(self.theta)
        self.theta.data[8] = 0.9

    def forward(self, num_samples):
        mu_1, mu_2, gamma_1, gamma_2, log_sigma_1, log_sigma_2, arctanh_rho_s, arctanh_rho_t, beta = self.theta
        sigma_1 = torch.exp(log_sigma_1)
        sigma_2 = torch.exp(log_sigma_2)
        rho_s = torch.tanh(arctanh_rho_s)
        rho_t = torch.tanh(arctanh_rho_t)

        beta = beta.detach()
        rho_t = rho_t.detach()

        logits = self.stack([mu_1, mu_2, gamma_1, gamma_2, log_sigma_1, log_sigma_2, arctanh_rho_s, arctanh_rho_t, beta])
        return logits

class Generator(torch.nn.Module):

    def __init__(self, in_features = 1, out_features = 1, bias=True):
        super(Generator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.theta = Parameter(torch.Tensor(9))
        self.reset_parameters()    

    def reset_parameters(self):
        self.theta.data = torch.tensor([inverse_transform_sigmoid(2, lower=1., upper=3.), # mu_1
                                        inverse_transform_sigmoid(2, lower=1., upper=3.), # mu_2
                                        inverse_transform_sigmoid(0, lower=-0.5, upper=1.5), # gamma_1
                                        inverse_transform_sigmoid(0, lower=-1., upper=1.), # gamma_2
                                        inverse_transform_sigmoid(1, lower=0., upper=2.), # sigma_1
                                        inverse_transform_sigmoid(1, lower=0., upper=2.), # sigma_2
                                        math.atanh(0), # rho_s
                                        math.atanh(0), # rho_t
                                        0.9]) # beta

    def forward(self, num_samples):
        lambda_val = 1
        logit_mu_1, logit_mu_2, logit_gamma_1, logit_gamma_2, log_sigma_1, log_sigma_2, arctanh_rho_s, arctanh_rho_t, beta = self.theta

        mu_1 = transform_sigmoid(logit_mu_1, lower=1., upper=3.)
        mu_2 = transform_sigmoid(logit_mu_2, lower=1., upper=3.)
        gamma_1 = transform_sigmoid(logit_gamma_1, lower=-0.5, upper=1.5)
        gamma_2 = transform_sigmoid(logit_gamma_2, lower=-1., upper=1.)
        sigma_1 = transform_sigmoid(log_sigma_1, lower=0., upper=2.)
        sigma_2 = transform_sigmoid(log_sigma_2, lower=0., upper=2.)
        rho_s = torch.tanh(arctanh_rho_s)
        rho_t = torch.tanh(arctanh_rho_t)

        beta = beta.detach()
        rho_t = rho_t.detach()

        eps_mu = torch.zeros(4)

        eps_sigma = torch.tensor([[sigma_1**2, rho_s * sigma_1 * sigma_2, rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2],
                                [rho_s * sigma_1 * sigma_2, sigma_2**2, rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2],
                                [rho_t * sigma_1**2, rho_s * rho_t * sigma_1 * sigma_2, sigma_1**2, rho_s * sigma_1 * sigma_2],
                                [rho_s * rho_t * sigma_1 * sigma_2, rho_t * sigma_2**2, rho_s * sigma_1 * sigma_2, sigma_2**2]])

        eps = torch.distributions.MultivariateNormal(eps_mu, eps_sigma).rsample((num_samples,))

        # Log wages at t = 1 for each sector

        logw1m = torch.column_stack((mu_1 + eps[:, 0],
                                mu_2 + eps[:, 1]))

        # Value function at t = 1 for each sector

        logv1m = torch.column_stack((
            torch.logaddexp(logw1m[:, 0], torch.log(beta) + logEexpmax(mu_1 + gamma_1, mu_2, sigma_1, sigma_2, rho_s)),
            torch.logaddexp(logw1m[:, 1], torch.log(beta) + logEexpmax(mu_1, mu_2 + gamma_2, sigma_1, sigma_2, rho_s))
        ))

        # Sector choices at t = 1

        d1 = torch.argmax(logv1m, axis=1)

        # Observed log wages at t = 1

        logw1 = logv1m[torch.arange(num_samples), d1]

        # Log wages at t == 2

        logw2m = torch.column_stack((
            mu_1 + gamma_1 * (d1 == 0) + eps[:, 2],
            mu_2 + gamma_2 * (d1 == 1) + eps[:, 3]
        ))

        # % Observed log wages and sector choices at t = 2

        logw2 = torch.max(logw2m, axis=1).values
        d2 = torch.argmax(logw2m, axis=1)

        d1_smooth = smoothing_function(x = logv1m, lambda_val = lambda_val)
        d2_smooth = smoothing_function(x = logw2m, lambda_val = lambda_val)
        return torch.stack([logw1, d1_smooth, logw2, d2_smooth], dim = -1)
    
    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features is not None
        )

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.stack = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        logits = self.stack(x)
        return logits
    
class Discriminator_paper(nn.Module): #flat net like in the paper

    def __init__(self, input_size=4, hidden_size=10, output_size=1):
        super(Discriminator_paper, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        #self.activation = nn.Tanh()

    def forward(self, x):
        x = self.output(self.hidden(x))
        #x = self.output(x)
        return x

class LogisticDiscriminator(nn.Module):
    def __init__(self):
        super(LogisticDiscriminator, self).__init__()
        self.linear = nn.Linear(7, 1)  # 7 input features, 1 output
        #self.sigmoid = nn.Sigmoid()  # Λ function

    def forward(self, x):

        log_w1, d1, log_w2, d2 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

        # Compute the additional features
        log_w1_squared = log_w1 ** 2
        log_w2_squared = log_w2 ** 2
        log_w1_log_w2 = log_w1 * log_w2

        # Concatenate all features
        x = torch.stack([log_w1, d1, log_w2, d2, log_w1_squared, log_w2_squared, log_w1_log_w2], dim=-1)

        # Apply linear transformation
        return self.linear(x)

def train(Generator_object, Discriminator_object, criterion, inverse_theta, num_iterations = 1500, num_samples = 300, num_repetitions = 10, d_every = 1, g_every = 1, wd = 0.01):
    
    all_mu_values = [[] for _ in range(num_repetitions)]
    all_gamma_values = [[] for _ in range(num_repetitions)]
    all_sigma_values = [[] for _ in range(num_repetitions)]
    all_rho_values = [[] for _ in range(num_repetitions)]
    all_discriminator_losses = [[] for _ in range(num_repetitions)]
    all_generator_losses = [[] for _ in range(num_repetitions)]
    iteration_numbers = []

    for rep in tqdm(range(num_repetitions)):

        discriminator = Discriminator_object()
        generator = Generator_object()

        optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-2, weight_decay=wd)
        optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-2)

        true_generator = Generator_object()
        true_generator.theta.data = inverse_theta

        true_samples = true_generator.forward(num_samples).detach()

        for i in tqdm(range(num_iterations)):

            # Train the discriminator every 10 iterations
            if i % d_every == 0:
                optimizerD.zero_grad()
                
                #generator.theta.data[generator.theta.data > 10.] = 10.
                try:
                    fake_samples = generator.forward(num_samples)
                except Exception as e:
                    print(e)
                    print(generator.theta.data)
                    print(generator.theta.grad)

                fake_logits = discriminator(fake_samples)  # Detach fake samples from the generator's graph
                true_logits = discriminator(true_samples)
                
                discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))
                
                discriminator_loss.backward()
                optimizerD.step()
            
            # Train the generator
            if i % g_every == 0:
                optimizerG.zero_grad()
                
                fake_samples = generator.forward(num_samples)
                fake_logits = discriminator(fake_samples)
                
                generator_loss = criterion(fake_logits, torch.ones_like(fake_logits))
                
                generator_loss.backward()
                optimizerG.step()

            # Current parameter values

            new_mu = transform_sigmoid(generator.theta.data[0:2], lower=1., upper=3.).detach().cpu().numpy().copy()
            new_gamma_1 = transform_sigmoid(generator.theta.data[2], lower=-0.5, upper=1.5).detach().cpu().numpy().copy()
            new_gamma_2 = transform_sigmoid(generator.theta.data[3], lower=-1., upper=1.).detach().cpu().numpy().copy()
            new_sigma = transform_sigmoid(generator.theta.data[4:6], lower=0., upper=2.).detach().cpu().numpy().copy()
            new_rho = torch.tanh(generator.theta.data[6:8]).detach().cpu().numpy().copy()

            # Store current parameter values
            all_mu_values[rep].append(new_mu)
            all_gamma_values[rep].append([new_gamma_1, new_gamma_2])
            all_sigma_values[rep].append(new_sigma)
            all_rho_values[rep].append(new_rho)
            all_discriminator_losses[rep].append(discriminator_loss.item())
            all_generator_losses[rep].append(generator_loss.item())

            if rep == 0:  # Only need to store iteration numbers once
                    iteration_numbers.append(i)
                
    return [all_mu_values, all_gamma_values, all_sigma_values, all_rho_values, all_discriminator_losses, all_generator_losses, iteration_numbers]

def plot_results(results):
    
    all_mu_values, all_gamma_values, all_sigma_values, all_rho_values, all_discriminator_losses, all_generator_losses, iteration_numbers = results
    
    true_values = {
        'mu_1': 1.8, 'mu_2': 2.0,
        'gamma_1': 0.5, 'gamma_2': 0.0,
        'sigma_1': 1.0, 'sigma_2': 1.0,
        'rho_s': 0.5, 'rho_t': 0.0
    }
    
    num_repetitions = len(all_mu_values)
    
    plt.figure(figsize=(15, 10))

    # Plot mu
    plt.subplot(2, 2, 1)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, [mu[0] for mu in all_mu_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
        plt.plot(iteration_numbers, [mu[1] for mu in all_mu_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
    plt.axhline(y=true_values['mu_1'], color='C0', linestyle='--', label='True μ₁')
    plt.axhline(y=true_values['mu_2'], color='C1', linestyle='--', label='True μ₂')
    plt.title('μ over iterations')
    plt.legend()

    # Plot gamma
    plt.subplot(2, 2, 2)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, [gamma[0] for gamma in all_gamma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
        plt.plot(iteration_numbers, [gamma[1] for gamma in all_gamma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
    plt.axhline(y=true_values['gamma_1'], color='C0', linestyle='--', label='True γ₁')
    plt.axhline(y=true_values['gamma_2'], color='C1', linestyle='--', label='True γ₂')
    plt.title('γ over iterations')
    plt.legend()

    # Plot sigma
    plt.subplot(2, 2, 3)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, [sigma[0] for sigma in all_sigma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
        plt.plot(iteration_numbers, [sigma[1] for sigma in all_sigma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
    plt.axhline(y=true_values['sigma_1'], color='C0', linestyle='--', label='True σ₁')
    plt.axhline(y=true_values['sigma_2'], color='C1', linestyle='--', label='True σ₂')
    plt.title('σ over iterations')
    plt.legend()

    # Plot rho
    plt.subplot(2, 2, 4)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, [rho[0] for rho in all_rho_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
        plt.plot(iteration_numbers, [rho[1] for rho in all_rho_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
    plt.axhline(y=true_values['rho_s'], color='C0', linestyle='--', label='True ρₛ')
    plt.axhline(y=true_values['rho_t'], color='C1', linestyle='--', label='True ρₜ')
    plt.title('ρ over iterations')
    plt.legend()

    #plt.tight_layout()
    #plt.show()

    # Plot losses
    plt.figure(figsize=(15, 5))
    # Plot discriminator loss
    plt.subplot(1, 2, 1)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_discriminator_losses[rep], color='C0', alpha=0.5, linewidth=0.5)
    plt.title('Discriminator Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    # Plot generator loss
    plt.subplot(1, 2, 2)
    for rep in range(num_repetitions):
        plt.plot(iteration_numbers, all_generator_losses[rep], color='C1', alpha=0.5, linewidth=0.5)
    plt.title('Generator Loss over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.show()
    

# True data

mu_1 = 1.8
mu_2 = 2.
gamma_1 = 0.5
gamma_2 = 0.
sigma_1 = 1.
sigma_2 = 1.
rho_s = 0.5
rho_t = 0.
beta = 0.9

inverse_theta = torch.tensor([inverse_transform_sigmoid(mu_1, lower=1., upper=3.), 
                            inverse_transform_sigmoid(mu_2, lower=1., upper=3.),
                            inverse_transform_sigmoid(gamma_1, lower=-0.5, upper=1.5), 
                            inverse_transform_sigmoid(gamma_2, lower=-1., upper=1.),
                            inverse_transform_sigmoid(sigma_1, lower = 0., upper = 2.), 
                            inverse_transform_sigmoid(sigma_2, lower = 0., upper = 2.),
                            math.atanh(rho_s), 
                            math.atanh(rho_t), 
                            beta])

# GAN

criterion = nn.BCEWithLogitsLoss()

#all_mu_values, all_gamma_values, all_sigma_values, all_rho_values, all_discriminator_losses, all_generator_losses, iteration_numbers = train(Generator, Discriminator, criterion, inverse_theta)
#plot_results(all_mu_values, all_gamma_values, all_sigma_values, all_rho_values, all_discriminator_losses, all_generator_losses, iteration_numbers)

results = train(Generator, Discriminator, criterion, inverse_theta)
plot_results(results)

# Close to paper



'''
num_iterations = 2000
num_samples = 1000
num_repetitions = 10


#generator.theta.data = torch.tensor([1.8, 2.0, 0.5, 0., 1., 1., 0.5, 0., 0.9])

# Training Loop

# Initialize lists to store parameter values for each repetition
all_mu_values = [[] for _ in range(num_repetitions)]
all_gamma_values = [[] for _ in range(num_repetitions)]
all_sigma_values = [[] for _ in range(num_repetitions)]
all_rho_values = [[] for _ in range(num_repetitions)]
iteration_numbers = []


print("Starting Training Loop...")

for rep in tqdm(range(num_repetitions)):

    generator = Generator()    
    discriminator = Discriminator()

    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-2, weight_decay=0.01, betas=(0.9, 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=1e-2, betas=(0.9, 0.999))

    true_generator = Generator()
    true_generator.theta.data = torch.tensor([inverse_transform_sigmoid(mu_1, lower=1., upper=3.), 
                                            inverse_transform_sigmoid(mu_2, lower=1., upper=3.),
                                            inverse_transform_sigmoid(gamma_1, lower=-0.5, upper=1.5), 
                                            inverse_transform_sigmoid(gamma_2, lower=-1., upper=1.),
                                            inverse_transform_sigmoid(sigma_1, lower = 0., upper = 2.), 
                                            inverse_transform_sigmoid(sigma_2, lower = 0., upper = 2.),
                                            math.atanh(rho_s), 
                                            math.atanh(rho_t), 
                                            beta])
    true_samples = true_generator.forward(num_samples).detach()

    for i in tqdm(range(num_iterations)):

        # Train the discriminator every 10 iterations
        if i % 10 == 0:
            optimizerD.zero_grad()
            
            #generator.theta.data[generator.theta.data > 10.] = 10.
            fake_samples = generator.forward(num_samples)
            
            fake_logits = discriminator(fake_samples)  # Detach fake samples from the generator's graph
            true_logits = discriminator(true_samples)
            
            discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))
            
            discriminator_loss.backward()
            optimizerD.step()
        
        # Train the generator
        optimizerG.zero_grad()
        
        fake_samples = generator.forward(num_samples)
        fake_logits = discriminator(fake_samples)
        
        generator_loss = criterion(fake_logits, torch.ones_like(fake_logits))
        
        generator_loss.backward()
        optimizerG.step()

        # Current parameter values

        new_mu = transform_sigmoid(generator.theta.data[0:2], lower=1., upper=3.).detach().cpu().numpy().copy()
        new_gamma_1 = transform_sigmoid(generator.theta.data[2], lower=0.5, upper=1.5).detach().cpu().numpy().copy()
        new_gamma_2 = transform_sigmoid(generator.theta.data[3], lower=-1., upper=1.).detach().cpu().numpy().copy()
        new_sigma = transform_sigmoid(generator.theta.data[4:6], lower=0., upper=2.).detach().cpu().numpy().copy()

        
        # Output training stats
        if i % 10 == 0:    
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (i, num_iterations, discriminator_loss.item(), generator_loss.item()))
            #print(generator.theta.data)
            print('mu:', new_mu,
                '\ngamma: ', [new_gamma_1, new_gamma_2],
                '\nsigma: ', new_sigma,
                '\nrho: ', torch.tanh(generator.theta.data[6:8]),
                '\nbeta: ', generator.theta.data[8])
            print(generator.theta.grad)
        

        # Store current parameter values
        all_mu_values[rep].append(new_mu)
        all_gamma_values[rep].append([new_gamma_1, new_gamma_2])
        all_sigma_values[rep].append(new_sigma)
        all_rho_values[rep].append(torch.tanh(generator.theta.data[6:8]).detach().cpu().numpy())

        if rep == 0:  # Only need to store iteration numbers once
                iteration_numbers.append(i)

# True values
true_values = {
    'mu_1': 1.8, 'mu_2': 2.0,
    'gamma_1': 0.5, 'gamma_2': 0.0,
    'sigma_1': 1.0, 'sigma_2': 1.0,
    'rho_s': 0.5, 'rho_t': 0.0
}

# Plotting
plt.figure(figsize=(15, 10))

# Plot mu
plt.subplot(2, 2, 1)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [mu[0] for mu in all_mu_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [mu[1] for mu in all_mu_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['mu_1'], color='C0', linestyle='--', label='True μ₁')
plt.axhline(y=true_values['mu_2'], color='C1', linestyle='--', label='True μ₂')
plt.title('μ over iterations')
plt.legend()

# Plot gamma
plt.subplot(2, 2, 2)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [gamma[0] for gamma in all_gamma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [gamma[1] for gamma in all_gamma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['gamma_1'], color='C0', linestyle='--', label='True γ₁')
plt.axhline(y=true_values['gamma_2'], color='C1', linestyle='--', label='True γ₂')
plt.title('γ over iterations')
plt.legend()

# Plot sigma
plt.subplot(2, 2, 3)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [sigma[0] for sigma in all_sigma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [sigma[1] for sigma in all_sigma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['sigma_1'], color='C0', linestyle='--', label='True σ₁')
plt.axhline(y=true_values['sigma_2'], color='C1', linestyle='--', label='True σ₂')
plt.title('σ over iterations')
plt.legend()

# Plot rho
plt.subplot(2, 2, 4)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [rho[0] for rho in all_rho_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [rho[1] for rho in all_rho_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['rho_s'], color='C0', linestyle='--', label='True ρₛ')
plt.axhline(y=true_values['rho_t'], color='C1', linestyle='--', label='True ρₜ')
plt.title('ρ over iterations')
plt.legend()

plt.tight_layout()
plt.show()



discriminator = Discriminator_paper()

optimizerD = torch.optim.Adam(discriminator.parameters(), lr=1e-2, weight_decay=0.01)

# Training loop that's closer to the paper's
print("...like in the paper")

for rep in tqdm(range(num_repetitions)):

    for i in tqdm(range(num_iterations)):

        # Train the discriminator every 10 iterations
        if i % 1 == 0:
            optimizerD.zero_grad()
            
            #generator.theta.data[generator.theta.data > 10.] = 10.
            fake_samples = generator.forward(num_samples)
            
            fake_logits = discriminator(fake_samples)  # Detach fake samples from the generator's graph
            true_logits = discriminator(true_samples)
            
            discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))
            
            discriminator_loss.backward()
            optimizerD.step()
        
        # Train the generator
        optimizerG.zero_grad()
        
        fake_samples = generator.forward(num_samples)
        fake_logits = discriminator(fake_samples)
        
        generator_loss = criterion(fake_logits, torch.ones_like(fake_logits))
        
        generator_loss.backward()
        optimizerG.step()

        # Current parameter values

        new_mu = transform_sigmoid(generator.theta.data[0:2], lower=1., upper=3.).detach().cpu().numpy().copy()
        new_gamma_1 = transform_sigmoid(generator.theta.data[2], lower=0.5, upper=1.5).detach().cpu().numpy().copy()
        new_gamma_2 = transform_sigmoid(generator.theta.data[3], lower=-1., upper=1.).detach().cpu().numpy().copy()
        new_sigma = transform_sigmoid(generator.theta.data[4:6], lower=0., upper=2.).detach().cpu().numpy().copy()

        
        # Output training stats
        if i % 10 == 0:    
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (i, num_iterations, discriminator_loss.item(), generator_loss.item()))
            #print(generator.theta.data)
            print('mu:', new_mu,
                '\ngamma: ', [new_gamma_1, new_gamma_2],
                '\nsigma: ', new_sigma,
                '\nrho: ', torch.tanh(generator.theta.data[6:8]),
                '\nbeta: ', generator.theta.data[8])
            print(generator.theta.grad)
        

        # Store current parameter values
        all_mu_values[rep].append(new_mu)
        all_gamma_values[rep].append([new_gamma_1, new_gamma_2])
        all_sigma_values[rep].append(new_sigma)
        all_rho_values[rep].append(torch.tanh(generator.theta.data[6:8]).detach().cpu().numpy())

        if rep == 0:  # Only need to store iteration numbers once
                iteration_numbers.append(i)


# True values
true_values = {
    'mu_1': 1.8, 'mu_2': 2.0,
    'gamma_1': 0.5, 'gamma_2': 0.0,
    'sigma_1': 1.0, 'sigma_2': 1.0,
    'rho_s': 0.5, 'rho_t': 0.0
}

# Plotting
plt.figure(figsize=(15, 10))

# Plot mu
plt.subplot(2, 2, 1)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [mu[0] for mu in all_mu_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [mu[1] for mu in all_mu_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['mu_1'], color='C0', linestyle='--', label='True μ₁')
plt.axhline(y=true_values['mu_2'], color='C1', linestyle='--', label='True μ₂')
plt.title('μ over iterations')
plt.legend()

# Plot gamma
plt.subplot(2, 2, 2)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [gamma[0] for gamma in all_gamma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [gamma[1] for gamma in all_gamma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['gamma_1'], color='C0', linestyle='--', label='True γ₁')
plt.axhline(y=true_values['gamma_2'], color='C1', linestyle='--', label='True γ₂')
plt.title('γ over iterations')
plt.legend()

# Plot sigma
plt.subplot(2, 2, 3)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [sigma[0] for sigma in all_sigma_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [sigma[1] for sigma in all_sigma_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['sigma_1'], color='C0', linestyle='--', label='True σ₁')
plt.axhline(y=true_values['sigma_2'], color='C1', linestyle='--', label='True σ₂')
plt.title('σ over iterations')
plt.legend()

# Plot rho
plt.subplot(2, 2, 4)
for rep in range(num_repetitions):
    plt.plot(iteration_numbers, [rho[0] for rho in all_rho_values[rep]], color='C0', alpha=0.5, linewidth=0.5)
    plt.plot(iteration_numbers, [rho[1] for rho in all_rho_values[rep]], color='C1', alpha=0.5, linewidth=0.5)
plt.axhline(y=true_values['rho_s'], color='C0', linestyle='--', label='True ρₛ')
plt.axhline(y=true_values['rho_t'], color='C1', linestyle='--', label='True ρₜ')
plt.title('ρ over iterations')
plt.legend()

plt.tight_layout()
plt.show()
'''