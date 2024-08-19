# Implementation of the Roy model simulation that is very close to the original code

import torch

from NND import Discriminator_paper
from roy_helper_functions import royinv
from estimation import train_kpm

# Parameters
true_theta = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0, 0.5, 0.9])
num_samples = 300
num_repetitions = 10
n_discriminator = 100

# Training
results = train_kpm(royinv, Discriminator_paper, torch.nn.BCEWithLogitsLoss(), true_theta, num_samples=num_samples, num_repetitions=num_repetitions, n_discriminator=n_discriminator)
print(results)