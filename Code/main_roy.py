# Implementation of the Roy model simulation that is very close to the original code
import estimagic as em

from NND import Discriminator_paper
from roy_helper_functions import royinv
from estimation import train_kpm

# Parameters
true_theta = [1.8, 2, 0.5, 0, 1, 1, 0.5, 0, 0.9]
num_samples = 300
num_repetitions = 10
n_discriminator = 100

results = train_kpm(royinv, true_theta, num_hidden=10, g=5, num_samples=num_samples, num_repetitions=num_repetitions)
'''
plt_loss = em.criterion_plot(results)
plt_loss.show

plt_params = em.params_plot(results)
plt_params.show()

'''