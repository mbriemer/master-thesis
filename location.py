import torch
import matplotlib.pyplot as plt

class Generator_location(torch.nn.Module):
    def __init__(self):
        super(Generator_location, self).__init__()

    def forward(self, u, theta):
        mu, sigma = theta
        return torch.distributions.Normal(mu, sigma).icdf(u)
    
class Discriminator_location(torch.nn.Module):
    pass    

n = 1000
u = torch.rand(n)
theta = (0, 1)
location = Generator_location()
x = location(u, theta)

plt.hist(u, density=True)
plt.hist(x, density=True)
plt.show()