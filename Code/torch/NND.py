# Code for the neural network discriminator

import torch

class Discriminator_paper(torch.nn.Module): #flat net like in the paper
    def __init__(self, input_size=4, hidden_size=10, output_size=1):
        super(Discriminator_paper, self).__init__()
        self.layers = torch.nn.Sequential(
                        torch.nn.Linear(input_size, hidden_size),
                        torch.nn.Tanh(),
                        torch.nn.Linear(hidden_size, output_size),
                        torch.nn.Sigmoid()
                      )

    def forward(self, x):
        x = self.layers(x)
        return x

def NDD_train(true_samples, fake_samples, discriminator, optimizerD, criterion, n_discriminator = 15):
    for _ in range(n_discriminator):
        optimizerD.zero_grad()
        
        #fake_samples = generator.forward(u, num_samples)
        fake_logits = discriminator(fake_samples.detach())
        true_logits = discriminator(true_samples)
        
        discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))
        #discriminator_loss = torch.mean(torch.log(true_logits)) + torch.mean(torch.log(1 - fake_logits))

        discriminator_loss.backward()
        optimizerD.step()

    return discriminator

def generator_loss(true_samples, fake_samples, DiscriminatorClass, criterion, n_discriminator=15, g=30):
    """Train g discriminators and return the average loss"""
    discriminators = [DiscriminatorClass().to(true_samples.device) for _ in range(g)]
    for d in discriminators:
        optimizerD = torch.optim.Adam(d.parameters())
        NDD_train(true_samples, fake_samples, d, optimizerD, criterion, n_discriminator)
    
    #avg_discriminator = average_discriminators(discriminators, DiscriminatorClass)
    avg_fake_logits = torch.mean(torch.stack([d(fake_samples) for d in discriminators]), dim=0)
    avg_true_logits = torch.mean(torch.stack([d(true_samples) for d in discriminators]), dim=0)
    
    generator_loss = - (criterion(avg_fake_logits, torch.zeros_like(avg_fake_logits, device=true_samples.device)) + criterion(avg_true_logits, torch.ones_like(avg_true_logits, device=true_samples.device)))
    #generator_loss = torch.mean(torch.log(avg_true_logits)) + torch.mean(torch.log(1 - avg_fake_logits))

    return generator_loss