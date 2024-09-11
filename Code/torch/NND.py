# Code for the neural network discriminator

import torch

class Discriminator_paper(torch.nn.Module): #flat net like in the paper

    def __init__(self, input_size=4, hidden_size=10, output_size=1):
        super(Discriminator_paper, self).__init__()
        self.hidden = torch.nn.Linear(input_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.output(self.hidden(x))
        return x

def average_discriminators(discriminators, DiscriminatorClass):
    g = len(discriminators)
    avg_discriminator = DiscriminatorClass()
    
    with torch.no_grad():
        for param_name, param in avg_discriminator.named_parameters():
            avg_param = torch.zeros_like(param)
            
            for d in discriminators:
                avg_param += dict(d.named_parameters())[param_name]
            
            avg_param /= g
            param.copy_(avg_param)
    
    return avg_discriminator

def NDD_train(true_samples, fake_samples, discriminator, optimizerD, criterion, n_discriminator):
    for _ in range(n_discriminator):
        optimizerD.zero_grad()
        
        #fake_samples = generator.forward(u, num_samples)
        fake_logits = discriminator(fake_samples.detach())
        true_logits = discriminator(true_samples)
        
        discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))
        
        discriminator_loss.backward()
        optimizerD.step()

    return discriminator

def NDD_loss(true_samples, fake_samples, DiscriminatorClass, optimizerD, criterion, n_discriminator, g):
    """Train g discriminators and return the average loss"""
    discriminators = [DiscriminatorClass() for _ in range(g)]
    for d in discriminators:
        NDD_train(true_samples, fake_samples, d, optimizerD, criterion, n_discriminator)
    
    avg_discriminator = average_discriminators(discriminators, DiscriminatorClass)
    fake_logits = avg_discriminator(fake_samples)
    true_logits = avg_discriminator(true_samples)
    discriminator_loss = criterion(fake_logits, torch.zeros_like(fake_logits)) + criterion(true_logits, torch.ones_like(true_logits))

    return discriminator_loss