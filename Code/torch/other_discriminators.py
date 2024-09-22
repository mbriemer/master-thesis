import torch
from torch.distributions import TransformedDistribution, Uniform
from torch.distributions.transforms import SigmoidTransform, AffineTransform
#from tqdm import tqdm
import numpy as np
from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression


from roy import royinv, royinv_sp

class LogisticRegression_net(torch.nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.linear(x)
    
def logistic_loss2(X_1, X_2):
    """Functions "loss" from main_roy.m and "loss1" from main_case.m"""
    device = X_1[0].device
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1[:, 0], X_1[:, 1], X_1[:, 2], X_1[:, 3]
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2[:, 0], X_2[:, 1], X_2[:, 2], X_2[:, 3]
    n = len(d_1_1)
    m = len(d_1_2)
    assert n == m, "n == m required for logistic regression loss"

    Y = torch.concatenate([torch.ones(n, device=device), torch.zeros(m, device=device)])

    moments_1 = torch.column_stack([log_w_1_1, d_1_1, log_w_2_1, d_2_1, log_w_1_1**2, log_w_2_1**2, log_w_1_1 * log_w_2_1])
    moments_2 = torch.column_stack([log_w_1_2, d_1_2, log_w_2_2, d_2_2, log_w_1_2**2, log_w_2_2**2, log_w_1_2 * log_w_2_2])
    moments = torch.vstack([moments_1, moments_2])

    # Fit logistic regression model
    lr_model = LogisticRegression_net(input_size=7).to(device)
    optimizer = torch.optim.Adam(lr_model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    for _ in range(10000):
        optimizer.zero_grad()
        logits = lr_model.forward(moments).squeeze()
        loss = criterion(logits, Y)
        loss.backward(retain_graph=True)
        optimizer.step()

    # Extract coefficients (lambda in MATLAB code)
    coefficients = torch.cat([
        lr_model.linear.bias.view(-1, 1),
        lr_model.linear.weight[0].view(-1, 1)
    ], dim=0)

    # Set up logistic distribution
    base_distribution = Uniform(0, 1)
    transforms = [SigmoidTransform().inv, AffineTransform(loc=0, scale=1)]
    logistic = TransformedDistribution(base_distribution, transforms)
    
    # Calculate loss
    X1 = torch.cat((
        torch.ones((n, 1), device=device),
        moments_1
    ), dim=1)

    X2 = torch.cat((
        torch.ones((m, 1), device=device),
        moments_2
    ), dim=1)

    v = torch.mean(torch.log(logistic.cdf(X1 @ coefficients))) +\
        torch.mean(torch.log(logistic.cdf(- (X2 @ coefficients))))

    return v, coefficients

def logistic_loss3(X_1, X_2):
    """Function "loss2" from main_case.m"""
    log_w_1_1, d_1_1, log_w_2_1, d_2_1 = X_1
    log_w_1_2, d_1_2, log_w_2_2, d_2_2 = X_2
    n = len(d_1_1)
    m = len(d_1_2)
    assert n == m, "n == m required for logistic regression loss"

    Y = np.concatenate([np.ones(n), 0 * np.ones(m)])

    moments_1 = np.column_stack([log_w_1_1, d_1_1, log_w_2_1, d_2_1, log_w_1_1**2, log_w_2_1**2, log_w_1_1 * log_w_2_1])
    moments_2 = np.column_stack([log_w_1_2, d_1_2, log_w_2_2, d_2_2, log_w_1_2**2, log_w_2_2**2, log_w_1_2 * log_w_2_2])

    # Fit logistic regression model
    lr_model = LogisticRegression(fit_intercept=True)
    lr_model.fit(np.vstack([moments_1, moments_2]), Y)

    # Extract coefficients (lambda in MATLAB code)
    coefficients = np.concatenate([lr_model.intercept_, lr_model.coef_[0]])

    # Calculate loss
    v = np.mean(logistic.logcdf(np.dot(np.column_stack([np.ones(n), moments_1]), coefficients))) + \
        np.mean(logistic.logcdf(-np.dot(np.column_stack([np.ones(m), moments_2]), coefficients)))
                
    return v, coefficients


""" # Testing
u_1 = torch.rand(100, 4)
u_2 = torch.rand(100, 4)
theta1 = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0])
theta2 = torch.tensor([1.9, 2.1, 0.6, 0.1, 1.1, 1.1, 0.6, 0.1])
X_1 = royinv(u_1, theta1)
X_2 = royinv(u_2, theta2)
print(logistic_loss2(X_1, X_2)) """