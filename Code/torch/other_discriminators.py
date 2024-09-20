import torch
from tqdm import tqdm

from roy import royinv

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
    
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
    lr_model = LogisticRegression(input_size=7).to(device)
    optimizer = torch.optim.Adam(lr_model.parameters())
    criterion = torch.nn.BCELoss()
    for _ in tqdm(range(10000)):
        optimizer.zero_grad()
        logits = lr_model.forward(moments).squeeze()
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

    # Extract coefficients (lambda in MATLAB code)
    coefficients = torch.concatenate([lr_model.linear.bias, lr_model.linear.weight[0]])

    # Calculate loss
    v = criterion(lr_model.forward(moments).squeeze(), Y)

    return v, coefficients

# Testing
u_1 = torch.rand(100, 4)
u_2 = torch.rand(100, 4)
theta1 = torch.tensor([1.8, 2, 0.5, 0, 1, 1, 0.5, 0])
theta2 = torch.tensor([1.9, 2.1, 0.6, 0.1, 1.1, 1.1, 0.6, 0.1])
X_1 = royinv(u_1, theta1)
X_2 = royinv(u_2, theta2)
print(logistic_loss2(X_1, X_2))