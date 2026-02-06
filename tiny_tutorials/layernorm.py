import torch
import torch.nn as nn

# Calculate mean and std
x_token = torch.tensor([[[0.3, -0.2, 0.8, 0.5]]])
mean = x_token.mean(dim=-1, keepdim=True)
std = x_token.std(dim=-1, keepdim=True)

print(mean)
print(std)

# Normalize the vector
epsilon = 1e-5
x_hat = (x_token - mean) / torch.sqrt(std**2 + epsilon)

print(x_hat.mean(dim=-1, keepdim=True))
print(x_hat.std(dim=-1, keepdim=True))

# Apply learnable parameters
# dim = 4
# ln = nn.LayerNorm(dim) # Gamma is weight, and bias is beta
# manually set weights
gamma = torch.tensor([1.5, 1.0, 1.0, 1.0])
beta = torch.tensor([0.5, 0.0, 0.0, 0.0])
y = gamma * x_hat + beta

print(x_hat)
print(y)

