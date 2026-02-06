import torch

T = 4

mask = torch.tril(torch.ones(T, T))

print(mask)