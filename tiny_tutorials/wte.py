import torch
import torch.nn as nn

#config
vocab_size = 10
n_dim = 3

token_table = nn.Embedding(vocab_size, n_dim)

print("Shape: ", token_table.weight.shape)
print("Tokens: ")
print(token_table.weight)