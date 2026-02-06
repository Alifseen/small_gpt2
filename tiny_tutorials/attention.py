import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

batch, length, n_dim = 1, 4, 2

# example 2D embeddings
x = torch.tensor([
    [[0.1,0.1], #A
     [1.0,0.2], #crane
     [0.1,0.9], #ate
     [0.8,0.0]] #fish
]).float()

# Manually set weights for the learnable QKV for this tutorial
q_project = nn.Linear(n_dim, n_dim, bias=False)
k_project = nn.Linear(n_dim, n_dim, bias=False)
v_project = nn.Linear(n_dim, n_dim, bias=False)

torch.manual_seed(42)
q_project.weight.data = torch.randn(n_dim,n_dim)
k_project.weight.data = torch.randn(n_dim,n_dim)
v_project.weight.data = torch.randn(n_dim,n_dim)

q = q_project(x)
k = k_project(x)
v = v_project(x)

# Calculate attention scores. A 4x4 matrix
scores = q @ k.transpose(-2, -1)

# Scale and Softmax
last_dimension_k = k.size(-1)
scaled_scores = scores / math.sqrt(last_dimension_k)
attention_weights = F.softmax(scaled_scores, dim=-1)

# Add V
output = attention_weights @ v


print(output.size())
print(output)