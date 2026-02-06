import torch

batch, length, n_dim, n_head = 1, 4, 768, 12
head_dim = int(n_dim / n_head)

# without heads
q = torch.randn(batch, length, n_dim)
print(q.shape)

# reshape with heads
q_reshaped = q.view(batch, length, n_head, head_dim)
print(q_reshaped.shape)

# Transpose to bring heads to the front to isolate them so pytorch can process them in parallel.
q_final = q_reshaped.transpose(1,2)
print(q_final.shape)


