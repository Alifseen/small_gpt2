import torch

x_initial = torch.tensor([[[0.2, 0.1, 0.3, 0.4]]])

attention_output = torch.tensor([[[0.1, -0.1, 0.2, -0.3]]])

x_after_attn = x_initial + attention_output

print(x_initial)
print(attention_output)
print(x_after_attn)