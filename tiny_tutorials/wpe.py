import torch
import torch.nn as nn

#config
batch, length, n_dim = 2, 5, 3
vocab_size = 10
block_size = 8

token_table = nn.Embedding(vocab_size, n_dim)

position_table = nn.Embedding(block_size, n_dim)

input_index = torch.randint(0, vocab_size, (batch, length)) # 2,5

tok_emb = token_table(input_index)

pos = torch.arange(0, length, dtype=torch.long) # 0,1,2,3,4
pos_emb = position_table(pos)

x = tok_emb + pos_emb

print("Token Embedding Shape: ", tok_emb.shape)
print("Position Embedding Shape: ", pos_emb.shape)
print("Final Embedding Shape: ", x.shape)