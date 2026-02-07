from dataclasses import dataclass


@dataclass
class GPTSettings:
    vocab_size: int
    block_size: int
    n_layer: int = 12
    n_head: int = 12
    n_dim: int = 768
    dropout: float = 0.1
