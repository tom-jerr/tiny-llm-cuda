import torch
from torch import nn
from .basics import linear


class EmbeddingLayer(nn.Module):
    """The embedding layer maps one or more tokens (represented as an integer) to one or more vector of dimension embedding_dim.
    Embedding::froward
    weight: vocab_size x embedding_dim
    Input: N.. (tokens)
    Output: N.. x embedding_dim (vectors)

    In the Qwen2 model, the embedding layer can also be used as a linear layer to map the embeddings back to the token space.
    Embedding::as_linear
    weight: vocab_size x embedding_dim
    Input: N.. x embedding_dim
    Output: N.. x vocab_size
    """

    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.Tensor):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x, :]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight)
