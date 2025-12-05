from torch import Tensor

import torch
import torch.nn as nn


class CLIPTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module) -> None:
        super().__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.text_projection = clip_model.text_projection
        self.ln_final = clip_model.ln_final
        self.dtype = clip_model.dtype

    def forward(self, text: Tensor) -> Tensor:
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text(self, text: Tensor) -> Tensor:
        return self.forward(text)
