import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbedding3D(nn.Module):
    def __init__(self, in_channels, patch_shape, emb_size, img_shape):
        # Call parent class
        super().__init__()

        # Class attributes
        self.patch_shape = patch_shape
        self.emb_size = emb_size

        # Embedding
        n_patches = np.prod([img_shape[i] // patch_shape[i] for i in range(3)])
        self.pos_embedding = nn.Parameter(torch.randn(1, n_patches, emb_size))
        self.proj = nn.Conv3d(
            in_channels, emb_size, kernel_size=patch_shape, stride=patch_shape
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.proj(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = x + self.pos_embedding
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, heads, mlp_dim, dropout=0.1):
        # Call parent class
        super().__init__()

        # Attention head
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = nn.MultiheadAttention(
            emb_size, heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT3D(nn.Module):
    def __init__(
        self,
        in_channels=1,
        img_shape=(64, 64, 64),
        patch_shape=(8, 8, 8),
        emb_size=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
    ):
        # Call parent class
        super().__init__()

        # Class attributes
        self.patch_shape = patch_shape
        self.grid_size = [img_shape[i] // patch_shape[i] for i in range(3)]

        # Transformer Layers
        self.patch_embed = PatchEmbedding3D(
            in_channels, patch_shape, emb_size, img_shape
        )
        self.transformer = nn.Sequential(
            *[
                TransformerEncoderBlock(emb_size, heads, mlp_dim)
                for _ in range(depth)
            ]
        )
        self.output_head = nn.Linear(
            emb_size, np.prod(patch_shape) * in_channels
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.output_head(x)
        x = x.view(batch_size, -1, *self.patch_shape)
        x = rearrange(
            x,
            "(b d h w) c pd ph pw -> b c (d pd) (h ph) (w pw)",
            b=batch_size,
            d=self.grid_size[0],
            h=self.grid_size[1],
            w=self.grid_size[2],
            pd=self.patch_shape[0],
            ph=self.patch_shape[1],
            pw=self.patch_shape[2],
        )
        return x
