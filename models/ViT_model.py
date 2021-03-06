from torch import cat, randn
from torch.nn import MultiheadAttention
from einops.layers.torch import Reduce
from torch import nn
from einops import repeat
from einops.layers.torch import Rearrange

class EmbeddingLayer(nn.Module):
    def __init__(self, patch_size, channels, dim, num_patches):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls = nn.Parameter(randn(1, 1, dim))
        self.positions = nn.Parameter(randn(num_patches + 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        tokens = repeat(self.cls, '() n e -> b n e', b=batch_size) # genera cls tokens per tutti gli elementi del batch size
        x = cat([tokens, x], dim=1) # concatena cls tokens alle proiezioni delle patches
        x += self.positions  # position embeddings aggiunti alle proiezioni
        return x

class MLP(nn.Sequential):
    def __init__(self, dim, mlp_size, dropout):
        super().__init__(
            nn.Linear(dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, dim),
            nn.Dropout(dropout)
        )

class Norm(nn.Module):
    def __init__(self, dim, next_block):
        super().__init__()
        self.next_block = next_block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **next_block_args):
        if isinstance(self.next_block, MultiheadAttention):
            x = self.norm(x)
            x = self.next_block(x, x, x)[0]
        else:
            x = self.next_block(self.norm(x), **next_block_args)
        return x

class ResidualConnection(nn.Module):
    def __init__(self, previous_block):
        super().__init__()
        self.previous_block = previous_block

    def forward(self, x, **previous_block_args):
        residual = x
        x = self.previous_block(x, **previous_block_args)
        return x + residual

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, dim, num_heads, feedforward_dim, dropout=0.):
        super().__init__(
            ResidualConnection(Norm(dim, MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout))),
            ResidualConnection(Norm(dim, MLP(dim=dim, mlp_size=feedforward_dim, dropout=dropout)))
        )

class TransformerEncoder(nn.Sequential):
    def __init__(self, dim, num_heads, feedforward_dim, dropout, depth):
        super().__init__(*[TransformerEncoderBlock(dim=dim, num_heads=num_heads,
                                                feedforward_dim=feedforward_dim, dropout=dropout) for _ in range(depth)])

class MLPHead(nn.Sequential):
    def __init__(self, dim, n_classes):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes))

class ViT(nn.Sequential):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, num_heads, feedforward_dim, channels=3, dropout=0.):
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        super().__init__(
            EmbeddingLayer(patch_size=patch_size, channels=channels, dim=dim, num_patches=num_patches),
            TransformerEncoder(dim=dim, depth=depth, num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout),
            MLPHead(dim=dim, n_classes=num_classes)
        )
