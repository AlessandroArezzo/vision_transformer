from torch import nn, cat, randn
from einops import repeat
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops.layers.torch import Rearrange, Reduce

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

class ClassificationHead(nn.Sequential):
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
            TransformerEncoder(TransformerEncoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=feedforward_dim,
                                                       dropout=dropout, activation="gelu"), depth),
            ClassificationHead(dim, num_classes)
        )


