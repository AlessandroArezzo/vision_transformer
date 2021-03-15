from torch import cat, randn, einsum
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch.nn.functional import pad


class EmbeddingLayer(nn.Module):
    def __init__(self, patch_size, dim, num_patches, channels=3):
        super().__init__()
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(channels * patch_size ** 2, dim),
        )
        self.cls = nn.Parameter(randn(1, 1, dim))
        self.positions = nn.Parameter(randn(1, num_patches + 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        tokens = repeat(self.cls, '() n e -> b n e', b=batch_size) # genera cls tokens per tutti gli elementi del batch size
        x = cat((tokens, x), dim=1) # concatena cls tokens alle proiezioni delle patches
        x += self.positions[:, :(x.shape[1] + 1)] # position embeddings aggiunti alle proiezioni
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
        return self.next_block(self.norm(x), **next_block_args)

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
            ResidualConnection(Norm(dim, MultiheadAttention(dim=dim, n_heads=num_heads, dropout=dropout))),
            ResidualConnection(Norm(dim, MLP(dim=dim, mlp_size=feedforward_dim, dropout=dropout)))
        )

class MultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.n_heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class TransformerEncoder(nn.Sequential):
    def __init__(self, dim, num_heads, feedforward_dim, dropout, depth):
        super().__init__(*[TransformerEncoderBlock(dim=dim, num_heads=num_heads,
                                                feedforward_dim=feedforward_dim, dropout=dropout) for _ in range(depth)])

class MLPHead(nn.Sequential):
    def __init__(self, dim, n_classes):
        super().__init__(
            nn.Identity(),
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes))

class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, num_heads, feedforward_dim, channels=3, dropout=0.):
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        super().__init__()
        self.embedding_layer = EmbeddingLayer(patch_size=patch_size, channels=channels, dim=dim, num_patches=num_patches)
        self.transformer = TransformerEncoder(dim=dim, depth=depth, num_heads=num_heads, feedforward_dim=feedforward_dim, dropout=dropout)
        self.mlp_head = MLPHead(dim=dim, n_classes=num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

