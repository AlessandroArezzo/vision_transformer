from torch import cat, randn, einsum
from torch import nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch.nn.functional import pad

class ViT(nn.Module):
    """
    This class contains the Vision Transformer.

    Attributes:
    embedding_layer   reference to an object of class EmbeddingLayer
    transformer       reference to an object of class TransformerEncoder
    mpl_head          reference to an object of class MLPHead
    """

    def __init__(self, image_size, patch_size, num_classes, dim, depth, num_heads, feedforward_dim, channels=3,
                 dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, "Patch size dimension error"
        num_patches = (image_size // patch_size) ** 2
        self.embedding_layer = EmbeddingLayer(patch_size=patch_size, channels=channels, dim=dim,
                                              num_patches=num_patches)
        self.transformer = TransformerEncoder(dim=dim, depth=depth, num_heads=num_heads,
                                              feedforward_dim=feedforward_dim, dropout=dropout)
        self.mlp_head = MLPHead(dim=dim, n_classes=num_classes)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = self.transformer(x)
        return self.mlp_head(x[:, 0])

class EmbeddingLayer(nn.Module):
    """
    This class contains the module of the ViT that generates patch embeddings from the input image.

    Attributes:
    projection        object of nn.Sequential class that splits the image into patches
                      and generates projections in the size of the embedding
    cls               object of nn.Parameter class that generates cls token to append to all patch embeddings
    positions         object of nn.Parameter class that generates the positional encoding vectors to add to all patch
                      embeddings
    """
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
        tokens = repeat(self.cls, '() n e -> b n e', b=batch_size)
        x = cat((tokens, x), dim=1)
        x += self.positions[:, :(x.shape[1] + 1)]
        return x

class TransformerEncoder(nn.Sequential):
    """
    This class represents the TransformerEncoder block. It is is composed of a transformer encoder blocks sequence
    which are references to objects of the TransformerEncoderBlock class.

    """
    def __init__(self, dim, num_heads, feedforward_dim, dropout, depth):
        super().__init__(*[TransformerEncoderBlock(dim=dim, num_heads=num_heads,
                                            feedforward_dim=feedforward_dim, dropout=dropout) for _ in range(depth)])

class MLPHead(nn.Sequential):
    """
    This class contains the module of the ViT that predicts the image class. It consists in the sequential execution of
    an identity layer, a normalization layer and a linear layer that maps the first output of the TransformerEncoder
    in a logit vector of the same length than the number of classes.

    """
    def __init__(self, dim, n_classes):
        super().__init__(
            nn.Identity(),
            nn.LayerNorm(dim),
            nn.Linear(dim, n_classes))

class TransformerEncoderBlock(nn.Sequential):
    """
    This class implements the primitive encoder block of the ViT architecture. This consists in applying the mechanism
    of self attention (MultiheadAttention) and running the feedforward network (MLP class) on its result.
    Normalization layers (Norm) and residual connections (ResidualConnection) are performed before and after
    each of these elements, respectively.

    """
    def __init__(self, dim, num_heads, feedforward_dim, dropout=0.):
        super().__init__(
            ResidualConnection(Norm(dim, MultiheadAttention(dim=dim, n_heads=num_heads, dropout=dropout))),
            ResidualConnection(Norm(dim, MLP(dim=dim, mlp_size=feedforward_dim, dropout=dropout)))
        )

class MultiheadAttention(nn.Module):
    """
    This class contains implementation of the self-attention mechanism.

    Attributes:
    n_heads           number of the self-attention heads
    to_qkv            an object of nn.Linear class that project the input in the query, key and value vector,
                      respectively
    scale             float value that contains the normalization factor to apply
    to_out            an object of nn.Sequential class that project the output of the self-attention module in
                      the encoder dimension and apply a dropout if required
    """

    def __init__(self, dim, n_heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.n_heads = n_heads
        inner_dim = dim_head * n_heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.n_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        query, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('bhid,bhjd->bhij', query, keys) * self.scale
        scores = dots.softmax(dim=-1)
        out = einsum('bhij,bhjd->bhid', scores, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Norm(nn.Module):
    """
    This class implements the pre-norm operation to execute before each element in a transformer encoder block.

    Attributes:
    next_block        the next block to be executed after the normalization
    norm              an object of nn.LayerNorm class
    """
    def __init__(self, dim, next_block):
        super().__init__()
        self.next_block = next_block
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **next_block_args):
        return self.next_block(self.norm(x), **next_block_args)

class ResidualConnection(nn.Module):
    """
    This class contains the residual connection mechanism to execute after each element in a transformer encoder block.

    Attributes:
    previous_block     the previous block to be executed before the residual connection
    """

    def __init__(self, previous_block):
        super().__init__()
        self.previous_block = previous_block

    def forward(self, x, **previous_block_args):
        residual = x
        x = self.previous_block(x, **previous_block_args)
        return x + residual

class MLP(nn.Sequential):
    """
    This class contains the feedforward network that must be executed after the self-attention mechanism in each
    transformer encoder block. This consists in the sequential execution of two linear layers and a GELU
    as activation function. Dropout is performed after each fully connected layer if required.

    """
    def __init__(self, dim, mlp_size, dropout):
        super().__init__(
            nn.Linear(dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, dim),
            nn.Dropout(dropout)
        )
