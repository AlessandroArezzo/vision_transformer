from einops import repeat
from torch import nn, arange, randn, cat
from torchvision import models
from .ViT_model import MLPHead, TransformerEncoder

class PositionalEncoding(nn.Module):
    def __init__(self, max_pos, dim, seq_length):
        super().__init__()
        self.emb = nn.Embedding(max_pos, dim)
        self.seq_length = seq_length
        self.register_buffer(
            "position_ids",
            arange(max_pos).expand((1, -1)),
        )

    def forward(self, x, pos_ids=None):
        if pos_ids is None:
            pos_ids = self.position_ids[:, : self.seq_length]
        position_embeddings = self.emb(pos_ids)
        return x + position_embeddings

class HybridEmbeddingLayer(nn.Module):
    def __init__(self, dim, image_size, features_dim):
        super().__init__()
        self.projection_encoding = nn.Linear(features_dim, dim)
        self.cls = nn.Parameter(randn(1, 1, dim))
        self.decoder_dim = int(image_size / 16.0) ** 2
        self.decoder_dim += 1
        self.positions = PositionalEncoding(
            self.decoder_dim, dim, self.decoder_dim
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection_encoding(x)
        tokens = repeat(self.cls, '() n e -> b n e', b=batch_size)
        x = cat((tokens, x), dim=1)
        x = self.positions(x)
        return x

class HybridViT(nn.Module):
    def __init__(self, image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout=0.,
                 backbone=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_model, self.features_dim = self.get_CNN_backbone()

        self.emb_layer = HybridEmbeddingLayer(dim=dim, image_size=image_size, features_dim=self.features_dim)
        self.transformer = TransformerEncoder(dim=dim, depth=depth, num_heads=num_heads, feedforward_dim=feedforward_dim,
                                              dropout=dropout)
        self.mlp_head = MLPHead(dim=dim, n_classes=num_classes)

    def forward(self, x):
        x = self.backbone_model(x)
        x = x.view(x.size(0), -1, self.features_dim)
        x = self.emb_layer(x)
        x = self.transformer(x)
        x = self.mlp_head(x)
        return x

class ResnetHybridViT(HybridViT):
    def __init__(self, image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout=0., backbone="resnet50"):
        super().__init__(
             image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout, backbone
        )

    def get_CNN_backbone(self):
        assert self.backbone == "resnet50", "Resnet type error: only resnet50 supported actually"
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-3]
        model = nn.Sequential(*modules)
        for p in model.parameters():
            p.requires_grad = False
        features_dim = 1024
        return model, features_dim