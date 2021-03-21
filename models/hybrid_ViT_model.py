from einops import repeat
from torch import nn, randn, cat
from torchvision import models
from torchvision.models.resnet import conv1x1

from .ViT_model import MLPHead, TransformerEncoder

class HybridViT(nn.Module):
    """
    This abstract class contains the structure of the Hybrid Vision Transformer.
    Is required the definition of the method get_CNN_backbone in each concrete classes.

    Attributes:
    backbone_model    reference to an object of a class that implements backbone CNN to use for the features extraction
    features_dim      integer that contains the dimension of the feature map extracted
    embedding_layer   reference to an object of class HybridEmbeddingLayer
    transformer       reference to an object of class TransformerEncoder
    mpl_head          reference to an object of class MLPHead
    """

    def __init__(self, image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout=0., downsample_ratio=1):
        super().__init__()
        self.downsample = conv1x1(3, 3, downsample_ratio)
        self.backbone_model, self.features_dim = self.get_CNN_backbone()
        self.embedding_layer = HybridEmbeddingLayer(dim=dim, image_size=image_size, features_dim=self.features_dim)
        self.transformer = TransformerEncoder(dim=dim, depth=depth, num_heads=num_heads,
                                              feedforward_dim=feedforward_dim, dropout=dropout)
        self.mlp_head = MLPHead(dim=dim, n_classes=num_classes)

    def forward(self, x):
        x = self.downsample(x)
        x = self.backbone_model(x)
        x = x.view(x.size(0), -1, self.features_dim)
        x = self.embedding_layer(x)
        x = self.transformer(x)
        x = self.mlp_head(x[:, 0])
        return x

    def get_CNN_backbone(self):
        raise NotImplementedError("Method must be implemented in the child classes!!")

class Resnet50HybridViT(HybridViT):
    """
    This class extends the HybridViT to implements an hybrid ViT that extarct the features from the fourth
    convolutional layer of a resnet50.

    """

    def __init__(self, image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout=0., downsample_ratio=1):
        super().__init__(
             #image_size, num_classes, dim, depth, num_heads, feedforward_dim, dropout
             image_size/downsample_ratio, num_classes, dim, depth, num_heads, feedforward_dim, dropout, downsample_ratio
        )

    def get_CNN_backbone(self):
        model = models.resnet50(pretrained=False)
        modules = list(model.children())[:-3]
        model = nn.Sequential(*modules)
        features_dim = 1024
        return model, features_dim


class HybridEmbeddingLayer(nn.Module):
    """

    This class contains the module of the ViT that generates patch embeddings from the input image.

    Attributes:
    projection        object of nn.Sequential class that maps all feature vector in the embedding dimension
    cls               object of nn.Parameter class that generates cls token to append to all patch embeddings
    num_features      integer that contains the number of the features extracted from the image using backbone
    positions         object of nn.Parameter class that generates the positional encoding vectors to add to all patch
                      embeddings
    """

    def __init__(self, dim, image_size, features_dim):
        super().__init__()
        self.projection = nn.Linear(features_dim, dim)
        self.cls = nn.Parameter(randn(1, 1, dim))
        self.num_features = int(image_size / 16.0) ** 2
        self.positions = nn.Parameter(randn(1, self.num_features + 1, dim))

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.projection(x)
        tokens = repeat(self.cls, '() n e -> b n e', b=batch_size)
        x = cat((tokens, x), dim=1)
        x += self.positions[:, :(x.shape[1] + 1)]
        return x