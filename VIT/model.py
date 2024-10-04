import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, drop_out):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size),
            nn.Flatten(2)
        )

        self.cls_token = nn.Parameter(torch.randn(size=(1,1,embed_dim)), requires_grad= True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1,num_patches+1,embed_dim)), requires_grad= True)
        self.dropout = nn.Dropout(p = drop_out)

def forword(self, x):
    cls_token = self.cls_token.expand(x.shape[0],-1,-1)

    x = self.patcher(x).permute(0,2,1)
    x = torch.cat((x, self.cls_token), dim=1)
    x = x + self.position_embedding
    x = self.dropout(x)
    return x


class VIT(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches, drop_out,
                 num_heads, activation, num_encoders, num_classes):
        super(VIT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, num_patches,drop_out)
        encoder_layer = nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dropout = drop_out,
                                                   activation = activation,
                                                   batch_first = True,
                                                   norm_first = True)
        self.encoder_layers = nn.TransformerEncoder(encoder_layer, num_layers =num_encoders)
        self.MLP = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(embed_dim,num_classes)
        )

    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.encoder_layers(x)
        x = self.MLP(x[:, 0, :])
        return x





