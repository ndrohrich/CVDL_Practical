import torch.nn as nn
import os 
import torch
from timm.models.vision_transformer import Block

class Hybrid(nn.Module):
    def __init__(self, 
                 num_classes=6, 
                 input_channels=1,
                 depth=6, 
                 embed_dim=256,
                 num_heads=3
                 ):  
        super(Hybrid, self).__init__()
    
        self.cnn_encoder = CNN_encoder(input_channels=input_channels)
        self.attention_module = self.transformer_blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads) for _ in range(int(depth))])
        self.mlp_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features=embed_dim, out_features=num_classes))
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, embed_dim))
        self.projection_layer = nn.Linear(in_features=256, out_features=embed_dim)

        
    def forward(self, x):
        # Forward through CNN encoder 
        embeddings = self.cnn_encoder(x)

        # Flatten embeddings for transformer processing
        b, c, h, w = embeddings.shape
        embeddings = torch.reshape(embeddings, [b, c, h*w])

        # Project to embedding dim 
        embeddings = self.projection_layer(embeddings)

        # Append cls_tokens
        cls_tokens = self.cls_token.repeat([embeddings.shape[0],1,1])
        embeddings = torch.concat([cls_tokens, embeddings], dim=1)

        # Forward through attention module 
        embeddings = self.attention_module(embeddings)

        # Get cls_tokens
        cls_tokens = embeddings[:, 0, :] 

        # Forward through MLP
        output = self.mlp_head(cls_tokens)
        
        return output 

# CNN Encoder returns outputs of shape [batch, 256, 16, 16] given [batch, 1, 64, 64] input images
class CNN_encoder(nn.Module): 
    def __init__(self, input_channels=1):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    def forward(self, x):
        output = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return output 
    