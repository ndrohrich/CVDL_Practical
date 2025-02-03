import torch.nn as nn
import os 
import torch
from models.Blocks.TransformerBlock import TransformerBlock
import torch.nn.functional as F
from timm.models.vision_transformer import Block

class Hybrid(nn.Module):
    def __init__(self, 
                 num_classes=6, 
                 input_channels=1,
                 depth=6, 
                 embed_dim=256,
                 num_heads=3, 
                 channel_multiplier=1, 
                 encoder='standard'
                 ):  
        super(Hybrid, self).__init__()

        match encoder:
            case 'standard':
                self.cnn_encoder = CNN_encoder(input_channels=input_channels, channel_multiplier=channel_multiplier)
            case 'vgg':
                self.cnn_encoder = VGG_encoder(input_channels=1)
        self.attention_module = nn.Sequential(*[TransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(int(depth))])
        self.mlp_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features=embed_dim, out_features=num_classes))
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, embed_dim))
        self.projection_layer = nn.Linear(in_features=64, out_features=embed_dim)
        self.demo = False  

        # Register Forward Hook for feature map given by last transformer block
        self.cnn_encoder.layer8.register_forward_hook(self.forward_hook)

        # Register Backward Hook for gradient 
        self.cnn_encoder.layer8.register_backward_hook(self.backward_hook)

        self.channel_multiplier = channel_multiplier

    # Define forward hook that gets executed for each forward pass
    def forward_hook(self, module, input, output): 
        self.features = output

    # Define backward hook that gets executed for each backward pass
    def backward_hook (self, module, grad_input, grad_output): 
        self.gradients = grad_output[0]

    def forward(self, x, apply_softmax=False):
        # Forward through CNN encoder 
        features = self.cnn_encoder(x)

        # Flatten embeddings for transformer processing
        b, c, h, w = features.shape
        embeddings = torch.reshape(features, [b, c, h*w])

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

        self.logits = output

        if apply_softmax:
            output = F.softmax(output, dim=1)
            
        return output

# CNN Encoder returns outputs of shape [batch, 256, 16, 16] given [batch, 1, 64, 64] input images
class CNN_encoder(nn.Module): 
    def __init__(self, input_channels=1, channel_multiplier=1):
        super().__init__()

        self.channel_multiplier = channel_multiplier

        self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 8*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8*self.channel_multiplier),
                nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8*self.channel_multiplier, 16*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16*self.channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16*self.channel_multiplier, 32*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32*self.channel_multiplier),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32*self.channel_multiplier, 64*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64*self.channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64*self.channel_multiplier, 128*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128*self.channel_multiplier),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128*self.channel_multiplier, 128*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128*self.channel_multiplier),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128*self.channel_multiplier, 256*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*self.channel_multiplier),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256*self.channel_multiplier, 256*self.channel_multiplier, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256*self.channel_multiplier),
            nn.ReLU()
        )
    
    def forward(self, x):
        output = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        return output 

class VGG_encoder(nn.Module): 
    def __init__(self, input_channels):
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
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)

        return x
