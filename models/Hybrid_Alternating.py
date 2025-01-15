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

        # First Stack of CNN feature extractors
        self.cnn_1 = CNN_Module(input_channels=input_channels, output_channels=16)
        self.cnn_2 = CNN_Module(input_channels=16, output_channels=64)

        # First Self-Attention layer applied to intermediate features 
        self.attention_1 = Block(dim=256, num_heads=1)

        # Second Stack of CNN feature extractors 
        self.cnn_3 = CNN_Module(input_channels=64, output_channels=128)
        self.cnn_4 = CNN_Module(input_channels=128, output_channels=256)

        # Second Self-Attention layer applied to final features 
        self.attention_2 = Block(dim=16, num_heads=1)
        self.cls_token = nn.Parameter(data=torch.randn(1, 1, 16))

        # Multilayer Perceptron Classification Head 
        self.mlp_head = nn.Sequential(nn.Linear(in_features=16, out_features=32), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features=32, out_features=num_classes))

        
    def forward(self, x):
        # Forward through first CNN feature extractor  
        embeddings = self.cnn_2(self.cnn_1(x))

        # Flatten embeddings for transformer processing
        b, c, h, w = embeddings.shape
        embeddings = torch.reshape(embeddings, [b, c, h*w])

        # Forward through attention module 
        embeddings = self.attention_1(embeddings)

        # Reshape to 2D 
        b, c, hw = embeddings.shape
        embeddings = torch.reshape(embeddings, [b, c, int(hw ** 0.5), int(hw ** 0.5)])

        # Forward through second CNN feature extractor  
        embeddings = self.cnn_4(self.cnn_3(embeddings))

        # Flatten embeddings for transformer processing
        b, c, h, w = embeddings.shape
        embeddings = torch.reshape(embeddings, [b, c, h*w])

        # Append cls_tokens
        cls_tokens = self.cls_token.repeat([embeddings.shape[0],1,1])
        embeddings = torch.concat([cls_tokens, embeddings], dim=1)

        # Forward through attention module 
        embeddings = self.attention_2(embeddings)

        # Get cls_tokens
        cls_tokens = embeddings[:, 0, :] 

        # Forward through MLP
        output = self.mlp_head(cls_tokens)
        
        return output 

class CNN_Module(nn.Module): 
    def __init__(self, input_channels=1, output_channels=16):
        super().__init__()
        intermediate_out = 8 if input_channels==1 else 2*input_channels
        self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, intermediate_out, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(intermediate_out),
                nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(intermediate_out, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        output = self.layer2(self.layer1(x))
        return output 
    