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
                 num_heads=3
                 ):  
        super(Hybrid, self).__init__()
    
        self.cnn_encoder = CNN_encoder(input_channels=input_channels)
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

    # Define forward hook that gets executed for each forward pass
    def forward_hook(self, module, input, output): 
        self.features = output

    # Define backward hook that gets executed for each backward pass
    def backward_hook (self, module, grad_input, grad_output): 
        self.gradients = grad_output[0]

    def forward(self, x):
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

        '''if self.demo: 
            # Clear out gradients 
            self.zero_grad()

            # Perform backward on most likely class logit 
            target_class = torch.argmax(output)
            target_class_logit = output[:, target_class]
            target_class_logit.backward(retain_graph=True)

            # Compute CAM
            weights = self.gradients.mean(dim=(2, 3))
            cam = torch.sum(weights[:, :, None, None] * features, dim=1) 
            cam = F.relu(cam) '''

        self.logits = output
        return output

# CNN Encoder returns outputs of shape [batch, 256, 16, 16] given [batch, 1, 64, 64] input images
class CNN_encoder(nn.Module): 
    def __init__(self, input_channels=1):
        super().__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(input_channels, 8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
    
    def forward(self, x):
        output = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        return output 

