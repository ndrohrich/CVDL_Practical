import torch
import torch.nn as nn
import numpy as np
from Blocks import TransformerBlock
from timm.models.vision_transformer import Block

'''
Here, we implement a simple Transformer Decoder for MAE pretraining
(which actually consists of Encoder Blocks in the nomenclature of the original Transformer paper)
from scratch, according to https://arxiv.org/abs/2111.06377
'''

class TransformerEncoder(nn.Module): 
    def __init__(self,
                 depth = 2, 
                 embed_dim = 192, 
                 num_heads = 3, 
                 image_size = 128,
                 num_channels = 1, 
                 patch_size = 16): 
        
        '''
        Performs Module initialization.  

        Parameters
        ----------
        depth: int
            Specifies the number of Transformer Blocks. 
        embed_dim: int 
            Specifies the dimensionality of the tokens processed by Transformer Blocks. 
        num_head: int
            Number of self-attention heads per Transformer Block. 
        image_size: int     
            Image size. Note we are using quadratic images only and the root of image_size has to be an integer. 
        num_channels: int 
            Number of input channels. We are using grayscale, i.e. 1 channel.
        patch_size: int 
            Size of quadratic patches. 
        '''

        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = (image_size / patch_size) ** 2

        # Init ViT layers 
        self.transformer_blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads) for _ in range(int(self.depth))])
        self.linear_projection = nn.Linear(in_features=embed_dim,
                                           out_features=num_channels*patch_size*patch_size)
        

    def forward(self, input_image):

        '''
        Performs forward pass for a given input image. 

        Parameters
        ----------
        input_image: torch.Tensor 
            Tensor of shape [C, H, W] where C=1 is channel dimension
        '''

        # Forward Embeddings through Transformer Blocks 
        embeddings = self.transformer_blocks(embeddings)

        # Eliminate [cls] token 
        embeddings = embeddings[:, :, 1:, :]

        # Linearly project tokens to image patch dimension
        patches_flattened = self.linear_projection(embeddings)

        # Reshape to 2D image 
        B = patches_flattened.shape[0]
        patches_2D = torch.reshape(patches_flattened, (B, self.num_channels, self.patch_size, self.patch_size))

        
        # Patchify Image 
        # [B, C, H, W] -> [B, num_patches, C, patch_size, patch_size]
        patches = torch.reshape(input=input_image,
                                shape=(input_image.shape[0], int(self.num_patches), int(self.patch_size), int(self.patch_size)))

        # Flatten Image 
        # [num_patches, C, patch_size, patch_size] -> [num_patches, C * patch_size ** 2]
        flattened_patches = torch.flatten(patches, start_dim=-2, end_dim=-1)

        # Linear Projection to Embedding Space 
        # [num_patches, C * patch_size ** 2] -> [num_patches, embed_dim]
        embeddings = self.linear_projection(flattened_patches)

        # Concat Embeddings with [cls_token] 
        # [num_patches, embed_dim] -> [num_patches+1, embed_dim]
        cls_tokens = self.cls_token.repeat([embeddings.shape[0],1,1])
        embeddings = torch.concat([cls_tokens, embeddings], dim=1)

        

        return embeddings