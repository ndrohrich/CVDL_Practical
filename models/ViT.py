import torch.nn as nn
from models.Blocks import Transformer_Encoder

'''
Here, we re-implement Vision Transformer from scratch, 
according to https://arxiv.org/abs/2010.11929 
'''

class VisionTransformer(nn.Module): 
    def __init__(self,
                 num_classes = 6, 
                 depth = 6, 
                 embed_dim = 192, 
                 num_heads = 3, 
                 image_size = 128,
                 num_channels = 1, 
                 patch_size = 16,
                 pretrained_encoder = None): 
        
        '''
        Performs Module initialization.  

        Parameters
        ----------
        num_classes: int
            Number of distinct emotions classified by the model.
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
        pretrained_encoder: nn.Module (optional)
            Transformer Encoder from MAE pretraining.
        '''

        super().__init__()
        self.num_classes = num_classes
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.image_size = image_size
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.num_patches = (image_size / patch_size) ** 2

        # Init transformer encoder
        if pretrained_encoder == None: 
            self.encoder = Transformer_Encoder.TransformerEncoder(depth=depth, 
                                                                embed_dim=embed_dim,
                                                                num_heads=num_heads,
                                                                image_size=image_size, 
                                                                num_channels=num_channels, 
                                                                patch_size=patch_size) 
        else:
            self.encoder = pretrained_encoder
        
        # Init classification head 
        self.mlp_head = nn.Sequential(nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim), 
                                      nn.ReLU(), 
                                      nn.Linear(in_features=self.embed_dim, out_features=self.num_classes))
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_image):

        '''
        Performs forward pass for a given input image. 

        Parameters
        ----------
        input_image: torch.Tensor 
            Tensor of shape [C, H, W] where C=1 is channel dimension
        '''
        
        # Get embeddings from transformer encoder 
        embeddings = self.encoder(input_image)

        # Get [cls_token]
        cls_tokens = embeddings[:, 0, :]

        # Apply Classification Head 
        logits = self.mlp_head(cls_tokens)

        return logits