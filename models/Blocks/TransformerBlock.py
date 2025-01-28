import torch
from timm.models.vision_transformer import Block
from timm.layers.mlp import Mlp
import torch.nn as nn
from typing import Optional
from timm.layers.config import use_fused_attn


'''
Here, we implement Transformer Encoder blocks as required for ViT, 
according to https://arxiv.org/abs/1706.03762
'''


class Attention_for_CAM(nn.Module):

    ''' Following timm.models.vision_transformer.Attention,
        added code for saving sigmoidal feature maps and gradients.'''

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fused_attn = use_fused_attn()

        # Added properties for CAM computation
        self.sigmoid = nn.functional.sigmoid
        self.gradients = None
        self.attention_scores = None 

    def save_gradients(self, grad):
        """Hook to save gradients of attention scores."""
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Save attention output 
        self.attention_scores = self.sigmoid(attn)

        attn = attn.softmax(dim=-1).requires_grad_()

        # Register backward hook
        attn.register_hook(self.save_gradients)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(Block):
    """
    A custom wrapper for the timm's Vision Transformer Block
    to extract and store attention scores during the forward pass.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ):
        super().__init__(dim, 
                         num_heads, 
                         mlp_ratio, 
                         qkv_bias,
                         qk_norm,
                         proj_drop,
                         attn_drop,
                         init_values,
                         drop_path,
                         act_layer,
                         norm_layer,
                         mlp_layer)
        
        self.attn = Attention_for_CAM(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.attention_scores = None
        self.gradients = None

    def forward(self, x):
        # Call the parent Block forward
        out = super().forward(x)
    
        # Save the feature maps and gradients for visualization
        self.attention_scores = self.attn.attention_scores
        self.gradients = self.attn.gradients
        return out
