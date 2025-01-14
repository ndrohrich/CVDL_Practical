import torch
import torch.nn as nn
import torch.nn.functional as F

# definition of Attention feature map

#defining the attention block
class alighnment_score(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(alighnment_score,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.fc_up = nn.Linear(in_channels,2*out_channels)
        self.fc_down = nn.Linear(2*out_channels,out_channels)
        self.layernorm = nn.LayerNorm(out_channels)
        self.Q=nn.Linear(in_channels,out_channels)
        self.K=nn.Linear(in_channels,out_channels)
    def forward(self,x):
        origin=x
        x=self.layernorm(x)
        Q=self.Q(x)
        K=self.K(x)
        alignment=torch.matmul(Q,K.transpose(-2,-1))/self.out_channels**0.5
        alighnment_score=F.softmax(alignment,dim=-1)
        alighnment_score=torch.matmul(alighnment_score,origin)
        #print(alighnment_score.shape)
        alighnment_score=alighnment_score+origin
        x=self.layernorm(alighnment_score)
        x=F.relu(x)
        return alighnment_score    
    
    # definine the convolution init layer
class window_init(nn.Module):
    def __init__(self,in_channels):
        super(window_init,self).__init__()
        self.in_channels=in_channels
        self.fc=nn.Linear(in_channels,in_channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.fc(x)
        x=self.relu(x)
        return x
    
class after_attention(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(after_attention,self).__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.fc=nn.Linear(in_channels,out_channels)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=x.view(x.size(0),-1)
        feature=x
        x=self.fc(x)
        x=self.relu(x)
        return feature,x

# defining the Attention feature cluster network
class AttentionFeatureCluster(nn.Module):
    def __init__(self,patch_size,feature_size,num_classes):
        super(AttentionFeatureCluster,self).__init__()
        self.patch_size=patch_size
        self.feature_size=feature_size
        self.num_classes=num_classes
        self.num_patches=(self.feature_size//self.patch_size)**2
        self.window_init=nn.ParameterList([window_init(self.patch_size**2) for _ in range(self.num_patches)])
        self.attention=alighnment_score(self.patch_size**2,self.patch_size**2)
        
        self.after_attention=after_attention(self.feature_size**2,self.num_classes)
        self.maxpool=nn.MaxPool2d(2)
    def forward(self,x):
        b,c,h,w=x.size()
        original=x
        
        # split the image into patches
        x=x.view(b,c,h//self.patch_size,self.patch_size,w//self.patch_size,self.patch_size)
        x=x.permute(0,2,4,1,3,5)
        x=x.contiguous().view(b,self.num_patches,c,self.patch_size,self.patch_size)
        #print(x.shape)
        #plt.imshow(x[0,0].detach().cpu().numpy())
        
        # apply the window init layer
        x=[self.window_init[i](x[:,i].view(b,-1)) for i in range(self.num_patches)]
        x=torch.stack(x,dim=1)
        x=x+original
        #print(x.shape)
        
        # apply the attention block
        x=self.attention(x)
        #print(x.shape)
        
        # reshape the output
        x=x.view(b,self.num_patches,c,self.patch_size,self.patch_size)
        #print(x.shape)
        #resample the patches into the original image
        x=x.permute(0,3,1,4,2)
        x=x.contiguous().view(b,c,h,w)
        #print(x.shape)
        #plt.imshow(x[0,0].detach().cpu().numpy())
        
        # apply the fc layer
        feature,x=self.after_attention(x)
        return feature,x
    