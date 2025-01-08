import torch
import torch.nn as nn
from torchvision import transforms as tf

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # Fixed issue with incorrect input
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        #print(out.shape)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, in_chanel=1,feature_dim=128, output_dim=256):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_fc = nn.Linear(512 * block.expansion, feature_dim)
        self.feature0_fc = nn.Linear(16*16, feature_dim)
        self.output_fc = nn.Linear(feature_dim, output_dim)
        
        self.initfc=nn.Linear(64*64,64*64)
        self.dropout=nn.Dropout(0.2)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"input image shape:{x.shape}")
        
        # x=x.view(x.size(0),64*64)
        # x=self.initfc(x)
        # self.features0=x
        # x=x.view(x.size(0),1,64,64)
        input=x
        x = self.conv1(x)

       
        
        x = self.bn1(x)
        x = self.relu(x)
        
        # self.features0 = x.mean(dim=1).view(x.size(0), -1)
        ending=x
       
        #x=x+tf.Resize((32,32))(input)
        self.features0 = x.mean(dim=1).view(x.size(0), -1)
        # x = self.avgpool(x)
        # print(f"aft avgpool: {x.shape}")
        x = self.dropout(x)

        
     
        # print(f"features0 shape: {features0.shape}")
    


        x = self.layer1(x)
        
        # drop out
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.maxpool(x)
        x=self.avgpool(x)
        x = self.dropout(x)

        
        x = torch.flatten(x, 1)
        features = self.feature_fc(x)
        output = self.output_fc(features)
        output = torch.nn.functional.softmax(output, dim=1)
        
        # print(f"features shape: {features.shape}, outputs shape: {output.shape}")

        return self.features0, output



#  Define hook for feature extraction
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = None
        self.gradients = None
        self.hook = self.model.layer1.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
  
        self.features = input[0].mean(dim=1).unsqueeze(1)
        #print(f"features shape: {self.features.shape}")
    
        
        # normalize the features
        self.features = self.features / self.features.max()
    
    def remove(self):
        self.hook.remove()
        return self.features
    
    def __call__(self, x):
        # print(f"input image shape:{x.shape}")
        self.model(x)
        #print(f"features shape: {self.features.shape}")
        return self.features
    

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer_names, use_cuda=False):
        """
        Args:
            model (nn.Module): The model to inspect.
            target_layer_names (list): List of layer names to target.
            use_cuda (bool): Whether to use CUDA.
        """
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.target_layers = target_layer_names
        self.feature_maps = {}
        self.gradients = {}

        # Register hooks
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                module.register_forward_hook(self.save_feature_maps(name))
                module.register_backward_hook(self.save_gradients(name))

    def save_feature_maps(self, layer_name):
        def hook(module, input, output):
            self.feature_maps[layer_name] = output.detach()
        return hook

    def save_gradients(self, layer_name):
        def hook(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()
        return hook

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            input = input.cuda()
        
        output = self.forward(input)[1]

        if index is None:
            index = torch.argmax(output, dim=1)

        # Zero grads
        self.model.zero_grad()

        # One-hot for the target class
        one_hot = torch.zeros_like(output)
        one_hot.scatter_(1, index.view(-1,1), 1.0)

        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)

        cam_dict = {}
        for layer_name in self.target_layers:
            gradients = self.gradients[layer_name]  # [N, C, H, W]
            activations = self.feature_maps[layer_name]  # [N, C, H, W]
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [N, 1, H, W]
            cam = F.relu(cam)
            # Normalize CAM
            cam = cam - cam.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
            cam = cam / (cam.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-8)
            cam_dict[layer_name] = cam
        return cam_dict
