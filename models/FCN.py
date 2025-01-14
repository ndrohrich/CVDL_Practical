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
    def __init__(self, model, layer):
        self.model = model
        self.features = None
        self.gradients = None
        target_layer = layer
        self.hook = target_layer.register_forward_hook(self.hook_fn)
    
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