import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn.functional as F

class TorchVisionResNet(nn.Module):
    def __init__(self, model_type='resnet18', num_classes=6, pretrained=True, input_channels=1):
        super(TorchVisionResNet, self).__init__()
        
        
        if model_type == 'resnet18':
            self.model = resnet18(pretrained=pretrained)
        elif model_type == 'resnet34':
            self.model = resnet34(pretrained=pretrained)
        elif model_type == 'resnet50':
            self.model = resnet50(pretrained=pretrained)
        elif model_type == 'resnet101':
            self.model = resnet101(pretrained=pretrained)
        elif model_type == 'resnet152':
            self.model = resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model type: {model_type}")

        
        if input_channels == 1:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        logits = self.model(x)
        return logits
