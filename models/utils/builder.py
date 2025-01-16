from models import ViT
from models import FCN
from models import CNN_LeNet
import torch
import os
from models import CNN_VGG
from models import CNN_ResNet
from models import CNN_TorchResnet
<<<<<<< HEAD
from models import ACN
from models import Hybrid
=======
from models import Hybrid_Simple_Stack
from models import Hybrid_Alternating
>>>>>>> origin/main

def get_model(args, pretrained_encoder=None):
    if args.model == 'vit': 
            model = ViT.VisionTransformer(depth=args.depth, 
                                          embed_dim=args.embed_dim, 
                                          num_heads=args.num_heads, 
                                          image_size=args.image_size,
                                          patch_size=args.patch_size,
                                          num_channels=args.num_channels, 
                                          num_classes=args.num_classes,
                                          pretrained_encoder=pretrained_encoder)
    elif args.model == 'fcn':
        model = FCN.ResNet(block=FCN.BasicBlock, 
                            layers=[2, 2, 2, 2], 
                            in_chanel=args.num_channels,
                            feature_dim=args.fcn_feature_dim, 
                            output_dim=args.num_classes)
    elif args.model=='ACN':
            model = ACN.AttentionFeatureCluster(patch_size=8, 
                                                 feature_size=64, 
                                                 num_classes=args.num_classes)
    elif args.model == 'lenet':
        model = CNN_LeNet.LeNet5(num_classes= args.num_classes)
    elif args.model == 'vgg':
        model = CNN_VGG.VGG(num_classes= args.num_classes, input_channels= args.num_channels)
    elif args.model == 'resnet':
        model = CNN_ResNet.ResNet18(num_classes= args.num_classes)
    elif args.model == 'torch_resnet':
        model = CNN_TorchResnet.TorchVisionResNet(
            model_type=args.torch_resnet.model_type,
            num_classes=args.torch_resnet.num_classes,
            pretrained=args.torch_resnet.pretrained,
            input_channels=args.torch_resnet.input_channels
            )
    elif args.model == 'hybrid': 
        model = Hybrid_Simple_Stack.Hybrid(
             num_classes=args.num_classes, 
             input_channels=args.num_channels, 
             depth=args.depth, 
             embed_dim=args.embed_dim,
             num_heads=args.num_heads)
    elif args.model == 'hybrid_alternating': 
        model = Hybrid_Alternating.Hybrid(
             num_classes=args.num_classes, 
             input_channels=args.num_channels, 
             depth=args.depth, 
             embed_dim=args.embed_dim,
             num_heads=args.num_heads
        )
    else:
            raise NotImplementedError
    
    if args.load_model:
        if not args.load_model=='false':
            if not os.path.exists(args.load_model):
                raise FileNotFoundError(f"Model not found at {args.load_model}. Train the model first!")
            else:
                model=torch.load(args.load_model, map_location=args.device)
    return model