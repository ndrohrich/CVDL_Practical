from models import ViT
from models import FCN
from models import CNN_LeNet
import torch
import os

def get_model(args, pretrained_encoder=None):
    match args.model:
        case 'vit': 
            model = ViT.VisionTransformer(depth=args.depth, 
                                          embed_dim=args.embed_dim, 
                                          num_heads=args.num_heads, 
                                          image_size=args.image_size,
                                          patch_size=args.patch_size,
                                          num_channels=args.num_channels, 
                                          num_classes=args.num_classes,
                                          pretrained_encoder=pretrained_encoder)
        case 'fcn':
            model = FCN.ResNet(block=FCN.BasicBlock, 
                               layers=[2, 2, 2, 2], 
                               in_chanel=args.num_channels,
                               feature_dim=args.fcn_feature_dim, 
                               output_dim=args.num_classes)
        case 'lenet':
            model = CNN_LeNet.LeNet5(num_classes= args.num_classes)
            
      
        
        case _:
            raise NotImplementedError
    
    if args.load_model:
        if not os.path.exists(args.load_model):
            raise FileNotFoundError(f"Model not found at {args.load_model}. Train the model first!")
        else:
            model=torch.load(args.load_model, map_location=args.device)
        
        
    return model