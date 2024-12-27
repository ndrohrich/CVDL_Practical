from models import ViT
from models import FCN

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
                               layers=[1, 1, 1, 1], 
                               in_chanel=args.num_channels,
                               feature_dim=args.fcn_feature_dim, 
                               output_dim=args.num_classes)
        case _: 
            raise NotImplementedError()
        
        
    return model