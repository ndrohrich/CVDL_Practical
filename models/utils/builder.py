from models import ViT
from models import FCN
from models import CNN_LeNet
from models import CNN_VGG
from models import CNN_ResNet
from models import CNN_TorchResnet
from models import Hybrid_Simple_Stack
from models import Hybrid_Alternating

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
        
        
    return model