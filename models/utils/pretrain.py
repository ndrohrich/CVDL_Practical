import numpy as np

def get_pretrained_models(args):
    match args.model: 
        case 'cnn':
            pretrainer = MAE_pretrainer(args)
        case 'vit': 
            pretrainer = CNN_pretrainer(args)
    pretrained_model = pretrainer.train()
    return pretrained_model

# TODO
def pretrain_MAE(args): 
    raise NotImplementedError

# TODO
def pretrain_CNN(args):
    raise NotImplementedError

# TODO
class MAE_pretrainer(): 
    def __init__(self, args): 

        # Generate random indexes for masking
        num_patches = (args.image_size // args.patch_size) ** 2
        index_permutation = np.random.permutation(num_patches)
        inverse_permutation = np.argsort(index_permutation)

        raise NotImplementedError
    def train(self):
        raise NotImplementedError

# TODO
class CNN_pretrainer(): 
    def __init__(self, args): 
        raise NotImplementedError
    def train(self):
        raise NotImplementedError