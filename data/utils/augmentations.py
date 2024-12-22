from torchvision import transforms as tf

def init_transforms(cfg): 
    '''
    Takes arguments specified in command line via hydra and returns 
    train and test transforms using torch transforms. 

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''
    
    train_transforms = tf.Compose([tf.Resize((64,64)),
                                   tf.Grayscale(),
                                   tf.ToTensor()])
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor()])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms