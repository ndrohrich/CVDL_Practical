from torchvision import transforms as tf
import numpy as np

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

def randomErasing_transforms(cfg): 
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
                                   tf.ToTensor(),
                                   tf.RandomErasing(cfg.probability, scale=(cfg.min_area, cfg.max_area), ratio=(cfg.min_aspect_ratio, cfg.max_aspect_ratio))])
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor()])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms

def randomErasing_flip_transforms(cfg): 
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
                                   tf.RandomHorizontalFlip(p=cfg.probability),
                                   tf.ToTensor(),
                                   tf.RandomErasing(cfg.probability, scale=(cfg.min_area, cfg.max_area), ratio=(cfg.min_aspect_ratio, cfg.max_aspect_ratio))])
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor()])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms

def randomErasing_flip_rotate_transforms(cfg):
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
                                   tf.RandomHorizontalFlip(p=cfg.probability),
                                   tf.RandomRotation(degrees=cfg.rotation_angle),
                                   tf.ToTensor(),
                                   tf.RandomErasing(cfg.probability, scale=(cfg.min_area, cfg.max_area), ratio=(cfg.min_aspect_ratio, cfg.max_aspect_ratio))])
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor()])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms

def random_Erasing_flip_Brightness_Contrast_transforms(cfg):
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
                                   tf.RandomHorizontalFlip(p=cfg.probability),
                                   tf.ColorJitter(brightness=np.random.uniform(cfg.brightness[0], cfg.brightness[1]), contrast=np.random.uniform(cfg.contrast[0], cfg.contrast[1])),
                                   tf.ToTensor(),
                                   tf.RandomErasing(cfg.probability, scale=(cfg.min_area, cfg.max_area), ratio=(cfg.min_aspect_ratio, cfg.max_aspect_ratio))])
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor()])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms


