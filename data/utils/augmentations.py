from torchvision import transforms as tf
import numpy as np
import torch
import random


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



def HOG_transformer(image):
    # Compute HOG features using torch
    gx = torch.nn.functional.conv2d(image.unsqueeze(0), torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32), padding=1)
    gy = torch.nn.functional.conv2d(image.unsqueeze(0), torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32), padding=1)
    magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
    orientation = torch.atan2(gy, gx).squeeze()
    # Convert back to tensor
    hog_image = torch.stack([magnitude, orientation], dim=0)
    # to grayscale
    hog_image = hog_image[0]
    return hog_image

def Er_Br_Con_hog_transforms(cfg):
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
                                   tf.Normalize(0.5, 0.5),
                                   tf.RandomHorizontalFlip(p=cfg.probability),
                                   tf.ColorJitter(brightness=np.random.uniform(cfg.brightness[0], cfg.brightness[1]), contrast=np.random.uniform(cfg.contrast[0], cfg.contrast[1])),
                                   tf.Lambda(lambda x: HOG_transformer(x)),
                                   tf.Lambda(lambda x: torch.unsqueeze(x, dim=0))
    ])
                                   
    test_transforms = tf.Compose([tf.Resize((64,64)),
                                  tf.Grayscale(),
                                  tf.ToTensor(),
                                  tf.Normalize(0.5, 0.5),
                                  tf.Lambda(lambda x: HOG_transformer(x)),
                                  tf.Lambda(lambda x: torch.unsqueeze(x, dim=0))
                                ])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms

def random_half_black(image,p=0.5):
    """
    Randomly black out half of the image.
    
    Parameters
    ----------
    image: torch.Tensor
        Input image tensor.
    
    Returns
    -------
    torch.Tensor
        Transformed image tensor.
    """
    # Get image dimensions
    _, h, w = image.shape
    
    # Randomly choose which half to black out
    if random.random() < p:
        # Black out the left half
        p=random.random()
        if p<0.5:
            image[:, :, :w//2] = 0 #black out left half
        else: # black out upper half
            image[:, :h//2, :] = 0
        
    else:
        # Black out the right half
        p=random.random()
        if p<0.5:
            image[:, w//2:, :] = 0
        else: # black out lower half
            image[h//2:, :, :] = 0
    return image

def random_half_transforms(cfg):
    '''
    Takes arguments specified in command line via hydra and returns 
    train and test transforms using torch transforms, including random half-black augmentation.
    
    Parameters
    ----------
    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    
    Returns
    -------
    dict
        Dictionary containing 'train' and 'test' transforms.
    '''
    
    train_transforms = tf.Compose([
        tf.Resize((64, 64)),
        tf.Grayscale(),
        tf.ToTensor(),
        tf.Lambda(lambda x: random_half_black(x,cfg.probability))
    ])
    
    test_transforms = tf.Compose([
        tf.Resize((64, 64)),
        tf.Grayscale(),
        tf.ToTensor()
    ])

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms