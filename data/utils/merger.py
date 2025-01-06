from data.CK_plus import CK_plus
from data.FER2013 import FER2013
import torch

available_datasets = {'ckplus': CK_plus, 'fer2013': FER2013}

def merge_all_datasets(cfg, train_transforms, test_transforms):
    '''
    Combines all used datasets into a single dataset, 
    applies transforms and returns train and test splits. 

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''

    # datasets = {'ckplus': CK_plus,'fer2013': FER2013}
    
    # # merge all datasets
    # train_dataset = torch.utils.data.ConcatDataset([datasets[dataset](train_transforms, 'train') for dataset in datasets])
    # test_dataset = torch.utils.data.ConcatDataset([datasets[dataset](test_transforms, 'test') for dataset in datasets])
    
    # #using ckplus as dataset
    # train_dataset = CK_plus(train_transforms, 'train')
    # test_dataset = CK_plus(test_transforms, 'test')
    
    #using fer2013 as dataset
    train_dataset = FER2013(train_transforms, 'train')
    test_dataset = FER2013(test_transforms, 'test')

    datasets = {'train': train_dataset, 
                'test': test_dataset}
    
    # printing img space and label space
    print(f"Image space: {train_dataset[0][0].shape}")
    print(f"Label space: {train_dataset[0][1].shape}")
    
    return datasets

def select_dataset(cfg, train_transforms, test_transforms):
    '''
    Selects dataset based on cfg.dataset and returns train and test splits. 
    Note: FER2013 dataset has 7 classes and CK+ dataset has 6 classes. please set the right number of classes in the config file.

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''

    dataset = available_datasets[cfg.dataset]
    train_dataset = dataset(train_transforms, 'train')
    test_dataset = dataset(test_transforms, 'test')
    
    datasets = {'train': train_dataset, 
                'test': test_dataset}
    
    return datasets
