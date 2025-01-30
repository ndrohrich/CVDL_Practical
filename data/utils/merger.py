from data.CK_plus import CK_plus
from data.FER2013 import FER2013
from data.AffectNet import AffectNetDataset
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

    if cfg.dataset_mode == 'ck_plus':
        train_dataset = CK_plus(split='train', 
                                transform=train_transforms)
        test_dataset = CK_plus(split='test',
                               transform=test_transforms)
    elif cfg.dataset_mode == 'affectnet':
        train_dataset = AffectNetDataset(split='train', 
                                         transform=train_transforms)
        test_dataset = AffectNetDataset(split='test', 
                                        transform=test_transforms)
    elif cfg.dataset_mode == 'fer': 
        train_dataset = FER2013(split='train',
                                transform=train_transforms)
        test_dataset = FER2013(split='test',
                               transform=test_transforms)
    else: 
        train_dataset = []
        test_dataset = []
        
        # Get CK+ dataset
        train_dataset1 = CK_plus(split='train', 
                                transform=train_transforms)
        test_dataset1 = CK_plus(split='test',
                               transform=test_transforms)
        
        # Get AffectNet dataset
        train_dataset2 = AffectNetDataset(split='train', 
                                         transform=train_transforms)
        test_dataset2 = AffectNetDataset(split='test', 
                                        transform=test_transforms)
        
        # Get FER2013 dataset
        train_dataset3 = FER2013(split='train',
                                transform=train_transforms)
        test_dataset3 = FER2013(split='test',
                               transform=test_transforms)

        # Merge datasets
        train_dataset.append(train_dataset1)
        train_dataset.append(train_dataset2)
        train_dataset.append(train_dataset3)

        test_dataset.append(test_dataset1)
        test_dataset.append(test_dataset2)
        test_dataset.append(test_dataset3)
        
        train_dataset = torch.utils.data.ConcatDataset(train_dataset)
        test_dataset = torch.utils.data.ConcatDataset(test_dataset)

    datasets = {'train': train_dataset, 
                'test': test_dataset}
    
    # Log input and label shape
    print(f"IMAGE SHAPE: {train_dataset[0][0].shape}")
    print(f"LABEL SHAPE: {train_dataset[0][1].shape}")

    # Log dataset lengths
    print(f'TRAIN DATASET LENGTH: {len(train_dataset)}')
    print(f'TEST DATASET LENGTH: {len(test_dataset)}')
    
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