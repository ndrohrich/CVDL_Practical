from data.CK_plus import CK_plus

def merge_all_datasets(cfg, train_transforms, test_transforms):
    '''
    Combines all used datasets into a single dataset, 
    applies transforms and returns train and test splits. 

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''

    train_dataset = CK_plus(split='train', 
                            transform=train_transforms)
    test_dataset = CK_plus(split='test',
                           transform=test_transforms)

    datasets = {'train': train_dataset, 
                'test': test_dataset}
    
    return datasets