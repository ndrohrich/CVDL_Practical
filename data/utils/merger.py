def merge_all_datasets(cfg):
    '''
    Combines all used datasets into a single dataset, 
    applies transforms and returns train and test splits. 

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''

    #TODO
    train_dataset = None
    test_dataset = None

    datasets = {'train': train_dataset, 
                  'test': test_dataset}
    
    return datasets