def init_transforms(cfg): 
    '''
    Takes arguments specified in command line via hydra and returns 
    train and test transforms using torch transforms. 

    Parameters
    ----------

    cfg: OmegaConf
        Hyperparameters parsed via command-line using hydra.
    '''

    #TODO
    train_transforms = None
    test_transforms = None

    transforms = {'train': train_transforms, 
                  'test': test_transforms}
    
    return transforms