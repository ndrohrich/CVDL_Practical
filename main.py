import hydra
from torch.utils.tensorboard import SummaryWriter
from training import train, Train_Classification
from datetime import datetime
import os

@hydra.main(version_base=None, config_path="Configs", config_name="config")
def main(cfg) -> None: 
    
   
    trainer = train.Trainer(cfg)

    print("NUMBER OF PARAMETERS: ", trainer.parameters)
    print("DEVICE TRAINED ON: ", next(trainer.model.parameters()))
    
    trainer.train()
    

if __name__ == '__main__': 
    main()
