import hydra
from training import train

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg) -> None: 
    
   
    trainer = train.Trainer(cfg)

    print("NUMBER OF PARAMETERS: ", trainer.num_parameters)
    
    trainer.train()
    

if __name__ == '__main__': 
    main()
