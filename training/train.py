import tqdm 
from torch.utils.data import DataLoader
from models.utils import builder, pretrain
from data.utils import merger, transforms


class Trainer(): 
    def __init__(self, cfg): 
        self.args = cfg 
        self.epochs = self.args.epochs

        if self.args.pretrain: 
            pretrained_model = pretrain.get_pretrained_models(self.args)

        self.model = builder.get_model(self.args, pretrained_model)

    def init_dataloaders(self): 
        datasets = merger.merge_all_datasets(self.args)
        dataset_train = datasets['train']
        dataset_test = datasets['test']

        self.dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True)
        self.dataloader_test = DataLoader(dataset=dataset_test, batch_size=self.args.batch_size, shuffle=False)


    def train(self): 
        self.init_dataloaders()
        for epoch in range(self.epochs):
            self.train_test_one_epoch()

    def train_test_one_epoch(): 
        # TODO
        raise NotImplementedError
        
    def train_one_epoch():
        # TODO
        raise NotImplementedError
    

        