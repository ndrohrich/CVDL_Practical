from tqdm import tqdm
import logging
import torch
import os 
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime

from models.utils import builder, pretrain
from data.utils import augmentations, merger

from models.utils import discriminative_loss


class Trainer(): 
    def __init__(self, cfg): 
        # Save arguments from hydra command-line passing 
        self.args = cfg 

        # Setup cuda    
        self.device = self.args.device if torch.cuda.is_available() else 'cpu'      

        # Init transforms and datasets 
        self._init_transforms() # TODO
        self._init_dataloaders() # TODO

        # If pretrain == True, perform pretraining and pass learned weights
        if self.args.pretrain: 
            self.pretrain()
        else:
            self.pretrained_model = None
        
        # Init model, loss and optimizer
        self.model = builder.get_model(self.args, self.pretrained_model) # TODO 
        self.num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # if using the model need discriminativ loss,else use cross entropy loss
        if self.args.model == 'FCN':
            self.loss = discriminative_loss.AffinityLoss(self.args.feature_dim, self.args.num_classes)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(params=self.model.parameters(), 
                                     lr=self.args.lr)
        self._to_device()
        
        # Create directories, init writer 
        self._make_model_dirs()
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logging.basicConfig(level=logging.INFO)
        
    def _init_transforms(self): 
        transforms = augmentations.init_transforms(self.args)
        self.train_transforms = transforms['train']
        self.test_transforms = transforms['test']

    def _init_dataloaders(self): 
        dataset = merger.merge_all_datasets(self.args, self.train_transforms, self.test_transforms)
        dataset_train = dataset['train']
        dataset_test = dataset['test']

        self.dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True)
        self.dataloader_test = DataLoader(dataset=dataset_test, batch_size=self.args.batch_size, shuffle=False)

    def _to_device(self):
        device = self.device
        self.model.to(device)
        self.loss.to(device)

    def _make_model_dirs(self):
        cwd = os.getcwd()
        now = datetime.now()
        now = now.strftime("%d_%m__%H_%M_%S")
        log_dir = os.path.join(cwd, 'training', 'trained_models', self.args.model, now, 'logs')
        model_dir = os.path.join(cwd, 'training', 'trained_models', self.args.model, now, 'model')

        os.makedirs(log_dir)
        os.makedirs(model_dir)

        self.model_dir = model_dir
        self.log_dir = log_dir

    def pretrain(self):
        self.pretrained_model = pretrain.get_pretrained_models(self.args)

    def train(self): 
        # Run training loop 
        for epoch in tqdm(range(self.args.epochs)):
            self.epoch = epoch
            logging_metrics = self.train_test_one_epoch()
            self.log_metrics(epoch, logging_metrics)
        self._save_model()

    def _save_model(self): 
        torch.save(self.model, os.path.join(self.model_dir, 'model.pth'))
        print(f'SAVED FINAL MODEL TO DIR: {self.model_dir}')

    def train_test_one_epoch(self): 
        # Init logging variables
        running_loss_train = 0.
        running_loss_test = 0.
        running_accuracy_train = 0.
        running_accuracy_test = 0.

        # Train one epoch
        with tqdm(self.dataloader_train) as iterator: 
            for images, targets in iterator: 
                images, targets = images.to(self.args.device), targets.to(self.args.device)

                loss, outputs = self.train_one_epoch(images, targets)
                accuracy = self.get_accuracy(outputs, targets)

                running_loss_train += loss
                running_accuracy_train += accuracy

        # Test one epoch
        with tqdm(self.dataloader_test) as iterator: 
            for images, targets in iterator: 
                images, targets = images.to(self.args.device), targets.to(self.args.device)

                loss, outputs = self.test_one_epoch(images, targets)
                accuracy = self.get_accuracy(outputs, targets)

                running_loss_test += loss
                running_accuracy_test += accuracy

        # Log metrics to tensorboard 
        self.writer.add_scalar("Epoch Training Loss", running_loss_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing Loss", running_loss_test/len(self.dataloader_test), self.epoch)
        self.writer.add_scalar("Epoch Training Accuracy", running_accuracy_train/len(self.dataloader_train), self.epoch)
        self.writer.add_scalar("Epoch Testing Accuracy", running_accuracy_test/len(self.dataloader_test), self.epoch)

        # Log metrics for terminal output
        logging_metrics = {'train_loss': running_loss_train/len(self.dataloader_train),
                           'test_loss': running_loss_test/len(self.dataloader_test),
                           'train_acc': running_accuracy_train/len(self.dataloader_train),
                           'test_acc': running_accuracy_test/len(self.dataloader_test)}

        return logging_metrics 
        
    def train_one_epoch(self, inputs, targets):
        # Clear Previous gradients 
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(inputs)

        # Backward pass
        loss = self.loss(outputs, targets)
        loss.backward()

        # Update weights 
        self.optimizer.step()

        return loss, outputs 
    
    def test_one_epoch(self, inputs, targets):
       
        # Forward pass
        outputs = self.model(inputs)

        # Compute loss
        loss = self.loss(outputs, targets)

        return loss, outputs 
    
    def get_accuracy(self, outputs, targets):
        counter = 0.
        batch_size = targets.shape[0]
        for i in range(batch_size): 
            if torch.argmax(outputs[i]) == torch.argmax(targets[i]):
                counter += 1
        return counter / self.args.batch_size

    def log_metrics(self, epoch, metrics): 
        total_epochs = self.args.epochs
        logging.info(f"EPOCH [{epoch}/{total_epochs}]")
        logging.info(f'''LOSS = [{metrics['train_loss']:.4f}/{metrics['test_loss']:.4f}], 
                     ACCURACY = [{metrics['train_acc']:.4f}/{metrics['test_acc']:.4f}]''')



        