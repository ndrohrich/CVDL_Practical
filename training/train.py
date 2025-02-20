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

from models.utils import discriminativ_loss

from models.FCN import FeatureExtractor
from models.utils.visualization import visualize_gradients
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt



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
        if self.args.model == 'fcn' or self.args.model == 'ACN' or self.args.model == 'hybrid_fcn':
            print('Using discriminative loss')
            self.loss = discriminativ_loss.combined_loss(feature_dim=self.args.fcn_feature_dim, 
                                                          num_classes=self.args.num_classes, 
                                                          alpha=self.args.combine_alpha, 
                                                          beta=self.args.combine_beta)
        else:
            self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(params=self.model.parameters(), 
                                     lr=self.args.lr)
        self._to_device()
        
        # assign feature extractor
        self.feature_extractor = self.use_hook(self.model)
        
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

        self.dataloader_train = DataLoader(dataset=dataset_train, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, pin_memory=True)
        self.dataloader_test = DataLoader(dataset=dataset_test, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        
        print(f"image size: {next(iter(self.dataloader_train))[0].shape}")

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
        checkpoint_dir = os.path.join(cwd, 'training', 'trained_models', self.args.model, now, 'checkpoints')


        os.makedirs(log_dir)
        os.makedirs(model_dir)
        os.makedirs(checkpoint_dir)


        self.model_dir = model_dir
        self.checkpoint_dir = checkpoint_dir 
        self.log_dir = log_dir

    def pretrain(self):
        self.pretrained_model = pretrain.get_pretrained_models(self.args)

    def train(self): 
        # Run training loop 
        for epoch in tqdm(range(self.args.epochs)):
            self.epoch = epoch
            logging_metrics = self.train_test_one_epoch()
            self.log_metrics(epoch, logging_metrics)
            self.writer.flush()
            if epoch % 10 == 0:
                self._save_checkpoint(epoch)
        self._save_model()

    def load_checkpoint(self,checkpoint_pth):
        self.model = torch.load(checkpoint_pth)

    def test(self): 
        all_preds = []
        all_targets = []
        with tqdm(self.dataloader_test) as iterator: 
            for images, targets in iterator: 
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                cls_pred = outputs.argmax(dim=-1).cpu().numpy()  
                target_cls = targets.argmax(dim=-1).cpu().numpy()  
                
                all_preds.extend(cls_pred)
                all_targets.extend(target_cls)

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        label_names = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

        cm = confusion_matrix(all_targets, all_preds, labels=np.arange(6)) 

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_names)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f'\n \n #{cm[i, j]}', 
                        ha='center', va='center', color='black')

        plt.title('Confusion Matrix Hybrid Model')
        plt.savefig('outputs/confusion_matrix2.png')
        

    def _save_checkpoint(self,epoch):

        # Ensure the checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)


        checkpoint_path = os.path.join(self.checkpoint_dir, f'model_{epoch}_epochs.pth')
        torch.save(self.model, checkpoint_path)
        print(f'SAVED CHECKPOINT TO DIR: {self.checkpoint_dir}')

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
            step = 0
            for images, targets in iterator: 
                images, targets = images.to(self.device), targets.to(self.device)

                loss, outputs = self.train_one_epoch(images, targets)
                accuracy = self.get_accuracy(outputs, targets)

                running_loss_train += loss
                running_accuracy_train += accuracy
                
                # # Log features to tensorboard as images
                # with torch.no_grad():
                #     if self.args.model == 'fcn' or self.args.model == 'torch_resnet' or self.args.model == 'ACN':
                #         if step % 80 == 0:
                #             features = self.feature_extractor(self.dataloader_test.dataset[3][0].unsqueeze(0).to(self.device))
                #             self.writer.add_images('Features', features, self.epoch*len(self.dataloader_train)+step)
            
                step += 1

        # Test one epoch
        with tqdm(self.dataloader_test) as iterator: 
            for images, targets in iterator: 
                images, targets = images.to(self.device), targets.to(self.device)

                loss, outputs = self.test_one_epoch(images, targets)
                accuracy = self.get_accuracy(outputs, targets)

                running_loss_test += loss
                running_accuracy_test += accuracy
                
        # log activation maps
        features = next(iter(self.dataloader_test))[0].to(self.device)
        labels = next(iter(self.dataloader_test))[1].to(self.device)
        self.log_activation_maps(self.epoch, features, labels)
        
        
        # free cuda memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

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
        self.model.train()
        # Clear Previous gradients 
        self.model.zero_grad()
        self.optimizer.zero_grad()

        
        if self.args.model=="fcn" or self.args.model=="ACN":
            features,outputs=self.model(inputs)
            loss=self.loss(features,outputs,targets)
        if self.args.model=='hybrid_fcn':
            outputs = self.model(inputs)
            features = self.model.features
            features = features.view(features.size(0), -1)
            loss = self.loss(features, outputs, targets)
        else:
            # Forward pass
            outputs = self.model(inputs)

            # Backward pass
            loss = self.loss(outputs, targets)
            
        loss.backward()

        # Update weights 
        self.optimizer.step()

        return loss, outputs 
    
    def test_one_epoch(self, inputs, targets):
        self.model.eval()
        with torch.no_grad(): # No need to compute gradients for validation
            if self.args.model=="fcn" or self.args.model=="ACN":
                features,outputs=self.model(inputs)
                loss=self.loss(features,outputs,targets)
            elif self.args.model=='hybrid_fcn':
                outputs = self.model(inputs)
                features = self.model.features
                features = features.view(features.size(0), -1)
                loss = self.loss(features, outputs, targets)
            else:
                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss = self.loss(outputs, targets)
        return loss, outputs 
    
    def use_hook(self, model, layer=None):
        feature_extractor = None
        match self.args.model:
            case 'fcn':
                layer = model.layer1
            case 'torch_resnet':
                layer = model.model.layer2
            case 'ACN':
                layer = model.after_attention
            case _:
                return None
        feature_extractor = FeatureExtractor(model, layer)
        return feature_extractor
    
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
        
    def log_activation_maps(self, epoch, features, labels):
        activation_map = visualize_gradients(self.model, features, labels,self.args)
        self.writer.add_images('Activation Maps', activation_map, epoch)
