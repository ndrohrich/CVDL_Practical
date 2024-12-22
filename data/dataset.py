import torch 
import numpy as np 
from torch.utils.data import Dataset,DataLoader
import pandas as pd 
import cv2 
from PIL import Image
from torchvision import transforms 
import matplotlib.pyplot as plt
import kagglehub 


class EmoticDataset(Dataset):
    def _init_(self,path='archive/',mode='train',transform=None):
        self.path = path 
        
        self.transforms = transforms.Compose([transforms.Resize((64,64)),
                                              transforms.Grayscale(),
                                              transforms.ToTensor()]) if transform is None else transform

        self.annots_file = f'{self.path}/annots_arrs/annot_arrs_{mode}.csv'
        self.image_path = f'{self.path}/img_arrs/'
        self.items = []
        self.transform = transform
        self.load_data()

    def load_img(self,fn):
        data = Image.fromarray(np.load(fn))
        return data

    def load_data(self):
        data = pd.read_csv(self.annots_file)
        for index, row in data.iterrows():
            item = {}
            item['happiness'] = bool(row.Happiness)
            item['surprise'] = bool(row.Surprise)
            item['sadness'] = bool(row.Sadness)
            item['anger'] = bool(row.Anger)
            item['disgust'] = bool(row.Aversion)
            item['fear'] = bool(row.Fear)
            item['fn'] = row.Crop_name
            img = self.load_img(self.image_path + row.Crop_name)
            item['img'] = img 
            self.items.append(item)


    def _len_(self):
        return len(self.items)

    def _getitem_(self,idx):
        item = self.items[idx]
        img = self.transform(item['img'])

        return {'img':img,
                'happines':item['happiness'],
                'surprise':item['surprise'],
                'sadness':item['sadness'],
                'fear':item['fear'],
                'disgust':item['disgust'],
                'anger':item['anger'],
                }

def visualize_grayscale_batch(tensor_images):
    images_np = tensor_images.squeeze(1).numpy()  

    num_images = len(images_np)
    grid_size = int(num_images**0.5) + 1  

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size , grid_size ))  

    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        if i < num_images:
            axes[i].imshow(images_np[i], cmap='gray')
        else:
            axes[i].axis('off') 

        axes[i].axis('off') 

    plt.tight_layout()
    plt.show()



if _name_ == '_main_':
    dataset = EmoticDataset(mode='train')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch in dataloader:
        #use batch here 

        #visualize_grayscale_batch(batch['img']) -> call for visualisation of batch images