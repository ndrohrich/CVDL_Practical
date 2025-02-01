import kagglehub
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

class FER2013(Dataset):
    """
    FER2013 Dataset class for loading and processing the FER2013 dataset.
    Args:
        transform (callable, optional): A function/transform to apply to the images.
        split (str, optional): The dataset split to use, either 'train' or 'test'. Default is 'train'.
        cache (bool, optional): Whether to cache the dataset in memory. Default is True.
    Attributes:
        datapath (str): Path to the downloaded dataset.
        splitdatapath (str): Path to the specific split of the dataset.
        transform (callable): The transform function to apply to the images.
        emoclass (list): List of emotion classes.
        len_dict (dict): Dictionary containing the number of images in each class.
        total_images (int): Total number of images in the dataset.
        cache (bool): Whether the dataset is cached in memory.
        datas (list): List of images if cached.
        labels (list): List of labels if cached.
    Methods:
        __len__(): Returns the total number of images in the dataset.
        __getitem__(idx): Returns the image and label at the specified index.
    """
    
    def __init__(self,transform=None,split='train',cache=False):
        super().__init__()
        if not split in ['train','test']:
            raise ValueError("Invalid split, choose from 'train' or 'test'")
        
        print(f"Initalizing FER2013 dataset for {split} split")
        self.datapath = kagglehub.dataset_download("msambare/fer2013")
        self.splitdatapath=os.path.join(self.datapath,split)
        
        self.transform=transform
        
        # self.emoclass=os.listdir(self.splitdatapath)
        # self.emoclass.remove('neutral')
        self.emoclass=['happy','surprise','sad','angry','disgust','fear'] # adjust the classes as per the dataset
        print(f"Emotion classes: {self.emoclass}")
        
        self.len_dict={}
        for i in self.emoclass:
            self.len_dict[i]=len(os.listdir(os.path.join(self.splitdatapath,i)))
        print(f"Number of images in each class: {self.len_dict}")
        
        self.total_images=sum(self.len_dict.values())
        print(f"Total number of images: {self.total_images}")
        
        # print totalsize in GB
        print(f"Total size of dataset: {sum([os.path.getsize(os.path.join(self.splitdatapath,i,img)) for i in self.emoclass for img in os.listdir(os.path.join(self.splitdatapath,i))])/(1024**3):.2f} GB")
        
        # if split=='test':
        #     cache=False
        self.cache=cache

        # Removed Cache 
        if cache:
            self.datas=[]
            self.labels=[]
            for i in self.emoclass:
                images=os.listdir(os.path.join(self.splitdatapath,i))
                print(f"Loading images from {i} class")
                for img in tqdm(images):
                    with Image.open(os.path.join(self.splitdatapath,i,img)) as im:
                        im.load()
                        self.datas.append(im.copy())
                    #one hot encoding
                    _label=torch.zeros(len(self.emoclass))
                    _label[self.emoclass.index(i)]=1
                    self.labels.append(_label)
                    
            # self.datas=torch.stack(self.datas)
            self.labels=torch.stack(self.labels)  
        
    def __len__(self):
        return self.total_images
    
    def __getitem__(self,idx):
        if self.cache:
            return self.transform(self.datas[idx]),self.labels[idx]
        else:
            for i in self.emoclass:
                images=os.listdir(os.path.join(self.splitdatapath,i))
                if idx<self.len_dict[i]:
                    if self.transform:
                        img=self.transform(Image.open(os.path.join(self.splitdatapath,i,images[idx])))
                    else:
                        img=Image.open(os.path.join(self.splitdatapath,i,images[idx]))
                        img=torch.tensor(np.array(img))
                    
                    
                    label=torch.tensor(self.emoclass.index(i))
                    
                    #one hot encoding
                    _label=torch.zeros(len(self.emoclass))
                    _label[label]=1
                    
                    return img,_label
                else:
                    idx-=self.len_dict[i]
                    
        raise IndexError
    
        
    
# from utils import augmentations

# if __name__=='__main__':
#     transforms=augmentations.init_transforms(None)
#     dataset=FER2013(transform=transforms['train'])
    
#     print(len(dataset))
#     img,target=dataset[np.random.randint(0,len(dataset))]
#     print(img.shape)
#     print(target)
#     plt.imshow(img[0])
#     plt.show()