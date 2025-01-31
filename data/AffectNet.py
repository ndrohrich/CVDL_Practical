import os
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from tqdm import tqdm
import kagglehub
from PIL import Image

class AffectNetDataset(Dataset):
    def __init__(self, transform, split):
        path = kagglehub.dataset_download("thienkhonghoc/affectnet")
        path = path + '/AffectNet/' + split + '/'
        self.class_folders = [0,1,2,3,4,6,7]
        self.dataset = []
        self.transforms = transform
        for cls in self.class_folders:
            class_path = path + str(cls)
            fs = os.listdir(class_path)
            for f in fs:
                self.dataset.append({
                    'image': class_path + '/' + f,
                    'label': int(cls)
                })


    def __len__(self):
        return len(self.dataset)
    
    def _open_image(self, img):
        return self.transforms(img)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self._open_image(Image.open(item['image']))
        label = self._label_to_onehot(item['label'])
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
    def _label_to_onehot(self, label_string): 
        one_hot_encoding = torch.zeros((6))
        match label_string: 
            case 3: 
                one_hot_encoding[0] = 1
            case 5: 
                one_hot_encoding[1] = 1
            case 4: 
                one_hot_encoding[2] = 1
            case 0: 
                one_hot_encoding[3] = 1
            case 1: 
                one_hot_encoding[4] = 1
            case 2: 
                one_hot_encoding[5] = 1
            case 1: # contempt -> Counted as Disgust 
                one_hot_encoding[4] = 1
            case _:
                raise ValueError(label_string)
        return one_hot_encoding

# Example usage
if __name__ == "__main__":
    from torchvision import transforms as tf

    train_transforms = tf.Compose([tf.Resize((64,64)),
                                   tf.Grayscale(),
                                   tf.ToTensor()])
    train_dataset = AffectNetDataset(split='train',transform=train_transforms)
    print(f"Number of training samples: {len(train_dataset)}")
    image, label = train_dataset[0]
