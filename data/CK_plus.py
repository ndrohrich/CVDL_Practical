import torch 
from torch.utils.data import Dataset, random_split
from datasets import load_dataset
import os,sys

# set the cache directory
root_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir))
cache_dir=os.path.join(root_dir,'data')

class CK_plus(Dataset):
    def __init__(self, transform, split):
        super().__init__()
        dataset = load_dataset("AlirezaF138/ckplus-dataset", cache_dir=cache_dir)['train']
        dataset_train, dataset_test = random_split(dataset, lengths=(0.8, 0.2), generator=torch.Generator().manual_seed(42))
        self.dataset = dataset_train if split == 'train' else dataset_test
        self.transforms = transform
        
    def _open_image(self, img):
        return self.transforms(img)
    
    def _label_to_onehot(self, label_string): 
        one_hot_encoding = torch.zeros((6))
        match label_string: 
            case 'happy': 
                one_hot_encoding[0] = 1
            case 'surprise': 
                one_hot_encoding[1] = 1
            case 'sadness': 
                one_hot_encoding[2] = 1
            case 'anger': 
                one_hot_encoding[3] = 1
            case 'disgust': 
                one_hot_encoding[4] = 1
            case 'fear': 
                one_hot_encoding[5] = 1
            case 'contempt': # Counted as Disgust 
                one_hot_encoding[4] = 1
            case _:
                raise ValueError(label_string)
        return one_hot_encoding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset.__getitem__(idx)
        img = self._open_image(item['image'])
        label = self._label_to_onehot(item['label'])

        return img, label

