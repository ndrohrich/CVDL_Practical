import os
import torch
from torch.utils.data import Dataset, random_split
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from tqdm import tqdm

class AffectNetDataset(Dataset):
    def __init__(self, transform, split):
        cache_dir = 'data/cache'
        train_cache_path = os.path.join(cache_dir, 'affectnet_train.pt')
        test_cache_path = os.path.join(cache_dir, 'affectnet_test.pt')

        
        if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
            # Load cached datasets
            dataset_train = torch.load(train_cache_path)
            dataset_test = torch.load(test_cache_path)
        else:
            # Load and filter dataset
            dataset = load_dataset('chitradrishti/AffectNet', cache_dir=cache_dir)['train']
            
                
            filtered_dataset = []
            for item in tqdm(dataset, desc='filter') :
                if item['label'] != 5:
                    filtered_dataset.append(item)
                    
            
            # Split dataset
            train_len = int(0.8 * len(filtered_dataset))
            test_len = len(filtered_dataset) - train_len
            dataset_train, dataset_test = random_split(filtered_dataset, lengths=(train_len, test_len), generator=torch.Generator().manual_seed(42))
            
            # Save datasets to cache
            torch.save(dataset_train, train_cache_path)
            torch.save(dataset_test, test_cache_path)

        self.dataset = dataset_train if split == 'train' else dataset_test
        self.transforms = transform

    def __len__(self):
        return len(self.dataset)
    
    def _open_image(self, img):
        return self.transforms(img)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self._open_image(item['image'])
        label = self._label_to_onehot(item['label'])
        return image, label
    
    def __len__(self):
        return len(self.dataset)
    
    def _label_to_onehot(self, label_string): 
        one_hot_encoding = torch.zeros((6))
        match label_string: 
            case 4: 
                one_hot_encoding[0] = 1
            case 7: 
                one_hot_encoding[1] = 1
            case 6: 
                one_hot_encoding[2] = 1
            case 0: 
                one_hot_encoding[3] = 1
            case 2: 
                one_hot_encoding[4] = 1
            case 3: 
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