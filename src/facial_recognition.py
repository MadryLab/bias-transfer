import torchvision
import torch
import numpy as np
import pandas as pd
import os

CELEBA_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

class SpuriousAttributeCelebAAge(torch.utils.data.Dataset):
    def __init__(self, return_secondary=False, **kwargs):
        self.ds = torchvision.datasets.CelebA(**kwargs, transform=CELEBA_TRANSFORM)
        self.attr_names = self.ds.attr_names
        
        young_index = self.attr_names.index('Young')
        male_index = self.attr_names.index('Male')
        
        self.young_targets = self.ds.attr[:, young_index]
        self.male_targets = self.ds.attr[:, male_index]
        self.return_secondary = return_secondary
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        primary = self.young_targets[idx]
        secondary = self.male_targets[idx]
        if self.return_secondary:
            return x, primary, secondary
        else:
            return x, primary
        
class FairFace:
    def __init__(self, ds_root, split, return_secondary=False, mini_dataset=-1):
        self.ds_root = ds_root
        if split == 'train':
            labels_path = os.path.join(self.ds_root, "fairface_label_train.csv")
        else:
            labels_path = os.path.join(self.ds_root, "fairface_label_val.csv")
        
        self.young_labels = np.array(['0-2', '10-19', '20-29', '3-9'])
        self.labels = pd.read_csv(labels_path)
        if mini_dataset > 0 and mini_dataset != 1:
            N = len(self.labels)
            indices = np.round(np.linspace(0, N, int(N*mini_dataset), endpoint=False)).astype(int)
            self.labels = self.labels.iloc[indices, :]
            print(f"using {len(self.labels)} out of {N} points")
        
        ages = self.labels['age'].to_numpy()
        genders = self.labels['gender'].to_numpy()
        self.labels['is_young'] = (np.in1d(ages, self.young_labels).astype(int))
        self.labels['is_male'] = (genders == 'Male').astype(int)
        
        self.file_names = self.labels['file'].to_numpy()
        self.is_young = self.labels['is_young'].to_numpy()
        self.is_male = self.labels['is_male'].to_numpy()
        self.return_secondary = return_secondary
        
        self.pil_loader = torchvision.datasets.folder.pil_loader
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = os.path.join(self.ds_root, self.file_names[idx])
        is_young = self.is_young[idx]
        is_male = self.is_male[idx]
        
        img = self.pil_loader(file_name)
        img = CELEBA_TRANSFORM(img)
        
        if self.return_secondary:
            return img, is_young, is_male
        else:
            return img, is_young
        
        
def get_training_loaders(ds_name, ds_path, shuffle_train=True, batch_size=512, num_workers=10,
                         train_indices=None, test_indices=None, mini_dataset=False):
    if shuffle_train:
        shuffle=True
        drop_last=True
    print("mini dataset", mini_dataset)
        
    if ds_name == 'celeba':
        train_ds = SpuriousAttributeCelebAAge(root=ds_path, split='train')
        test_ds =  SpuriousAttributeCelebAAge(root=ds_path, split='test')
    elif ds_name == 'fairface':
        train_ds = FairFace(ds_root=ds_path, split='train', mini_dataset=mini_dataset)
        test_ds = FairFace(ds_root=ds_path, split='test')    
    
    if train_indices is not None:
        train_ds = torch.utils.data.Subset(train_ds, train_indices)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                                     drop_last=drop_last, num_workers=num_workers)
    
    
    if test_indices is not None:
        test_ds = torch.utils.data.Subset(test_ds, test_indices)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                           drop_last=False, num_workers=num_workers)
    return train_dl, test_dl
