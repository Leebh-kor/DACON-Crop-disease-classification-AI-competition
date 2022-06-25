import cv2
import torch
import numpy as np
from os.path import join as opj
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Train_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.img_path = df['img_path'].values
        self.target = df['disease_code'].values
        self.transform = transform

        print(f'Dataset size:{len(self.img_path)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj('../data/', self.img_path[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        target = self.target[idx]

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image, target

    def __len__(self):
        return len(self.img_path)

class Test_dataset(Dataset):
    def __init__(self, df, transform=None):
        self.test_img_path = df['img_path'].values
        self.transform = transform

        print(f'Test Dataset size:{len(self.test_img_path)}')

    def __getitem__(self, idx):
        image = cv2.imread(opj('../data/', self.test_img_path[idx])).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        if self.transform is not None:
            image = self.transform(torch.from_numpy(image.transpose(2,0,1)))

        return image

    def __len__(self):
        return len(self.test_img_path)

def get_loader(df, phase: str, batch_size, shuffle,
               num_workers, transform):
    if phase == 'test':
        dataset = Test_dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    else:
        dataset = Train_Dataset(df, transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True,
                                 drop_last=False)
    return data_loader

def get_train_augmentation(img_size, ver):
    if ver==1: # for validset
        transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                ])

    if ver==2:
        transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine((20)),
                transforms.RandomRotation(90),
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    
    return transform
