import pandas as pd
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CUBDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(
            os.path.join(root_dir, 'images.txt'),
            sep=' ',
            names=['image_id', 'image_path'],
        )
        self.labels_df = pd.read_csv(
            os.path.join(root_dir, 'image_class_labels.txt'),
            sep=' ',
            names=['image_id', 'label'],
        )

        self.split_df = pd.read_csv(
            os.path.join(root_dir, 'train_test_split.txt'),
            sep=' ',
            names=['image_id', 'is_training'],
        )
        self.data = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data = pd.merge(self.data, self.split_df, on='image_id')

        self.data = self.data[self.data.is_training == (1 if is_train else 0)]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, 'images', row['image_path'])
        label = row['label'] - 1
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label

def create_cub_data_loader(dataset_path, is_train, batch_size, num_workers, pin_memory):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = None
    if is_train:
        dataset = CUBDataset(dataset_path, is_train=True, transform=train_transform)
    else:
        dataset = CUBDataset(dataset_path, is_train=False, transform=val_transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2,
    )
    return loader
