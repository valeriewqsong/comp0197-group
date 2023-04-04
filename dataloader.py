import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import random

def preprocess_mask(mask, label):
    mask = np.float32(mask)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = label
    return mask


class OxfordPetsDataset(Dataset):
    """Oxford-IIIT Pet dataset."""

    def __init__(self, img_dir, img_labels=None, transform=None, labeled=True):
        """
        Initialize the dataset.

        Args:
            img_dir (str): Path to the directory containing the images.
            img_labels (list): List of tuples containing image names and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
            labeled (bool): Whether the dataset should be labeled or not.
        """
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.mask_dir = os.path.join("annotations", "trimaps")
        self.transform = transform
        self.mask_transform = transforms.Compose([transforms.ToTensor(),     
                            transforms.Resize((256, 256)),
                            transforms.CenterCrop(224),  
                            transforms.Lambda(lambda x: (x).squeeze().type(torch.LongTensor)) ])

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label, seg_mask) where image and seg_mask are PIL.Image.Image instances and label is an integer.
        """
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        
        label = self.img_labels[idx][1]
        
        seg_mask_path = os.path.join(self.mask_dir, self.img_labels[idx][0] + ".png")
        seg_mask = preprocess_mask(Image.open(seg_mask_path), float(1))
        
        if self.transform:
            image = self.transform(image)
            seg_mask = self.mask_transform(seg_mask)
            seg_mask = seg_mask.unsqueeze(0)
            
        return image, seg_mask


def split_data(annotations_file, split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split the data into training, validation, and test sets based on the given ratios.

    Args:
        annotations_file (str): Path to the annotations file containing image names and labels.
        split_ratios (tuple): A tuple containing the ratios for training, validation, and test sets, respectively.
                              The sum of the ratios must be equal to 1. Default is (0.8, 0.1, 0.1).
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (train_data, val_data, test_data), where each is a list of tuples containing image names and labels.
    """
    with open(annotations_file, 'r') as f:
        img_labels = [tuple(line.strip().split(' ')[:2]) for line in f if line.strip() and line.strip().split(' ')[0]]

    np.random.seed(seed)
    np.random.shuffle(img_labels)

    train_samples = int(len(img_labels) * split_ratios[0])
    val_samples = int(len(img_labels) * split_ratios[1])

    train_data = img_labels[:train_samples]
    val_data = img_labels[train_samples:train_samples + val_samples]
    test_data = img_labels[train_samples + val_samples:]
    
    return train_data, val_data, test_data


def split_labeled_unlabeled(data, UtoL_ratio=4.0, seed=42):
    """
    Split the given data into labeled and unlabeled sets based on the given ratio.

    Args:
        data (list): List of tuples containing image names and labels.
        UtoL_ratio (float, optional): Ratio of unlabeled samples to labeled samples. Default is 4.0 (4 unlabeled sample for every 1 labeled sample).
        seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (labeled_data, unlabeled_data), where each is a list of tuples containing image names and labels.
    """
    np.random.seed(seed)
    np.random.shuffle(data)

    labeled_samples = int(len(data) / (1 + UtoL_ratio))
    labeled_data = data[:labeled_samples]
    unlabeled_data = data[labeled_samples:]
    
    return labeled_data, unlabeled_data


def get_data_loader(basedir="./", batch_size=4, UtoL_ratio=4.0, num_workers=0):
    """Create and return Data Loaders for semi-supervised learning.

    Args:
        basedir (str): Base directory where the images and annotations folders are located.
        batch_size (int, optional): Number of labeled samples per batch. Default is 4.
        UtoL_ratio (float, optional): Ratio of unlabeled samples to labeled samples to use per batch. Default is 4.0.
        num_workers (int, optional): Number of workers for data loading. Default is 0.

    Returns:
        tuple: (train_labeled_loader, train_unlabeled_loader, test_labeled_loader) where each is a torch.utils.data.DataLoader instance.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_files = [os.path.join(basedir, 'annotations/trainval.txt'), os.path.join(basedir, 'annotations/test.txt')]
    output_file = os.path.join(basedir, 'annotations/combined.txt')

    with open(output_file, 'w') as outfile:
        for file in input_files:
            with open(file, 'r') as infile:
                content = infile.read()
                outfile.write(content)

    train_data, val_data, test_data = split_data(output_file) 
    
    print(f"Training data length: {len(train_data)}")
    print(f"Validation data length: {len(val_data)}")
    print(f"Test data length: {len(test_data)}")

    labeled_data, unlabeled_data = split_labeled_unlabeled(train_data, UtoL_ratio=UtoL_ratio)

    train_labeled_dataset = OxfordPetsDataset(os.path.join(basedir, 'images'), img_labels=labeled_data, transform=data_transforms)
    train_unlabeled_dataset = OxfordPetsDataset(os.path.join(basedir, 'images'), img_labels=unlabeled_data, transform=data_transforms, labeled=False)
    val_labeled_dataset = OxfordPetsDataset(os.path.join(basedir, 'images'), img_labels=val_data, transform=data_transforms)
    test_labeled_dataset = OxfordPetsDataset(os.path.join(basedir, 'images'), img_labels=test_data, transform=data_transforms)

    print(f"Training labeled dataset length: {len(train_labeled_dataset)}")
    print(f"Training unlabeled dataset length: {len(train_unlabeled_dataset)}")
    print(f"Validation dataset length: {len(val_labeled_dataset)}")
    print(f"Test dataset length: {len(test_labeled_dataset)}")
    
    unlabeled_batch_size = int(batch_size * UtoL_ratio)
    
    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=unlabeled_batch_size, shuffle=True, num_workers=num_workers) if len(train_unlabeled_dataset) != 0 else None
    val_labeled_loader = DataLoader(val_labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_labeled_loader = DataLoader(test_labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader

