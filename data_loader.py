import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
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
        #seg_mask_path = os.path.join(self.img_dir, self.seg_masks[idx][0] + ".png")
        seg_mask_path = os.path.join(self.mask_dir, self.img_labels[idx][0] + ".png")
        seg_mask = preprocess_mask(Image.open(seg_mask_path), float(1))
        if self.transform:
            image = self.transform(image)
            seg_mask = self.mask_transform(seg_mask)
            seg_mask = seg_mask.unsqueeze(0)
        return image, seg_mask

def split_data(annotations_file, split_ratio=11, test=False):
    """Split the data into labeled and unlabeled sets based on random sampling.

    Args:
        annotations_file (str): Path to the annotations file containing image names and labels.
        split_ratio (int, optional): Ratio of labeled to unlabeled samples. Default is 11. 11 means 10:1. 12 means 11:1. 

    Returns:
        tuple: If test is False, returns (labeled_data, unlabeled_data), where each is a list of tuples containing image names and labels/segmentation masks. If test is True, returns a list of tuples containing image names and labels/segmentation masks.
    """
    with open(annotations_file, 'r') as f:
        img_labels = [tuple(line.strip().split(' ')[:2]) for line in f if line.strip() and line.strip().split(' ')[0]]
    np.random.shuffle(img_labels)
        if not test: 
            labeled_samples = len(img_labels) // split_ratio
            labeled_data = img_labels[:labeled_samples]
            unlabeled_data = img_labels[labeled_samples:]

            print("labeled_data:", len(labeled_data))
            print("unlabeled_data:", len(unlabeled_data))

            return labeled_data, unlabeled_data
        else: 
            return img_labels

def get_data_loader(basedir="./", batch_size=32, ratio=8.0, num_workers=0):
    """Create and return Data Loaders for semi-supervised learning.

    Args:
        basedir (str): Base directory where the images and annotations folders are located.
        batch_size (int, optional): Number of labeled samples per batch. Default is 32.
        ratio (float, optional): Ratio of unlabeled samples to labeled samples to use per batch. Default is 8.0.
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

    labeled_data, unlabeled_data = split_data(os.path.join(basedir, 'annotations/trainval.txt'), split_ratio=10)
    test_labeled_data = split_data(os.path.join(basedir,'annotations/test.txt'), split_ratio=10, test=True)
    
    train_labeled_dataset = OxfordPetsDataset(os.path.join(basedir,'images'), img_labels=labeled_data, transform=data_transforms)
    train_unlabeled_dataset = OxfordPetsDataset(os.path.join(basedir,'images'), img_labels=unlabeled_data, transform=data_transforms, labeled=False)
    test_labeled_dataset = OxfordPetsDataset(os.path.join(basedir,'images'), img_labels=test_labeled_data, transform=data_transforms)

    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=int(batch_size*ratio), shuffle=True, num_workers=num_workers)
    test_labeled_loader = DataLoader(test_labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(len(train_unlabeled_dataset), "This loader has this many examples")
    print(unlabeled_data)

    return train_labeled_loader, train_unlabeled_loader, test_labeled_loader

