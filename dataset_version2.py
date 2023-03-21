import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


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
        self.img_labels = img_labels if labeled else [("", -1) for _ in range(len(img_labels))]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_labels)

    def __getitem__(self, idx):
        """Get a sample from the dataset given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a PIL.Image.Image instance and label is an integer.
        """
        img_path = os.path.join(self.img_dir, self.img_labels[idx][0] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels[idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label

def split_data(annotations_file, labeled_samples=100):
    """Split the data into labeled and unlabeled sets based on random sampling.

    Args:
        annotations_file (str): Path to the annotations file containing image names and labels.
        labeled_samples (int): Number of labeled samples in the resulting split.

    Returns:
        tuple: (labeled_data, unlabeled_data) where each is a list of tuples containing image names and labels.
    """
    with open(annotations_file, 'r') as f:
        img_labels = [tuple(line.strip().split(' ')[:2]) for line in f if line.strip() and line.strip().split(' ')[0]]
    
    np.random.shuffle(img_labels)
    
    labeled_data = random.sample(img_labels, labeled_samples)
    unlabeled_data = [sample for sample in img_labels if sample not in labeled_data]
    
    return labeled_data, unlabeled_data



def get_data_loader(batch_size=32, num_workers=0, labeled_samples=100):
    """Create and return Data Loaders for semi-supervised learning.

    Args:
        batch_size (int, optional): Number of samples per batch. Default is 32.
        num_workers (int, optional): Number of workers for data loading. Default is 0.
        labeled_samples (int, optional): Number of labeled samples in the training dataset. Default is 100.

    Returns:
        tuple: (train_labeled_loader, train_unlabeled_loader, test_loader) where each is a torch.utils.data.DataLoader instance.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_labeled_data, train_unlabeled_data = split_data('annotations/trainval.txt', labeled_samples=labeled_samples)
    with open('annotations/test.txt', 'r') as f:
        test_img_labels = [tuple(line.strip().split(' ')[:2]) for line in f if line.strip() and line.strip().split(' ')[0]]
    test_img_labels = split_data('annotations/test.txt', labeled_samples=len(test_img_labels))[0]  # Get all test data as labeled

    train_labeled_loader = DataLoader(OxfordPetsDataset('images', img_labels=train_labeled_data, transform=data_transforms), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_unlabeled_loader = DataLoader(OxfordPetsDataset('images', img_labels=train_unlabeled_data, transform=data_transforms, labeled=False), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(OxfordPetsDataset('images', img_labels=test_img_labels, transform=data_transforms), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_labeled_loader, train_unlabeled_loader, test_loader





def imshow(img):
    img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    # Set desired number of labeled samples
    labeled_samples = 100

    # Create the data loaders
    train_labeled_loader, train_unlabeled_loader, test_loader = get_data_loader(labeled_samples=labeled_samples)

    # Display some images from the labeled and unlabeled training sets
    images, labels = next(iter(train_labeled_loader))
    imshow(torchvision.utils.make_grid(images))
    print("Labeled training images:")
    print("Labels:", labels.tolist())

    images, _ = next(iter(train_unlabeled_loader))
    imshow(torchvision.utils.make_grid(images))
    print("Unlabeled training images:")

    exit()





