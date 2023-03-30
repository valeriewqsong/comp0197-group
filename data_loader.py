import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
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
        seg_mask = preprocess_mask(Image.open(seg_mask_path), float(label))
        seg_mask = to_categorical(seg_mask, num_classes=38)
        if self.transform:
            image = self.transform(image)
            seg_mask = self.mask_transform(seg_mask)
        return image, seg_mask



def split_data(annotations_file, split_ratio=11, test=False):
    """Split the data into labeled and unlabeled sets based on random sampling.

    Args:
        annotations_file (str): Path to the annotations file containing image names and labels.
        split_ratio (int, optional): Ratio of labeled to unlabeled samples. Default is 11. 11 means 10:1. 12 means 11:1. 

    Returns:
        tuple: (labeled_data, labeled_masks, unlabeled_data, unlabeled_masks) where each is a list of tuples containing image names and labels/segmentation masks.
    """
    with open(annotations_file, 'r') as f:
        img_labels = [tuple(line.strip().split(' ')[:2]) for line in f if line.strip() and line.strip().split(' ')[0]]
    np.random.shuffle(img_labels)
    if not test : 
        labeled_samples = len(img_labels) // split_ratio
        labeled_data = img_labels[:labeled_samples]
        unlabeled_data = img_labels[labeled_samples:]
        #labeled_masks and unlabeled_masks contain tuples of (img_name, mask_name). img_name is name of input image and
        #mask_name is name of corresponding mask file.
        #Here, we assume that segmentation mask files have same name as input files, put with .png

        print("labeled_data:", len(labeled_data))
        print("unlabeled_data:", len(unlabeled_data))

        return labeled_data, unlabeled_data
    else : 
        return img_labels



def get_data_loader(batch_size=32, num_workers=0):
    """Create and return Data Loaders for semi-supervised learning.

    Args:
        batch_size (int, optional): Number of samples per batch. Default is 32.
        num_workers (int, optional): Number of workers for data loading. Default is 0.
        labeled_samples (int, optional): Number of labeled samples in the training dataset. Default is 100.

    Returns:
        tuple: (train_labeled_loader, train_unlabeled_loader, test_labeled_loader, test_unlabeled_loader) where each is a torch.utils.data.DataLoader instance.
    """
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    labeled_data, unlabeled_data = split_data('annotations/trainval.txt', split_ratio=10)
    test_labeled_data = split_data('annotations/test.txt', split_ratio=10, test=True)
    
    train_labeled_dataset = OxfordPetsDataset('images', img_labels=labeled_data, transform=data_transforms)
    train_unlabeled_dataset = OxfordPetsDataset('images', img_labels=unlabeled_data, transform=data_transforms, labeled=False)
    test_labeled_dataset = OxfordPetsDataset('images', img_labels=test_labeled_data, transform=data_transforms)

    train_labeled_loader = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_unlabeled_loader = DataLoader(train_unlabeled_dataset, batch_size=256, shuffle=True, num_workers=num_workers)
    test_labeled_loader = DataLoader(test_labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(len(train_unlabeled_dataset), "This motherfucking loader has this many examples")
    print(unlabeled_data)

    return train_labeled_loader, train_unlabeled_loader, test_labeled_loader



def imshow(img, mask=None, vs=None):
    """Show an image and an optional segmentation mask."""
    img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if vs is None : 
        # Show the segmentation mask if it's provided
        if mask is not None :
            # plt.imshow(np.transpose(mask.numpy(), (1, 2, 0))[:,:,int(label)], cmap='Reds')
            plt.show()
            plt.imshow(np.transpose(mask.numpy(), (1, 2, 0))[:,:,int(1)], cmap='Reds')

    else :
        fig, ax = plt.subplots()
        im = ax.imshow(np.transpose(mask.numpy(), (1, 2, 0)) , cmap='Reds')
        for i in range(mask.shape[1]):
            for j in range(mask.shape[2]):
                text = ax.text(j, i, mask[0, i, j].item(),
                    ha="center", va="center", color="b")
    plt.show()
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)), cmap='Reds')
    plt.show()



# Checking code
if __name__ == "__main__":

    labeled_samples = 10

    print("Splitting data...")
    # Split the data into labeled and unlabeled sets
    labeled_data, unlabeled_data = split_data('annotations/trainval.txt')

    print("Creating data loaders...")
    # Create the data loaders
    train_labeled_loader, train_unlabeled_loader, test_labeled_loader = get_data_loader()

    print("Getting sample from data loaders...")
    # Display some images and labels from the data loaders
    print(next(iter(train_labeled_loader))[0].shape)
    images, mask = next(iter(train_labeled_loader))
    print("Labeled training images:")
    imshow(images[0], mask[0])

    images, *_ = next(iter(train_unlabeled_loader))
    print("Unlabeled training images:")

    images, *_ = next(iter(test_labeled_loader))
    print("Labeled test images:")
    # print("Labels:", list(labels))
