import os
import torch
from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np

# Read the label files
all_files_name = os.path.join("annotations", 'list.txt')
train_path = os.path.join("annotations", 'trainval.txt')
test_path = os.path.join("annotations", 'test.txt')

# Get the labels and store into dictionary
f_train = open(train_path, 'r').readlines()
labels = {}
for i in range(len(f_train)):
    line = f_train[i]
    name_list = line.split(" ")[0].split("_")[:-1]
    name = "_".join(name_list)
    if name in labels:
        continue
    labels[name] = len(labels)
print(labels)

# Split the training data into with labels and without labels
if os.path.exists("trainlabel.txt") and os.path.exists("trainwithoutlabel.txt"):
    pass
else:
    # Create files to store labeled and unlabeled training data if they don't already exist
    f1 = open("trainlabel.txt", 'w')
    f2 = open("trainwithoutlabel.txt", 'w')

    train1 = {}
    train2 = {}

    train_total = {}
    for line in f_train:
        name_list = line.split(" ")[0].split("_")[:-1]
        name = "_".join(name_list)
        if name not in train_total:
            train_total[name] = []
        train_total[name].append(line.split(" ")[0])

    # Split the data into labeled and unlabeled sets
    for k in train_total:
        files = train_total[k] # get all the training files of the same label
        seg_len = len(files)//2
        for_label = random.sample(files, seg_len)

        for f in files:
            if f in for_label:
                f1.write(f+"\n")
            else:
                f2.write(f+"\n")

    f1.close()
    f2.close()
    
# Read the labeled and unlabeled training data
training1 = open("trainlabel.txt", 'r').readlines()
training2 = open("trainwithoutlabel.txt", 'r').readlines()

# Read the test data
test = open(test_path, 'r').readlines()
test = [i.split(' ')[0] for i in test]

# Create the dataset with labels
class segmentDataset_With_Label(Dataset):    
    def __init__(self, split):
        if split == "test":
            self.paths = test
        else:
            self.paths = training1
        pass

    def __getitem__(self, index):    
        img_name = training1[index]
        path = os.path.join('images', img_name.strip()+".jpg")
        
        
        img = Image.open(path).convert('RGB')   # To read image file as PIL object
        label = Image.open("annotations/trimaps/"+img_name.strip()+".png").convert('L') # To return labels as single-channel tensor wit integer values corresponding to the class labels for each pixel

        img = img.resize((224, 224))            # HWC
        label = label.resize((224, 224), Image.NEAREST)

        img = np.transpose(img, (2, 0, 1))      # CHW
        return torch.from_numpy(np.array(img, dtype=np.float32))/255, torch.from_numpy(np.array(label, dtype=np.long))

    def __len__(self):
        return len(self.paths)

# Create the dataset without labels (just for training)
class segmentDataset_Without_Label(Dataset):
    # just for training
    def __init__(self):
        pass

    def __getitem__(self, index):
        img_name = training2[index]
        img = Image.open("images/"+img_name.strip()+".jpg").convert('RGB')  # To read image file as PIL object
        img = img.resize((224, 224))
        img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(np.array(img, dtype=np.float32))/255
    
    def __len__(self):
        return len(training2)


trainset1 = segmentDataset_With_Label("train")
trainset2 = segmentDataset_Without_Label()
testset = segmentDataset_With_Label("test")

# Create dataloader objects
from torch.utils.data import DataLoader

train_loader_with_label = DataLoader(trainset1, batch_size=4, shuffle=True, drop_last=True)
train_loader_without_label = DataLoader(trainset2, batch_size=4, shuffle=True, drop_last=True)

test_loader = DataLoader(trainset1, batch_size=4, shuffle=True, drop_last=False)
