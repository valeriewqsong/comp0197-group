from train import train_segmentation_model
import torch
from data_loader import get_data_loader

base_dir = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# train model with iou loss. label to unlabeled data is 4:1
train_labeled_loader, train_unlabeled_loader, test_loader = get_data_loader(base_dir, batch_size=4, ratio=0.25)
print("Ratio of labeled to unlabeled data is 4:1")
iou_trained_model = train_segmentation_model(
    train_labeled_loader,
    train_unlabeled_loader,
    test_loader,
    device,
    num_epochs=100,        
    lr=1e-3, 
    use_dice = False
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_iou_4to1.pt')
print(f'Model trained with iou loss saved.')

# train model with iou loss. label to unlabeled data is 1:4
train_labeled_loader, train_unlabeled_loader, test_loader = get_data_loader(base_dir, batch_size=4, ratio=4.0)
print("Ratio of labeled to unlabeled data is 1:4")
iou_trained_model = train_segmentation_model(
    train_labeled_loader,
    train_unlabeled_loader,
    test_loader,
    device,
    num_epochs=100,        
    lr=1e-3, 
    use_dice = False
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_iou_1to4.pt')
print(f'Model trained with iou loss saved.')

