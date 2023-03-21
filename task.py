from dataset import train_loader_with_label, train_loader_without_label, test_loader
from train import train_segmentation_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dice_trained_model = train_segmentation_model(
    train_loader_with_label,
    train_loader_without_label,
    test_loader,
    device,
    num_epochs=2,          
    alpha=0.5,          
    lr=1e-4, 
    use_dice = True
)

iou_trained_model = train_segmentation_model(
    train_loader_with_label,
    train_loader_without_label,
    test_loader,
    device,
    num_epochs=2,        
    alpha=0.5,         
    lr=1e-4, 
    use_dice = False
)
