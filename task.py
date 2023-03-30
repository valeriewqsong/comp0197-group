from dataset_semi import train_loader_with_label, train_loader_without_label, test_loader
from train import train_segmentation_model
import torch
from data_loader import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_labeled_loader, train_unlabeled_loader, test_loader = get_data_loader()
# train model with dice loss
dice_trained_model = train_segmentation_model(
    train_labeled_loader,
    train_unlabeled_loader,
    test_loader,
    device,
    num_epochs=2,          
    lr=1e-4, 
    use_dice = True
)
# save trained model
torch.save(dice_trained_model.state_dict(), f'saved_model_dice.pt')
print(f'Model trained with dice loss saved.')

# train model with iou loss
iou_trained_model = train_segmentation_model(
    train_labeled_loader,
    train_unlabeled_loader,
    test_loader,
    device,
    num_epochs=2,        
    lr=1e-4, 
    use_dice = False
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_iou.pt')
print(f'Model trained with iou loss saved.')
