import torch
from dataloader import get_data_loader
from train_fn import train_labeled_and_unlabeled, train_labeled_only

base_dir = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Upper bound performance: all data is labeled
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader(base_dir, batch_size=32, UtoL_ratio=0.0)
print("All the data is labeled.")
iou_trained_model = train_labeled_only(
    train_labeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="fully_sup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_all_labeled.pt')
print('Model trained with iou loss and all labeled data saved.\n')


# High labeled data ratio. labeled to unlabeled ratio is 1:1
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader(base_dir, batch_size=32, UtoL_ratio=1.0)
print("Equal split, ratio of labeled to unlabeled data is 1:1")
iou_trained_model = train_labeled_and_unlabeled(
    train_labeled_loader,
    train_unlabeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to1_semisup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to1.pt')
print('Model trained with iou loss and 1:1 ratio saved.\n')

# Lower bound performance: only labeled data is used
print("Equal split, ratio of labeled to unlabeled data is 1:1, but only labeled data is used here.")
iou_trained_model = train_labeled_only(
    train_labeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to1_sup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to1_labeled_only.pt')
print('Model trained with iou loss and 1:1 ratio but labeled only saved.\n')


# Moderate labeled data ratio. labeled to unlabeled ratio is 1:3
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader(base_dir, batch_size=32, UtoL_ratio=3.0)
print("Ratio of labeled to unlabeled data is 1:3")
iou_trained_model = train_labeled_and_unlabeled(
    train_labeled_loader,
    train_unlabeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to3_semisup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to3.pt')
print('Model trained with iou loss and 1:3 ratio saved.\n')

# Lower bound performance: only labeled data is used
print("Ratio of labeled to unlabeled data is 1:3, but only labeled data is used here.")
iou_trained_model = train_labeled_only(
    train_labeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to3_sup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to3_labeled_only.pt')
print('Model trained with iou loss and 1:3 ratio but labeled only saved.\n')


# High unlabeled data ratio. labeled to unlabeled ratio is 1:10
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader(base_dir, batch_size=32, UtoL_ratio=10.0)
print("Ratio of labeled to unlabeled data is 1:10")
iou_trained_model = train_labeled_and_unlabeled(
    train_labeled_loader,
    train_unlabeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to5_semisup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to10.pt')
print('Model trained with iou loss and 1:10 ratio saved.\n')

# Lower bound performance: only labeled data is used
print("Ratio of labeled to unlabeled data is 1:10, but only labeled data is used here.")
iou_trained_model = train_labeled_only(
    train_labeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to5_sup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to10_labeled_only.pt')
print('Model trained with iou loss and 1:10 ratio but labeled only saved.\n')


# Moderate labeled data ratio. labeled to unlabeled ratio is 1:5
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader(base_dir, batch_size=32, UtoL_ratio=5.0)
print("Ratio of labeled to unlabeled data is 1:5")
iou_trained_model = train_labeled_and_unlabeled(
    train_labeled_loader,
    train_unlabeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to10_semisup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to5.pt')
print('Model trained with iou loss and 1:5 ratio saved.\n')

# Lower bound performance: only labeled data is used
print("Ratio of labeled to unlabeled data is 1:5, but only labeled data is used here.")
iou_trained_model = train_labeled_only(
    train_labeled_loader,
    val_labeled_loader,
    device,
    num_epochs=100,        
    lr=1e-5,
    csv_filename="1to10_sup.csv"
)
# save trained model
torch.save(iou_trained_model.state_dict(), f'saved_model_1to5_labeled_only.pt')
print('Model trained with iou loss and 1:5 ratio but labeled only saved.\n')




