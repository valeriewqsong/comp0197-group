from dataset import train_loader_with_label, train_loader_without_label, test_loader
from linknet import link_net
from train import train_segmentation_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = link_net(classes=37)
trained_model = train_segmentation_model(
    model,
    train_loader_with_label,
    train_loader_without_label,
    test_loader,
    device,
    epochs=50,          # You can adjust the number of epochs
    alpha=0.5,          # You can adjust the alpha value
    learning_rate=1e-4, # You can adjust the learning rate
    model_path="best_model.pth"
)
