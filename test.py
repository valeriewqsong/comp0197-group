import torch
import torch.nn as nn
from train_helper import iou_score, dice_score, precision, recall, specificity
from linknet import link_net
from dataloader import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_labeled_loader, train_unlabeled_loader, val_labeled_loader, test_labeled_loader = get_data_loader('./', batch_size=32, UtoL_ratio=0.0)

def evaluate (ptfile, test_loader, device):
    # Initialize the neural network
    model = link_net(classes=1).to(device)  
    
    # load the trained model
    model.load_state_dict(torch.load(ptfile, map_location=device))
    
    model.eval()
    test_loss = 0.0
    total_iou_score = 0.0
    total_dice_score = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_specificity = 0.0
    
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
            output = model(data)
            
            loss = criterion(output, target)
            test_loss += loss.item()
            total_iou_score += iou_score(output, target)
            total_dice_score += dice_score(output, target)
            total_precision += precision(output, target)
            total_recall += recall(output, target)
            total_specificity += specificity(output, target)
    
    print(f"Test Loss = {(test_loss/len(test_loader)):.6f}, IoU Score = {(total_iou_score/len(test_loader)):.6f}, Dice Score = {(total_dice_score/len(test_loader)):.6f}, Precision = {(total_precision/len(test_loader)):.6f}, Recall = {(total_recall/len(test_loader)):.6f}, Specificity = {(total_specificity/len(test_loader)):.6f}")

print("Evaluating model trained on all data as labeled...")
evaluate('saved_model_fully_sup.pt', test_labeled_loader, device)

print("Evaluating model trained on 1:1 labeled to unlabeled data ratio...")
evaluate('saved_model_1to1_semisup.pt', test_labeled_loader, device)
print("Evaluating model trained on 1:1 labeled to unlabeled data ratio, but only labeled data used...")
evaluate('saved_model_1to1_sup.pt', test_labeled_loader, device)

print("Evaluating model trained on 1:3 labeled to unlabeled data ratio...")
evaluate('saved_model_1to3_semisup.pt', test_labeled_loader, device)
print("Evaluating model trained on 1:3 labeled to unlabeled data ratio, but only labeled data used...")
evaluate('saved_model_1to3_sup.pt', test_labeled_loader, device)

print("Evaluating model trained on 1:5 labeled to unlabeled data ratio...")
evaluate('saved_model_1to5_semisup.pt', test_labeled_loader, device)
print("Evaluating model trained on 1:5 labeled to unlabeled data ratio, but only labeled data used...")
evaluate('saved_model_1to5_sup.pt', test_labeled_loader, device)

print("Evaluating model trained on 1:10 labeled to unlabeled data ratio...")
evaluate('saved_model_1to10_semisup.pt', test_labeled_loader, device)
print("Evaluating model trained on 1:10 labeled to unlabeled data ratio, but only labeled data used...")
evaluate('saved_model_1to10_sup.pt', test_labeled_loader, device)