import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
from train_helper import semi_supervised_bce_loss, iou_score, dice_score

def train_labeled_and_unlabeled(train_loader_with_label, train_loader_without_label, val_loader, device, num_epochs=50, lr=1e-4):
    """
    Train a semi-supervised segmentation model with labeled and unlabeled data.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        train_loader_without_label (DataLoader): Unlabeled training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Default is 50.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.

    Returns:
        nn.Module: The trained segmentation model.
    """
    # Initialize the neural network
    model = link_net(classes=1).to(device)    

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training with BCE loss function..")
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        # Set alpha based on epoch
        t1 = 10
        t2 = 60
        alpha_f = 3
        
        if epoch < t1:
            alpha = 0
        elif epoch < t2:
            alpha = (epoch - t1) / (t2 - t1) * alpha_f
        else:
            alpha = alpha_f
        
        # Train on both labeled and unlabeled data during each epoch of training
        train_iter_without_label = iter(train_loader_without_label)
        for i, (images_with_label, labels) in enumerate(train_loader_with_label):
            try:
                images_without_label, _ = next(train_iter_without_label)
            except StopIteration:
                train_iter_without_label = iter(train_loader_without_label)
                images_without_label, _ = next(train_iter_without_label)

            images_with_label, labels = images_with_label.to(device), labels.to(device)
            images_without_label = images_without_label.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward pass
            pred_with_label = model(images_with_label)
            pred_without_label = model(images_without_label)
            
            loss, labeled_loss, unlabeled_loss = semi_supervised_bce_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            
            # backward propagation
            loss.backward()
            
            # optimise
            optimizer.step()
            
            # print stats every iteration
            print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f}, labeled loss = {labeled_loss.item():.6f}, unlabeled loss = {unlabeled_loss.item():.6f}, alpha = {alpha}")
            
            # # print statistics every 100 iteratrions
            # running_loss += loss.item()
            # if i % 100 == 99:
            #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 100:.6f} alpha = {alpha}")
            #     running_loss = 0.0

        # Evaluate the model on the validation set
        model.eval()

        val_loss = 0
        total_iou_score = 0
        total_dice_score = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if use_dice:
                    dice_loss = supervised_dice_loss(output, target)
                    dice_score = 1 - dice_loss.item()
                    val_loss += dice_loss.item()
                    total_dice_score += dice_score
                    
                else:
                    iou_loss = supervised_iou_loss(output, target)
                    iou_score = 1 - iou_loss.item()
                    val_loss += iou_loss.item()
                    total_iou_score += iou_score
                    
                
        if use_dice:
            print('After Epoch {}: Validation Loss = {:.6f}, Dice Score = {:.6f}'.format(epoch+1, val_loss/len(val_loader), total_dice_score/len(val_loader)))
        else:
            print('After Epoch {}: Validation Loss = {:.6f}, IoU Score = {:.6f}'.format(epoch+1, val_loss/len(val_loader), total_iou_score/len(val_loader)))

    print("Training completed.")

    return model

def train_labeled_only(train_loader_with_label, val_loader, device, num_epochs=50, lr=1e-4, use_dice=True):
    """
    Train a supervised segmentation model with labeled data only.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        val_loader (DataLoader): Validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Default is 50.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        use_dice (bool, optional): If True, dice loss used. If IOU loss used. Default is True.

    Returns:
        nn.Module: The trained segmentation model.
    """
    # Initialize the neural network
    model = link_net(classes=1).to(device)

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if use_dice:
        print("Training with dice loss function...")
    else:
        print("Training with iou loss function..")

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Train on labeled data only
        for i, (images_with_label, labels) in enumerate(train_loader_with_label):
            images_with_label, labels = images_with_label.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred_with_label = model(images_with_label)

            # determine the loss function used
            if use_dice:
                loss = supervised_dice_loss(pred_with_label, labels)
            else:
                loss = supervised_iou_loss(pred_with_label, labels)

            loss.backward()
            optimizer.step()

            # print stats every iteration
            print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f}")
            
            # # print statistics every 100 iteratrions
            # running_loss += loss.item()
            # if i % 100 == 99:
            #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 100:.6f}")
            #     running_loss = 0.0

        # Evaluate the model on the validation set
        model.eval()

        val_loss = 0
        total_iou_score = 0
        total_dice_score = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                if use_dice:
                    val_loss += supervised_dice_loss(output, target).item()
                    total_dice_score += dice_score(output, target)

                else:
                    val_loss += supervised_iou_loss(output, target).item()
                    total_iou_score += iou_score(output, target)

        if use_dice:
            print('Epoch {}, Validation Loss: {:.6f} Dice Score: {:.6f}'.format(epoch+1, val_loss/len(val_loader), total_dice_score/len(val_loader)))
        else:
            print('Epoch {}, Validation Loss: {:.6f} IoU Score: {:.6f}'.format(epoch+1, val_loss/len(val_loader), total_iou_score/len(val_loader)))
            
    print("Training completed.")

    return model
