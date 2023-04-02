import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
from lossfn_1ch import supervised_dice_loss, supervised_iou_loss, semi_supervised_dice_loss, semi_supervised_iou_loss, create_pseudo_labels

def train_labeled_and_unlabeled(train_loader_with_label, train_loader_without_label, test_loader, device, num_epochs=50, lr=1e-4, use_dice=True):
    """
    Train a semi-supervised segmentation model with labeled and unlabeled data.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        train_loader_without_label (DataLoader): Unlabeled training data loader.
        test_loader (DataLoader): Test/validation data loader.
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
            
            # Set alpha based on i
            t1 = 100
            t2 = 600
            if i < t1:
                alpha = 0
            elif i < t2:
                alpha = (i - t1) / (t2 - t1)
            else:
                alpha = 3

            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            pred_with_label = model(images_with_label)
            pred_without_label = model(images_without_label)
            
            # determine the loss function used
            if use_dice:
                loss = semi_supervised_dice_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            else:
                loss = semi_supervised_iou_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            
            loss.backward()
            optimizer.step()
            
            # # print stats every iteration
            # print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f} alpha = {alpha}")
            
            # print statistics every 10 iteratrions
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 10:.6f} alpha = {alpha}")
                running_loss = 0.0

        # Evaluate the model on the test set
        model.eval()

        test_loss = 0
        total_iou_score = 0
        total_dice_score = 0
        accuracy = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if use_dice:
                    dice_loss = supervised_dice_loss(output, target)
                    dice_score = 1 - dice_loss.item()
                    test_loss += dice_loss.item()
                    total_dice_score += dice_score
                    
                else:
                    iou_loss = supervised_iou_loss(output, target)
                    iou_score = 1 - iou_loss.item()
                    test_loss += iou_loss.item()
                    total_iou_score += iou_score
                    
                
        if use_dice:
            print('Epoch {}, Test Loss: {:.6f} Dice Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_dice_score/len(test_loader)))
        else:
            print('Epoch {}, Test Loss: {:.6f} IoU Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_iou_score/len(test_loader)))

    print("Training completed.")

    return model

def train_labeled_only(train_loader_with_label, test_loader, device, num_epochs=50, lr=1e-4, use_dice=True):
    """
    Train a supervised segmentation model with labeled data only.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        test_loader (DataLoader): Test/validation data loader.
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

            # print statistics every 10 iteratrions
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 10:.6f}")
                running_loss = 0.0

        # Evaluate the model on the test set
        model.eval()

        test_loss = 0
        total_iou_score = 0
        total_dice_score = 0
        accuracy = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)

                if use_dice:
                    dice_loss = supervised_dice_loss(output, target)
                    dice_score = 1 - dice_loss.item()
                    test_loss += dice_loss.item()
                    total_dice_score += dice_score

                else:
                    iou_loss = supervised_iou_loss(output, target)
                    iou_score = 1 - iou_loss.item()
                    test_loss += iou_loss.item()
                    total_iou_score += iou_score

        if use_dice:
            print('Epoch {}, Test Loss: {:.6f} Dice Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_dice_score/len(test_loader)))
        else:
            print('Epoch {}, Test Loss: {:.6f} IoU Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_iou_score/len(test_loader)))
            
    print("Training completed.")

    return model
