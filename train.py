import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
from lossfn import semisup_dice_loss, semisup_iou_loss, dice_loss_and_score, iou_loss_and_score

def train_segmentation_model(train_loader_with_label, train_loader_without_label, test_loader, device, num_epochs=50, alpha=0.5, lr=1e-4, use_dice=True):
    """
    Train a semi-supervised segmentation model with labeled and unlabeled data.

    Args:
        train_loader_with_label (DataLoader): Labeled training data loader.
        train_loader_without_label (DataLoader): Unlabeled training data loader.
        test_loader (DataLoader): Test/validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        num_epochs (int, optional): Number of training epochs. Default is 50.
        alpha (float, optional): Weight for the unlabeled loss. Default is 0.5.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-4.
        use_dice (bool, optional): If True, semisup_dice_loss used. If False, semisup_iou_loss used. Default is True.
        
    Returns:
        nn.Module: The trained segmentation model.
    """
    # Initialize the neural network
    model = link_net(classes=37).to(device)    
    
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
        train_iter_with_label = iter(train_loader_with_label)
        
        # Use 32 examples of labeled data per epoch
        for i in range(32):
            try:
                images_with_label, labels = next(train_iter_with_label)
            except StopIteration:
                train_iter_with_label = iter(train_loader_with_label)
                images_with_label, labels = next(train_iter_with_label)

            # Use 256 (32 * 8) examples of unlabeled data per epoch
            for j in range(8):
                try:
                    images_without_label_batch = next(train_iter_without_label)
                except StopIteration:
                    train_iter_without_label = iter(train_loader_without_label)
                    images_without_label_batch = next(train_iter_without_label)
                if j == 0:
                    images_without_label = images_without_label_batch
                else:
                    images_without_label = torch.cat((images_without_label, images_without_label_batch), 0)

            images_with_label, labels = images_with_label.to(device), labels.to(device)
            images_without_label = images_without_label.to(device)

            # Set alpha based on epoch number
            t1 = 100
            t2 = 600
            if epoch < t1:
                alpha = 0
            elif epoch < t2:
                alpha = (epoch - t1) / (t2 - t1)
            else:
                alpha = 3
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            pred_with_label = model(images_with_label)
            pred_without_label = model(images_without_label)
            
            # determine the loss function used
            if use_dice:
                loss = semisup_dice_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            else:
                loss = semisup_iou_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            
            loss.backward()
            optimizer.step()
            
            # print statistics every iteration
            print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.3f}")

            # # print statistics every 10 iteratrions
            # running_loss += loss.item()
            # if i % 10 == 9:
            #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 10:.3f}")
            #     running_loss = 0.0

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
                    dice_loss, dice_score = dice_loss_and_score(output, target)
                    test_loss += dice_loss.item()
                    total_dice_score += dice_score
                    
                else:
                    iou_loss, iou_score = iou_loss_and_score(output, target)
                    test_loss += iou_loss.item()
                    total_iou_score += iou_score
                    
                correct = (output.argmax(dim=1) == target).sum().item()
                total = target.size(0) * target.size(1) * target.size(2)
                accuracy += correct / total
                
        if use_dice:
            print('Epoch {}, Test Loss: {:.6f} Dice Score: {:.6f} Accuracy: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_dice_score/len(test_loader), accuracy/len(test_loader)))
        else:
            print('Epoch {}, Test Loss: {:.6f} IoU Score: {:.6f} Accuracy: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_iou_score/len(test_loader), accuracy/len(test_loader)))

    print("Training completed.")

    return model
