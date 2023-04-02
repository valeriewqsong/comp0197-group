import torch
import torch.nn as nn
import torch.optim as optim
from linknet import link_net
import os
from lossfn_1ch import supervised_dice_loss, supervised_iou_loss, semi_supervised_dice_loss, semi_supervised_iou_loss, create_pseudo_labels
from torch.utils.tensorboard import SummaryWriter

base_dir = "./"

def train_segmentation_model(train_loader_with_label, train_loader_without_label, test_loader, device, num_epochs=50, lr=1e-4, use_dice=True):
    global base_dir
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
    sw = SummaryWriter(os.path.join(base_dir, 'logs'))
    step = 0 # for tensorboard
    # Initialize the neural network
    model = link_net(classes=1).to(device)    
    # sw = SummaryWriter()

    # Define loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    experiment = "dice" if use_dice else "iou"
    if use_dice:
        print("Training with dice loss function...")
    else:
        print("Training with iou loss function..")
    for epoch in range(num_epochs):
        running_loss = 0.0
        examples = 0
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
            # print("The shape of the labels is: ", labels.shape)
            
            # Set alpha based on i (or should it be epoch number?)
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
            # print("The shape of the output is: ", pred_with_label.shape, pred_without_label.shape)
            
            # determine the loss function used
            if use_dice:
                loss = semi_supervised_dice_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            else:
                loss = semi_supervised_iou_loss(pred_with_label, labels, pred_without_label, alpha=alpha)
            examples = examples + images_with_label.size(0)
            running_loss += loss.item() * images_with_label.size(0)
            loss.backward()
            optimizer.step()
            
            # print stats every iteration
            print(f"Epoch {epoch+1}, iteration {i+1}: loss = {loss.item():.6f} alpha = {alpha}")
            step += 1
            # # print statistics every 50 iteratrions
            # running_loss += loss.item()
            # if i % 50 == 49:
            #     print(f"Epoch {epoch+1}, iteration {i+1}: loss = {running_loss / 50:.6f} alpha = {alpha}")
            #     running_loss = 0.0
        sw.add_scalar("training/{experiment}", running_loss / examples, step)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), os.path.join(base_dir, f"models/{experiment}_model_{epoch}.pth"))
        # Evaluate the model on the test set
        model.eval()

        test_loss = 0
        total_score = 0
        accuracy = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if use_dice:
                    loss = supervised_dice_loss(output, target)
                else:
                    loss = supervised_iou_loss(output, target)
                score = 1 - loss.item()
                test_loss += loss.item()
                total_score += score
                    
        sw.add_scalar("testing/{experiment}_loss", test_loss, step)
        sw.add_scalar("testing/{experiment}_score", total_score, step)
        if use_dice:
            print('Epoch {}, Test Loss: {:.6f} Dice Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_score/len(test_loader)))
        else:
            print('Epoch {}, Test Loss: {:.6f} IoU Score: {:.6f}'.format(epoch+1, test_loss/len(test_loader), total_score/len(test_loader)))

    print("Training completed.")

    return model
