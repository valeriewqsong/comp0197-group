import torch
import torch.nn as nn

def semi_supervised_bce_loss(y_pred, y_true, unlabeled_pred, alpha=0.5):
    """
    Compute the semi-supervised BCE loss for labeled and unlabeled data using pseudo-labels.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, num_classes, height, width).
        unlabeled_pred (torch.Tensor): Predictions for unlabeled data, of shape (batch_size, num_classes, height, width).
        alpha (float): Weight for consistency regularization. Default is 0.5.

    Returns:
        torch.Tensor: Semi-supervised BCE loss.
    """
    # Binary Cross Entropy loss function.
    # Sigmoid function will be applied to convert raw model outputs into probabilities with a range of [0, 1].
    criterion = nn.BCEWithLogitsLoss()
    
    # Compute labeled loss
    labeled_loss = criterion(y_pred, y_true)
    
    # Generate pseudo-labels from unlabeled data
    # Apply sigmoid to convert raw model outputs into probabilities with a range of [0, 1]
    # Convert the probabilities into binary values (0 or 1) using a threshold of 0.5
    unlabeled_pred_pseudo = (torch.sigmoid(unlabeled_pred) > 0.5).float()
    
    # Compute unlabeled loss
    unlabeled_loss = criterion(unlabeled_pred, unlabeled_pred_pseudo)

    # Combining the losses
    return labeled_loss + alpha * unlabeled_loss, labeled_loss, unlabeled_loss



def dice_score(y_pred, y_true, smooth=1e-5):
    """
    Compute the Dice score for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, num_classes, height, width).
        smooth (float): Smoothing factor to prevent division by zero. Default is 1e-5.

    Returns:
        float: Mean Dice score across the batch.
    """
    # Apply sigmoid to convert raw model outputs into probabilities with a range of [0, 1]
    y_pred = torch.sigmoid(y_pred)
    
    # Convert the probabilities into binary values (0 or 1) using a threshold of 0.5
    y_pred = (y_pred > 0.5).float()

    # Compute the intersection and union between the ground truth and the predictions
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection

    # Calculate the Dice score and average it across the batch
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(y_pred, y_true, smooth=1e-5):
    """
    Compute the Intersection over Union (IoU) score for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, num_classes, height, width).
        smooth (float): Smoothing factor to prevent division by zero. Default is 1e-5.

    Returns:
        float: Mean IoU score across the batch.
    """
    # Apply sigmoid to convert raw model outputs into probabilities with a range of [0, 1]
    y_pred = torch.sigmoid(y_pred)
    
    # Convert the probabilities into binary values (0 or 1) using a threshold of 0.5
    y_pred = (y_pred > 0.5).float()

    # Compute the intersection and union between the ground truth and the predictions
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection

    # Calculate the IoU score and average it across the batch
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
