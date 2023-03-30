import torch
import torch.nn.functional as F

def supervised_dice_loss(y_pred, y_true):
    """
    Compute the supervised Dice loss for labeled data.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised Dice loss.
    """
    # Converts raw model outputs into probabilities with a range of [0, 1] to ensure that the predicted values are in the same range as the ground truth masks
    y_pred = torch.sigmoid(y_pred)
    
    # Needed to prevent denominator from becoming zero
    smooth = 1e-5   

    # Compute labeled loss by comparing predicted mask with ground truth mask pixel by pixel.
    # Sums the values for each class (num_classes) at each pixel location, over all the rows (height) and columns (width) of the image.
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3))
    dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)

    # Average the losses across the batch
    return dice_loss.mean()


def supervised_iou_loss(y_pred, y_true):
    """
    Compute the supervised IOU loss for labeled data.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised IOU loss.
    """
    # Converts raw model outputs into probabilities with a range of [0, 1] to ensure that the predicted values are in the same range as the ground truth masks
    y_pred = torch.sigmoid(y_pred)
    
    # Needed to prevent denominator from becoming zero
    smooth = 1e-5   

    # Compute labeled loss by comparing predicted mask with ground truth mask pixel by pixel.
    # Sums the values for each class (num_classes) at each pixel location, over all the rows (height) and columns (width) of the image.
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3))
    iou_loss = 1 - (intersection + smooth) / (union + smooth)

    # Average the losses across the batch
    return iou_loss.mean()


def create_pseudo_labels(unlabeled_pred):
    """
    Generate pseudo-labels from the unlabeled predictions.
    
    Args:
        unlabeled_pred (torch.Tensor): Predictions with shape (batch_size, num_classes, height, width).
        
    Returns:
        torch.Tensor: Pseudo-labels with shape (batch_size, num_classes, height, width), where the channel with the 
        highest likelihood for each pixel is set to 1, and all other channels are set to 0.
    """
    # Get the channel with the highest likelihood for each pixel
    max_channels = torch.argmax(unlabeled_pred, dim=1, keepdim=True)
    
    # Create a tensor filled with zeros, matching the shape of unlabeled_pred
    pseudo_labels = torch.zeros_like(unlabeled_pred)
    
    # Set the channel with the highest likelihood for each pixel to 1
    pseudo_labels.scatter_(1, max_channels, 1)
    
    return pseudo_labels


def semi_supervised_dice_loss(y_pred, y_true, unlabeled_pred, alpha=0.5):
    """
    Compute the semi-supervised Dice loss for labeled and unlabeled data using pseudo-labels.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, num_classes, height, width).
        unlabeled_pred (torch.Tensor): Predictions for unlabeled data, of shape (batch_size, num_classes, height, width).
        alpha (float): Weight for consistency regularization. Default is 0.5.

    Returns:
        torch.Tensor: Semi-supervised Dice loss.
    """
    # Converts raw model outputs into probabilities with a range of [0, 1] to ensure that the predicted values are in the same range as the ground truth masks
    y_pred = torch.sigmoid(y_pred)
    
    # Compute labeled loss
    labeled_loss = supervised_dice_loss(y_pred, y_true)
    
    # Generate pseudo-labels from unlabeled data
    unlabeled_pred_pseudo = create_pseudo_labels(unlabeled_pred)
    
    # Compute unlabeled loss
    unlabeled_loss = F.mse_loss(unlabeled_pred, unlabeled_pred_pseudo)

    # Combining the losses
    return labeled_loss + alpha * unlabeled_loss


def semi_supervised_iou_loss(y_pred, y_true, unlabeled_pred, alpha=0.5):
    """
    Compute the semi-supervised IOU loss for labeled and unlabeled data using pseudo-labels.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, num_classes, height, width).
        unlabeled_pred (torch.Tensor): Predictions for unlabeled data, of shape (batch_size, num_classes, height, width).
        alpha (float): Weight for consistency regularization. Default is 0.5.

    Returns:
        torch.Tensor: Semi-supervised IOU loss.
    """
    # Converts raw model outputs into probabilities with a range of [0, 1] to ensure that the predicted values are in the same range as the ground truth masks
    y_pred = torch.sigmoid(y_pred)
    
    # Compute labeled loss
    labeled_loss = supervised_iou_loss(y_pred, y_true)
    
    # Generate pseudo-labels from unlabeled data
    unlabeled_pred_pseudo = create_pseudo_labels(unlabeled_pred)
    
    # Compute unlabeled loss
    unlabeled_loss = F.mse_loss(unlabeled_pred, unlabeled_pred_pseudo)

    # Combining the losses
    return labeled_loss + alpha * unlabeled_loss