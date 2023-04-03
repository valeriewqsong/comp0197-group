'''
For use with Binary Semantic Segmentation with only 1 channel, with values representing the probability of a pixel belonging to the foreground.
'''
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
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
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
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection
    iou_loss = 1 - (intersection + smooth) / (union + smooth)

    # Average the losses across the batch
    return iou_loss.mean()


def create_pseudo_labels(unlabeled_pred, threshold=0.5):
    """
    Generate pseudo-labels from the unlabeled predictions for binary segmentation.
    
    Args:
        unlabeled_pred (torch.Tensor): Predictions with shape (batch_size, 1, height, width).
        threshold (float): Threshold to convert probabilities into binary values. Default is 0.5.
        
    Returns:
        torch.Tensor: Pseudo-labels with shape (batch_size, 1, height, width), where values above the threshold are set to 1, and values below or equal to the threshold are set to 0.
    """
    # Apply sigmoid to convert raw model outputs into probabilities with a range of [0, 1]
    unlabeled_pred = torch.sigmoid(unlabeled_pred)
    
    # Create binary pseudo-labels using the threshold
    pseudo_labels = (unlabeled_pred > threshold).float()

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
    unlabeled_loss = supervised_dice_loss(unlabeled_pred, unlabeled_pred_pseudo)

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
    unlabeled_loss = supervised_iou_loss(unlabeled_pred, unlabeled_pred_pseudo)
    print(f"labeled loss: {labeled_loss:.6f}, unlabeled loss: {unlabeled_loss:.6f}, alpha: {alpha}")

    # Combining the losses
    return labeled_loss + alpha * unlabeled_loss