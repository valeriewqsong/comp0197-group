import torch
import torch.nn as nn

def logits_to_binary(y_pred, threshold=0.5):
    '''
    Convert raw model outputs (logits) into probabilities, and then into binary values using a threshold.
    
    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, 1, height, width).
        threshold (float): Threshold to convert probabilities into binary values. Default is 0.5.
        
    Returns:
        torch.Tensor: Tensor with shape (batch_size, 1, height, width), where values above the threshold are set to 1, and values below or equal to the threshold are set to 0.
    '''
    
    # Apply sigmoid to convert raw model outputs into probabilities with a range of [0, 1]
    y_pred = torch.sigmoid(y_pred)
    
    # Convert the probabilities into binary values (0 or 1) using a threshold of 0.5
    y_pred = (y_pred > threshold).float()
    
    return y_pred


def semi_supervised_bce_loss(y_pred, y_true, unlabeled_pred, alpha=0.5):
    """
    Compute the semi-supervised BCE loss for labeled and unlabeled data using pseudo-labels.

    Args:
        y_pred (torch.Tensor): Predictions for labeled data, of shape (batch_size, 1, height, width).
        y_true (torch.Tensor): Ground truth masks for labeled data, of shape (batch_size, 1, height, width).
        unlabeled_pred (torch.Tensor): Predictions for unlabeled data, of shape (batch_size, 1, height, width).
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
    unlabeled_pred_pseudo = logits_to_binary(unlabeled_pred)
    
    # Compute unlabeled loss
    unlabeled_loss = criterion(unlabeled_pred, unlabeled_pred_pseudo)

    # Combining the losses
    return labeled_loss + alpha * unlabeled_loss, labeled_loss, unlabeled_loss


def iou_score(y_pred, y_true, smooth=1e-5):
    """
    Compute the Intersection over Union (IoU) score for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, 1, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, 1, height, width).
        smooth (float): Smoothing factor to prevent division by zero. Default is 1e-5.

    Returns:
        float: Mean IoU score across the batch.
    """
    # Convert model outputs to binary masks
    y_pred = logits_to_binary(y_pred)

    # Compute the intersection and union between the ground truth and the predictions
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    union = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) - intersection

    # Calculate the IoU score and average it across the batch
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def dice_score(y_pred, y_true, smooth=1e-5):
    """
    Compute the Dice score for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, 1, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, 1, height, width).
        smooth (float): Smoothing factor to prevent division by zero. Default is 1e-5.

    Returns:
        float: Mean Dice score across the batch.
    """
    # Convert model outputs to binary masks
    y_pred = logits_to_binary(y_pred)
    
    # Compute the intersection and union between the ground truth and the predictions
    intersection = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    total = torch.sum(y_true, dim=(1, 2, 3)) + torch.sum(y_pred, dim=(1, 2, 3)) 

    # Calculate the Dice score and average it across the batch
    dice = (2 * intersection + smooth) / (total + smooth)
    return dice.mean().item()


def precision(y_pred, y_true):
    """
    Compute the Precision for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, num_classes, height, width).

    Returns:
        float: Mean Precision across the batch.
    """
    y_pred = logits_to_binary(y_pred)

    true_positives = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    false_positives = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))

    precision = true_positives / (true_positives + false_positives)
    return precision.mean().item()


def recall(y_pred, y_true):
    """
    Compute the Recall for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, num_classes, height, width).

    Returns:
        float: Mean Recall across the batch.
    """
    y_pred = logits_to_binary(y_pred)

    true_positives = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    false_negatives = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))

    recall = true_positives / (true_positives + false_negatives)
    return recall.mean().item()


def specificity(y_pred, y_true, smooth=1e-5):
    """
    Compute the Specificity for given predictions and ground truth masks.

    Args:
        y_pred (torch.Tensor): Predictions, of shape (batch_size, num_classes, height, width).
        y_true (torch.Tensor): Ground truth masks, of shape (batch_size, num_classes, height, width).
        smooth (float): Smoothing factor to prevent division by zero. Default is 1e-5.

    Returns:
        float: Mean Specificity across the batch.
    """
    y_pred = logits_to_binary(y_pred)

    true_negative = torch.sum((1 - y_true) * (1 - y_pred), dim=(1, 2, 3))
    false_positive = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))

    specificity_score = (true_negative + smooth) / (true_negative + false_positive + smooth)
    return specificity_score.mean().item()