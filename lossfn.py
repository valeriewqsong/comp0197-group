import torch
import torch.nn.functional as F

def to_one_hot(tensor, num_classes):
    """
    Converts a tensor of shape (N, H, W) into a one-hot encoded tensor of shape (N, num_classes, H, W).

    Args:
        tensor (torch.Tensor): The tensor to be one-hot encoded.
        num_classes (int): The number of classes in the one-hot encoded tensor.

    Returns:
        torch.Tensor: The one-hot encoded tensor.
    """
    tensor = tensor.unsqueeze(1)
    one_hot = torch.zeros(tensor.size(0), num_classes, tensor.size(2), tensor.size(3), device=tensor.device)
    return one_hot.scatter_(1, tensor.long(), 1)

def semisup_dice_loss(pred, target, pred_unlabeled, alpha, smooth=1):
    """
    Computes the semi-supervised Dice loss between labeled and unlabeled predictions and targets.
    
    Dice loss is a measure of the overlap between two sets, defined as:
        Dice = (2 * intersection + smooth) / (union + smooth)
        
    Args:
        pred (torch.Tensor): Labeled data predictions, shape (N, C, H, W)
        target (torch.Tensor): Labeled data targets, shape (N, H, W)
        pred_unlabeled (torch.Tensor): Unlabeled data predictions, shape (N, C, H, W)
        alpha (float): The balancing factor between the labeled and unlabeled data contributions
        smooth (float, optional): Smoothing factor to avoid division by zero, default=1

    Returns:
        torch.Tensor: The combined Dice loss for labeled and unlabeled data
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    
    # Labeled data
    intersection_labeled = (pred * target_one_hot).sum(dim=(2, 3))
    union_labeled = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice_labeled = (2 * intersection_labeled + smooth) / (union_labeled + smooth)
    dice_loss_labeled = 1 - dice_labeled.mean()

    # Unlabeled data
    true_label_unlabeled = (pred_unlabeled > 0.5).float()
    intersection_unlabeled = (pred_unlabeled * true_label_unlabeled).sum(dim=(2, 3))
    union_unlabeled = pred_unlabeled.sum(dim=(2, 3)) + true_label_unlabeled.sum(dim=(2, 3))
    dice_unlabeled = (2 * intersection_unlabeled + smooth) / (union_unlabeled + smooth)
    dice_loss_unlabeled = 1 - dice_unlabeled.mean()

    # Combining the losses
    loss = dice_loss_labeled + alpha * dice_loss_unlabeled

    return loss

def semisup_iou_loss(pred, target, pred_unlabeled, alpha, eps=1e-6):
    """
    Computes the semi-supervised IoU loss between labeled and unlabeled predictions and targets.
    
    The IoU loss is a measure of the overlap between two sets, defined as:
        IoU = (intersection + eps) / (union + eps)
    Args:
        pred (torch.Tensor): Labeled data predictions, shape (N, C, H, W)
        target (torch.Tensor): Labeled data targets, shape (N, H, W)
        pred_unlabeled (torch.Tensor): Unlabeled data predictions, shape (N, C, H, W)
        alpha (float): The balancing factor between the labeled and unlabeled data contributions
        eps (float, optional): Small constant to avoid division by zero, default=1e-6

    Returns:
        torch.Tensor: The combined IoU loss for labeled and unlabeled data
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    
    # Labeled data
    intersection_labeled = (pred * target_one_hot).sum(dim=(2, 3))
    union_labeled = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection_labeled
    iou_labeled = (intersection_labeled + eps) / (union_labeled + eps)
    iou_loss_labeled = 1 - iou_labeled.mean()

    # Unlabeled data
    true_label_unlabeled = (pred_unlabeled > 0.5).float()
    intersection_unlabeled = (pred_unlabeled * true_label_unlabeled).sum(dim=(2, 3))
    union_unlabeled = pred
    
def dice_loss(pred, target, smooth=1):
    """
    Computes the Dice loss between predictions and targets.
    
    Args:
        pred (torch.Tensor): Predictions, shape (N, C, H, W)
        target (torch.Tensor): Targets, shape (N, C, H, W)
        smooth (float, optional): Smoothing factor to avoid division by zero, default=1

    Returns:
        torch.Tensor: The Dice loss
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    return dice_loss

def iou_loss(pred, target, eps=1e-6):
    """
    Computes the IoU loss between predictions and targets.
    
    Args:
        pred (torch.Tensor): Predictions, shape (N, C, H, W)
        target (torch.Tensor): Targets, shape (N, C, H, W)
        eps (float, optional): Small constant to avoid division by zero, default=1e-6

    Returns:
        torch.Tensor: The IoU loss
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    iou_loss = 1 - iou.mean()
    return iou_loss

def dice_score(pred, target):
    """
    Computes the Dice coefficient between two tensors.

    The Dice coefficient is a measure of the overlap between two sets, defined as:
        Dice = (2 * intersection) / (pred + target)

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W)
        target (torch.Tensor): The target tensor of shape (N, H, W)

    Returns:
        float: The average Dice coefficient between the predictions and targets
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=2).sum(dim=2)
    numerator = 2 * intersection
    denominator = pred.sum(dim=2).sum(dim=2) + target_one_hot.sum(dim=2).sum(dim=2)
    dice = (numerator + 1e-6) / (denominator + 1e-6)
    return dice.mean().item()


def iou_score(pred, target):
    """
    Computes the intersection over union (IoU) between two tensors.

    The IoU is a measure of the overlap between two sets, defined as:
        IoU = intersection / union

    Args:
        pred (torch.Tensor): The predicted tensor of shape (N, C, H, W)
        target (torch.Tensor): The target tensor of shape (N, H, W)

    Returns:
        float: The average IoU between the predictions and targets
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target_one_hot.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


