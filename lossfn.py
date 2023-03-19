import torch
import torch.nn.functional as F

def semisup_dice_loss(pred, target, pred_unlabeled, alpha, smooth=1):
    """
    Computes the semi-supervised Dice loss between labeled and unlabeled predictions and targets.
    
    Args:
        pred (torch.Tensor): Labeled data predictions, shape (N, C, H, W)
        target (torch.Tensor): Labeled data targets, shape (N, C, H, W)
        pred_unlabeled (torch.Tensor): Unlabeled data predictions, shape (N, C, H, W)
        alpha (float): The balancing factor between the labeled and unlabeled data contributions
        smooth (float, optional): Smoothing factor to avoid division by zero, default=1

    Returns:
        torch.Tensor: The combined Dice loss for labeled and unlabeled data
    """
    # Labeled data
    intersection_labeled = (pred * target).sum(dim=(2, 3))
    union_labeled = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
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
    
    Args:
        pred (torch.Tensor): Labeled data predictions, shape (N, C, H, W)
        target (torch.Tensor): Labeled data targets, shape (N, C, H, W)
        pred_unlabeled (torch.Tensor): Unlabeled data predictions, shape (N, C, H, W)
        alpha (float): The balancing factor between the labeled and unlabeled data contributions
        eps (float, optional): Small constant to avoid division by zero, default=1e-6

    Returns:
        torch.Tensor: The combined IoU loss for labeled and unlabeled data
    """
    # Labeled data
    intersection_labeled = (pred * target).sum(dim=(2, 3))
    union_labeled = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection_labeled
    iou_labeled = (intersection_labeled + eps) / (union_labeled + eps)
    iou_loss_labeled = 1 - iou_labeled.mean()

    # Unlabeled data
    true_label_unlabeled = (pred_unlabeled > 0.5).float()
    intersection_unlabeled = (pred_unlabeled * true_label_unlabeled).sum(dim=(2, 3))
    union_unlabeled = pred

def dice_score(pred, target):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    numerator = 2 * intersection
    denominator = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2)
    dice = (numerator + 1e-6) / (denominator + 1e-6)
    return dice.mean().item()

def iou_score(pred, target):
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    union = pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

