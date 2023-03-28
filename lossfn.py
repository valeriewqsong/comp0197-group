import torch
import torch.nn.functional as F

def to_one_hot(tensor, num_classes):
    """
    Converts a tensor of shape (N,) into a one-hot encoded tensor of shape (N, num_classes, H, W).
    Args:
        tensor (torch.Tensor): The tensor to be one-hot encoded.
        num_classes (int): The number of classes in the one-hot encoded tensor.
    Returns:
        torch.Tensor: The one-hot encoded tensor.
    """
    # Get the shape of the input tensor
    shape = tensor.shape

    # Create an empty tensor of shape (N, num_classes)
    one_hot = torch.zeros(shape[0], num_classes, dtype=torch.float32)

    # Fill the one_hot tensor with ones at the indices corresponding to the input tensor
    one_hot.scatter_(1, tensor.unsqueeze(1), 1)

    return one_hot.unsqueeze(2).unsqueeze(3)

def semisup_dice_loss(pred, target, pred_unlabeled, alpha, smooth=1):
    """
    Computes the semi-supervised Dice loss between labeled and unlabeled predictions and targets.
    
    Dice loss is a measure of the overlap between two sets, defined as:
        Dice = (2 * intersection + smooth) / (union + smooth)
        
    Args:
        pred (torch.Tensor): Labeled data predictions, shape (N, C, H, W)
        target (torch.Tensor): Labeled data targets, shape (N,)
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
        target (torch.Tensor): Labeled data targets, shape (N,)
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
    union_unlabeled = pred_unlabeled.sum(dim=(2, 3)) + true_label_unlabeled.sum(dim=(2, 3)) - intersection_unlabeled
    iou_unlabeled = (intersection_unlabeled + eps) / (union_unlabeled + eps)
    iou_loss_unlabeled = 1 - iou_unlabeled.mean()
    
    # Combined loss
    loss = iou_loss_labeled + alpha * iou_loss_unlabeled
    
    return loss
    
def dice_loss_and_score(pred, target, smooth=1):
    """
    Computes the Dice loss between predictions and targets and returns the loss and score.
    
    Args:
        pred (torch.Tensor): Predictions, shape (N, C, H, W)
        target (torch.Tensor): Targets, shape (N,)
        smooth (float, optional): Smoothing factor to avoid division by zero, default=1

    Returns:
        Tuple[torch.Tensor, float]: The Dice loss and score
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()
    dice_score = dice.mean().item()
    return dice_loss, dice_score

def iou_loss_and_score(pred, target, eps=1e-6):
    """
    Computes the IoU loss between predictions and targets.
    
    Args:
        pred (torch.Tensor): Predictions, shape (N, C, H, W)
        target (torch.Tensor): Targets, shape (N,)
        eps (float, optional): Small constant to avoid division by zero, default=1e-6

    Returns:
        tuple(torch.Tensor, float): The IoU loss and the average IoU score
    """
    target_one_hot = to_one_hot(target, num_classes=37)
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    iou_loss = 1 - iou.mean()
    iou_score = iou.mean().item()
    return iou_loss, iou_score

def accuracy_score(outputs, target):
    """
    Computes the accuracy between predictions and targets.

    Args:
        outputs (torch.Tensor): Predictions, shape (N, C, H, W)
        target (torch.Tensor): Targets, shape (N,)

    Returns:
        accuracy (float): The accuracy of the prediction
    """
    # Get the class with the highest probability for each pixel
    predicted = outputs.argmax(dim=1)

    # Convert target tensor to one-hot encoding
    target_one_hot = torch.zeros_like(outputs)
    target_expanded = target.view(-1, 1, 1).expand_as(outputs[:, 0, :, :])
    target_one_hot.scatter_(1, target_expanded.unsqueeze(1), 1)

    # Compare the predicted segmentation masks with the ground truth labels
    correct_pixels = (predicted == target_one_hot.argmax(dim=1)).float()
    accuracy = correct_pixels.sum() / (correct_pixels.numel())

    return accuracy.item()
