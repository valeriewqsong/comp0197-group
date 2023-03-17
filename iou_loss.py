import torch 
import torch.nn as nn
import torch.optim as optim



def giou_loss(pred_boxes,target_boxes,mask):
    """
    Computes the Generalized Intersection over Union (GIoU) loss
    between the predicted bounding boxes and target bounding boxes.

    Args:
        pred_boxes: Tensor of predicted bounding boxes in format (x1, y1, x2, y2)
        target_boxes: Tensor of target bounding boxes in format (x1, y1, x2, y2)

    Returns:
        Tensor representing the GIoU loss.
    """
    # Compute box areas
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])


    # x1,y1 is the top left corner of the box
    # x2,y2 is the bottom right corner of the box
    
    # Compute box intersection extents
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    # Compute box intersection areas
    # clamps all elements in input into the specified range
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

    # Compute box union areas
    union_area = pred_area + target_area - intersection_area

    # Compute box ious
    ious = intersection_area / union_area

    # Compute box enclosures
    enclosures_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enclosures_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enclosures_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enclosures_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    # Compute box enclosure areas
    enclosures_area = (enclosures_x2 - enclosures_x1) * (enclosures_y2 - enclosures_y1)

    # Compute box giou
    giou = ious - (enclosures_area - union_area) / enclosures_area

    # Compute box giou loss
    giou_loss = 1 - giou
    # Mask out unlabeled boxes
    giou_loss = giou_loss * mask.float()

    # Compute mean loss over labeled boxes
    num_labeled_boxes = torch.sum(mask.float())
    giou_loss = torch.sum(giou_loss) / num_labeled_boxes

    return giou_loss    
    
  
if __name__ == '__main__':
    # Define your model architecture
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*32*32, 10)
    )

# Define your optimizer and learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    
    train_dataset = []
    
    # Load your dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

    # Train your model
    for images, labels, boxes, mask in train_loader:
        # Forward pass
        outputs = model(images)
        loss = giou_loss(outputs, boxes, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Use the trained model to make predictions on the unlabeled data
    unlabeled_train_loader = torch.utils.data.DataLoader(unlabeled_train_dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        model.eval()
        for images, _, _, _ in unlabeled_train_loader:
            outputs = model(images)
            # Convert the model outputs to bounding boxes
            pred_boxes = outputs_to_boxes(outputs)
            # Generate pseudo-labels using the predicted bounding boxes
            pseudo_labels = generate_pseudo_labels(images, pred_boxes)

    # Combine the labeled and pseudo-labeled data, and retrain the model
    combined_train_dataset = torch.utils.data.ConcatDataset([labeled_train_dataset, pseudo_labeled_train_dataset])
    combined_train_loader = torch.utils.data.DataLoader(combined_train_dataset, batch_size=32, shuffle=True)

    for images, labels, boxes, mask in combined_train_loader:
        # Forward pass
        outputs = model(images)
        loss = giou_loss(outputs, boxes, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



