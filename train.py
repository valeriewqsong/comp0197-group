def train_segmentation_model(model, train_loader_with_label, train_loader_without_label, test_loader, device, epochs=50, alpha=0.5, learning_rate=1e-4, model_path="best_model.pth"):
    """
    Train a semi-supervised segmentation model with labeled and unlabeled data.

    Args:
        model (nn.Module): The segmentation model to be trained.
        train_loader_with_label (DataLoader): Labeled training data loader.
        train_loader_without_label (DataLoader): Unlabeled training data loader.
        test_loader (DataLoader): Test/validation data loader.
        device (str): Device to run the training on, e.g., "cpu" or "cuda".
        epochs (int, optional): Number of training epochs. Default is 50.
        alpha (float, optional): Weight for the unlabeled loss. Default is 0.5.
        learning_rate (float, optional): Learning rate for the optimizer. Default is 1e-4.
                model_path (str, optional): Path to save the best model during training. Default is "best_model.pth".

    Returns:
        nn.Module: The trained segmentation model.
    """

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_iou = 0.0

    for epoch in range(epochs):
        model.train()
        train_iter_without_label = iter(train_loader_without_label)

        for i, (images_with_label, labels) in enumerate(train_loader_with_label):
            try:
                images_without_label = next(train_iter_without_label)
            except StopIteration:
                train_iter_without_label = iter(train_loader_without_label)
                images_without_label = next(train_iter_without_label)

            images_with_label, labels = images_with_label.to(device), labels.to(device)
            images_without_label = images_without_label.to(device)

            pred_with_label = model(images_with_label)
            pred_without_label = model(images_without_label)

            # Compute the pseudo labels for the unlabeled data
            target_unlabeled = (pred_without_label > 0.5).float()

            loss_dice = semi_supervised_dice_loss(pred_with_label, labels, pred_without_label, target_unlabeled, alpha)
            loss_iou = semi_supervised_iou_loss(pred_with_label, labels, pred_without_label, target_unlabeled, alpha)

            loss = loss_dice + loss_iou

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader_with_label)}], Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            iou_score = 0.0
            total_samples = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                pred = model(images)
                pred = (pred > 0.5).float()

                intersection = (pred * labels).sum(dim=2).sum(dim=2)
                union = pred.sum(dim=2).sum(dim=2) + labels.sum(dim=2).sum(dim=2) - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                iou_score += iou.sum().item()
                total_samples += labels.size(0)

            iou_score /= total_samples
            print(f"Epoch [{epoch+1}/{epochs}], IoU Score: {iou_score}")

            if iou_score > best_iou:
                best_iou = iou_score
                torch.save(model.state_dict(), model_path)
                print(f"Best IoU Score updated: {best_iou}, Model saved to {model_path}")

    model.load_state_dict(torch.load(model_path))
    return model
