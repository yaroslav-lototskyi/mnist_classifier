import matplotlib.pyplot as plt
import torch

def visualize_predictions(model, test_loader, device, num_images=10):
    model.eval()  # Переводимо модель у режим оцінки
    images, labels, predictions = [], [], []

    with torch.no_grad():
        for data in test_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            images.extend(inputs.cpu())
            labels.extend(targets.cpu())
            predictions.extend(preds.cpu())
            
            if len(images) >= num_images:
                break

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for idx in range(num_images):
        img = images[idx].squeeze()
        axes[idx].imshow(img, cmap="gray")
        axes[idx].set_title(f"P: {predictions[idx]} | T: {labels[idx]}")
        axes[idx].axis("off")
    plt.tight_layout()
    plt.show()

