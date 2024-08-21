import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np

def plot_image_with_boxes(image, boxes, labels, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    
    image = image.cpu().numpy()
    image = image.transpose((1, 2, 0))
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, fill=False, color="red", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x1, y1, f"Label: {labels[i]}", color="white", fontsize=12, backgroundcolor="red")

    ax.axis("off")
    return ax
