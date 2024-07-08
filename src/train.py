import os
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class PIDrayDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)['annotations']
        self.image_ids = list(set([anno['image_id'] for anno in self.annotations]))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, f"xray_{str(img_id).zfill(5)}.png")
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for anno in self.annotations:
            if anno['image_id'] == img_id:
                x, y, width, height = anno['bbox']
                boxes.append([x, y, x + width, y + height])
                labels.append(anno['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels}

        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Paths
    data_dir = "../data/pidray/train"
    annotation_file = "../data/pidray/annotations/xray_train.json"

    # Load dataset
    dataset = PIDrayDataset(data_dir, annotation_file, transform=F.to_tensor)
    
    # Select few images
    dataset = torch.utils.data.Subset(dataset, range(6))
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.train()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            images = list(image for image in images)
            targets = [{k: v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {losses.item()}")

if __name__ == "__main__":
    main()
