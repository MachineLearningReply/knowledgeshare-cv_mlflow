import os
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import mlflow
from mlflow.models import infer_signature
import numpy as np

EXPERIMENT_NAME = "new_experiment_1"


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

def setup_experiment(experiment_name):
    # Set the tracking URI

    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # Create a new experiment if it doesn't exist, with artifact_location set to tracking_uri
        experiment_id = mlflow.create_experiment(
            name=experiment_name
        )
        print(f"Created new experiment with id {experiment_id}")
    else:
        # Use the existing experiment
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with id {experiment_id}")
    
    mlflow.set_experiment(experiment_name)


def main():
    # Paths
    data_dir = "../data/pidray/train"
    annotation_file = "../data/pidray/annotations/xray_train.json"

    # Set up tracking directory relative to the repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tracking_uri = f"file://{os.path.join(repo_root, 'mlruns')}"
    mlflow.set_tracking_uri(tracking_uri)
    setup_experiment(EXPERIMENT_NAME)

    # Start the MLflow run
    with mlflow.start_run():
        # Load dataset
        dataset = PIDrayDataset(data_dir, annotation_file, transform=F.to_tensor)
        
        # Select few images
        dataset = torch.utils.data.Subset(dataset, range(6))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        # Load pre-trained model
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
        model.train()

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # Training loop
        num_epochs = 1
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
                mlflow.log_metric("loss", losses.item())
        
        # Save the trained model's state_dict to a .pt file
        model_path = os.path.join(repo_root, "model_weights.pt")
        torch.save(model.state_dict(), model_path)

        # Log the .pt file as an artifact
        mlflow.log_artifact(model_path)

        # Example input and output for inference signature
        example_input = next(iter(dataloader))[0]  # Get a batch of images
        with torch.no_grad():
            model.eval()  # Switch to evaluation mode

            # Run model on the first image of the batch and convert output to numpy
            example_output = model([example_input[0]])  # The output is a tensor
        example_output_np = [{key: value.detach().cpu().numpy() for key, value in output.items()} for output in example_output]

        # Infer signature
        signature = infer_signature(np.expand_dims(example_input[0].detach().cpu().numpy(), axis=0), example_output_np)
        
        # Log the model with signature
        mlflow.pytorch.log_model(model, "model", signature=signature)
        mlflow.log_artifact(os.path.join(os.path.dirname(__file__), "train.py"))
        mlflow.log_artifact(os.path.join(repo_root, "requirements.txt"))

if __name__ == "__main__":
    main()

