import json
import torch
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import mlflow
from mlflow.models import infer_signature
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb

EXPERIMENT_NAME = "pidray_experiment_2"
NUM_EPOCHS = 6
NUM_IMAGES = 6
DATA_DIR = Path("data/pidray/train")
ANNOTATION_FILE = "data/pidray/annotations/xray_train.json"

FILE_LOCATION = Path(__file__) 
REPO_ROOT = FILE_LOCATION.parents[1]
TRACKING_URI = f"file://{REPO_ROOT / 'mlruns'}"

def main():
    # Setup TensorBoard 
    writer = SummaryWriter(log_dir=str(REPO_ROOT / 'tensorboard_logs'))

    # Setup Weights & Biases
    wandb.init(project="pidray_experiment", config={"epochs": NUM_EPOCHS})

    # Setup MLFlow
    mlflow.set_tracking_uri(TRACKING_URI)
    setup_experiment(EXPERIMENT_NAME)

    # Load dataset
    dataset = PIDrayDataset(DATA_DIR, ANNOTATION_FILE, transform=F.to_tensor)

    # Select few images
    dataset = torch.utils.data.Subset(dataset, range(NUM_IMAGES))
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=True, collate_fn=lambda b: tuple(zip(*b))
    )

    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.train()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Start the MLflow run
    with mlflow.start_run():
        # Training loop
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0

            for images, targets in dataloader:
                images = list(image for image in images)
                targets = [{k: v for k, v in t.items()} for t in targets]

                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                # Update loss
                epoch_loss += losses.item()

                # Compute accuracy
                model.eval()  # Switch to evaluation mode to avoid affecting gradients
                with torch.no_grad():
                    outputs = model(images)
                batch_accuracy = compute_accuracy(outputs, targets)
                epoch_accuracy += batch_accuracy

                num_batches += 1

                # Log batch loss and accuracy to MLflow, TensorBoard, and W&B
                step = epoch * len(dataloader) + num_batches
                mlflow.log_metric("batch_loss", losses.item(), step=step)
                mlflow.log_metric("batch_accuracy", batch_accuracy, step=step)
                writer.add_scalar("Loss/Batch", losses.item(), step)
                writer.add_scalar("Accuracy/Batch", batch_accuracy, step)
                wandb.log({"batch_loss": losses.item(), "batch_accuracy": batch_accuracy, "step": step})

                model.train()  # Switch back to training mode

            # Calculate and log epoch metrics
            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            print(f"Epoch: {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("epoch_accuracy", epoch_accuracy, step=epoch)
            writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
            writer.add_scalar("Accuracy/Epoch", epoch_accuracy, epoch)
            wandb.log({"epoch_loss": epoch_loss, "epoch_accuracy": epoch_accuracy, "epoch": epoch})

        # Save the trained model's state_dict to a .pt file
        model_path = REPO_ROOT / "model_weights.pt"
        torch.save(model.state_dict(), model_path)

        # Log the .pt file as an artifact
        mlflow.log_artifact(model_path)
        wandb.save(str(model_path))

        # Example input and output for inference signature
        example_input = next(iter(dataloader))[0]  # Get a batch of images
        with torch.no_grad():
            model.eval()  # Switch to evaluation mode

            # Run model on the first image of the batch and convert output to numpy
            example_output = model([example_input[0]])  # The output is a tensor
        example_output_np = [
            {key: value.detach().cpu().numpy() for key, value in output.items()}
            for output in example_output
        ]

        # Infer signature
        signature = infer_signature(
            np.expand_dims(example_input[0].detach().cpu().numpy(), axis=0),
            example_output_np,
        )

        # Log the model with signature 
        mlflow.pytorch.log_model(model, "model", signature=signature)
        mlflow.log_artifact(FILE_LOCATION)
        mlflow.log_artifact(REPO_ROOT / "requirements.txt")

        # Close TensorBoard writer
        writer.close()



class PIDrayDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)["annotations"]
        self.image_ids = list(set([anno["image_id"] for anno in self.annotations]))

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = self.root_dir / f"xray_{str(img_id).zfill(5)}.png"
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for anno in self.annotations:
            if anno["image_id"] == img_id:
                x, y, width, height = anno["bbox"]
                boxes.append([x, y, x + width, y + height])
                labels.append(anno["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, target


def setup_experiment(experiment_name):
    # Set the tracking URI

    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        # Create a new experiment if it doesn't exist, with artifact_location set to tracking_uri
        experiment_id = mlflow.create_experiment(name=experiment_name)
        print(f"Created new experiment with id {experiment_id}")
    else:
        # Use the existing experiment
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with id {experiment_id}")

    mlflow.set_experiment(experiment_name)


def compute_accuracy(outputs, targets, iou_threshold=0.5):
    correct = 0
    total = 0

    for output, target in zip(outputs, targets):
        pred_boxes = output["boxes"].cpu()
        pred_labels = output["labels"].cpu()
        true_boxes = target["boxes"].cpu()
        true_labels = target["labels"].cpu()

        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            continue

        ious = box_iou(pred_boxes, true_boxes)
        matched_indices = ious.max(dim=1).indices

        for i, pred_label in enumerate(pred_labels):
            if ious[i, matched_indices[i]] >= iou_threshold:
                if pred_label == true_labels[matched_indices[i]]:
                    correct += 1

        total += len(true_labels)

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    main()
