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
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import wandb
from visualize import plot_image_with_boxes
import torchmetrics
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from datetime import datetime

EXPERIMENT_NAME = "pidray_experiment_2"
NUM_EPOCHS = 2
NUM_IMAGES = 6
DATA_DIR = Path("data/pidray/train")
ANNOTATION_FILE = "data/pidray/annotations/xray_train.json"
LOG_FREQUENCY = 1 #frequency to which you log the larger items

FILE_LOCATION = Path(__file__) 
REPO_ROOT = FILE_LOCATION.parents[1]
TRACKING_URI = f"file://{REPO_ROOT / 'mlruns'}"

def main():
    # Setup TensorBoard 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=str(REPO_ROOT / 'tensorboard_logs'/f'run{timestamp}'))

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
            epoch_mAP = 0.0
            epoch_recall_50 = 0.0
            epoch_precision_50 = 0.0
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

                # Compute metrics using the updated function
                model.eval()  # Switch to evaluation mode to avoid affecting gradients
                with torch.no_grad():
                    outputs = model(images)
                mAP, recall_50, precision_50 = compute_metrics(outputs, targets)
                epoch_mAP += mAP
                epoch_recall_50 += recall_50
                epoch_precision_50 += precision_50

                num_batches += 1

                # Log batch loss and metrics to MLflow, TensorBoard, and W&B
                step = epoch * len(dataloader) + num_batches
                mlflow.log_metric("batch_loss", losses.item(), step=step)
                mlflow.log_metric("batch_mAP", mAP, step=step)
                mlflow.log_metric("batch_recall_50", recall_50, step=step)
                mlflow.log_metric("batch_precision_50", precision_50, step=step)
                writer.add_scalar("Loss/Batch", losses.item(), step)
                writer.add_scalar("mAP/Batch", mAP, step)
                writer.add_scalar("Recall@0.5/Batch", recall_50, step)
                writer.add_scalar("Precision@0.5/Batch", precision_50, step)
                wandb.log({
                    "batch_loss": losses.item(),
                    "batch_mAP": mAP,
                    "batch_recall_50": recall_50,
                    "batch_precision_50": precision_50,
                    "step": step
                })

                model.train()  # Switch back to training mode

            # Calculate and log epoch metrics
            epoch_loss /= num_batches
            epoch_mAP /= num_batches
            epoch_recall_50 /= num_batches
            epoch_precision_50 /= num_batches

            print(f"Epoch: {epoch}, Loss: {epoch_loss}, mAP: {epoch_mAP}, Recall@0.5: {epoch_recall_50}, Precision@0.5: {epoch_precision_50}")

            mlflow.log_metric("epoch_loss", epoch_loss, step=epoch)
            mlflow.log_metric("epoch_mAP", epoch_mAP, step=epoch)
            mlflow.log_metric("epoch_recall_50", epoch_recall_50, step=epoch)
            mlflow.log_metric("epoch_precision_50", epoch_precision_50, step=epoch)
            writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
            writer.add_scalar("mAP/Epoch", epoch_mAP, epoch)
            writer.add_scalar("Recall@0.5/Epoch", epoch_recall_50, epoch)
            writer.add_scalar("Precision@0.5/Epoch", epoch_precision_50, epoch)
            wandb.log({
                "epoch_loss": epoch_loss,
                "epoch_mAP": epoch_mAP,
                "epoch_recall_50": epoch_recall_50,
                "epoch_precision_50": epoch_precision_50,
                "epoch": epoch
            })

            if (epoch + 1) % LOG_FREQUENCY == 0:
                model.eval()  # Ensure the model is in evaluation mode
                with torch.no_grad():
                    sample_image_for_inference, sample_target = dataset[0]  # Get a sample image
                    sample_image_for_inference = sample_image_for_inference.unsqueeze(0)  # Add batch dimension
                    output = model(sample_image_for_inference)[0]

                # Plot the image with predicted bounding boxes
                fig, ax = plt.subplots(1, figsize=(12, 8))
                plot_image_with_boxes(sample_image_for_inference[0], output['boxes'], output['labels'], ax=ax)
                
                # Save the figure to a temporary file
                inference_image_path = REPO_ROOT / f"inference_epoch_{epoch + 1}.png"
                fig.savefig(inference_image_path, bbox_inches="tight")
                plt.close(fig)

                numpy_image = plt.imread(inference_image_path)
                if numpy_image.ndim == 3 and numpy_image.shape[2] == 4:
                    numpy_image = numpy_image[:, :, :3] 
                writer.add_image("Inference Image", numpy_image, global_step=epoch, dataformats='HWC')
                wandb.log({"inference_image": wandb.Image(numpy_image)})
                mlflow.log_image(Image.open(inference_image_path), artifact_file=f"inference_image_{epoch}.png")
                model.train()

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


def compute_metrics(outputs, targets):
    # Initialize the metric
    metric = MeanAveragePrecision(iou_type="bbox")

    # Update the metric with outputs and targets
    metric.update(outputs, targets)

    # Compute the results
    results = metric.compute()

    # Extract relevant metrics
    mAP = results['map'].item()  # Global mean average precision
    recall_50 = results['mar_1'].item()  # Mean average recall with max 1 detection per image, often used at IoU=0.5
    precision_50 = results['map_50'].item()  # Mean average precision at IoU=0.5

    return mAP, recall_50, precision_50





if __name__ == "__main__":
    main()
