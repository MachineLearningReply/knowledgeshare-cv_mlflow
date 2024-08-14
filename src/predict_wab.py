from PIL import Image
from torchvision.transforms import functional as F
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pathlib import Path
import wandb

# Set up paths
ENTITY_NAME = "sjva-machine-learning-reply-gmbh"
PROJECT_NAME = "pidray_experiment"
ARTIFACT_PATH = f"sjva-machine-learning-reply-gmbh/pidray_experiment/Runs/soft-snow-1/Files/knowledgeshare-cv_mlflow/model_weights.pt"
DOWNLOAD_DIR = Path("./model_artifacts_wandb")
MODEL_WEIGHTS_NAME = "model_weights.pt"
TEST_IMAGE = Path("data") / "pidray" / "easy" / "xray_easy09283.png"

# Initialize W&B API and download the artifact
api = wandb.Api()
artifact = api.artifact(ARTIFACT_PATH)
artifact_dir = artifact.download(DOWNLOAD_DIR)

# Load the model architecture and load the weights
model = fasterrcnn_resnet50_fpn(weights=None)  # Initialize without pre-trained weights
model.load_state_dict(torch.load(DOWNLOAD_DIR / MODEL_WEIGHTS_NAME))
model.eval()

# Open and prepare test image
image = Image.open(TEST_IMAGE).convert("RGB")
image = image.resize((400, 448))  # Resize to expected dimensions
transform = F.to_tensor
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    prediction = model(image_tensor)
print(prediction)
