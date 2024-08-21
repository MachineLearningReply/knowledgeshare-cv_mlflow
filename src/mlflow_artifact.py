from PIL import Image
from torchvision.transforms import functional as F
import mlflow
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pathlib import Path

# Set up paths
RUN_ID = "7822b622cb414464a25d38f053412d51"
DOWNLOAD_DIR = Path("./model_artifacts")
MODEL_WEIGHTS_NAME = "model_weights.pt"
TEST_IMAGE = Path("data") / "pidray" / "easy" / "xray_easy00000.png"

# Download the artifact to a local directory
mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path=MODEL_WEIGHTS_NAME, dst_path=DOWNLOAD_DIR
)

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
