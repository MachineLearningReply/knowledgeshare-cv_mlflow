from PIL import Image
from torchvision.transforms import functional as F
import mlflow
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pathlib import Path

# Set up paths
MODEL_NAME = "pidray1"
MODEL_VERSION = 1  # Specify the version of the model you want to load
TEST_IMAGE = Path("data") / "pidray" / "easy" / "xray_easy00000.png"

# Load the model from the MLflow Model Registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
model = mlflow.pytorch.load_model(model_uri)

# Open and prepare test image
image = Image.open(TEST_IMAGE).convert("RGB")
image = image.resize((400, 448))  # Resize to expected dimensions
transform = F.to_tensor
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
model.eval()
with torch.no_grad():
    prediction = model(image_tensor)
    
print(prediction)
