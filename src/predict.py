from PIL import Image
from torchvision.transforms import functional as F
import mlflow
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from pathlib import Path

# Set up paths
run_id = '71277e0d48b14cc8a960eb8430a74f5d'
data_dir = Path('../data')
image_path = data_dir / 'pidray' / 'easy' / 'xray_easy09283.png'

# Open and resize the image
image = Image.open(image_path).convert("RGB")
image = image.resize((400, 448))  # Resize to expected dimensions

# Load the model
# model_uri = f"runs:/{run_id}/model"
# model = mlflow.pyfunc.load_model(model_uri)

# Transform the image and add batch dimension
transform = F.to_tensor
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Convert to numpy array for model input
#input_example = image_tensor.numpy()

# Replace with your actual run_id and artifact path
artifact_path = 'model_weights.pt'
local_dir = './model_artifacts'
local_path = os.path.join(local_dir, artifact_path)

# Download the artifact to a local directory
mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=local_dir)
# Load the model architecture
model = fasterrcnn_resnet50_fpn(weights=None)  # Initialize without pre-trained weights

# Load the state dictionary from the downloaded .pt file
model.load_state_dict(torch.load(local_path))

# Set the model to evaluation mode
model.eval()

with torch.no_grad():
    prediction = model(image_tensor)

# Print or process the prediction
print(prediction)