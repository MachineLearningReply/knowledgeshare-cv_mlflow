import time
import mlflow
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.transforms import functional as F
from pathlib import Path

# Set up paths
MODEL_NAME = "pidray1"
CURRENT_VERSION = None  # Start with no version loaded
TEST_IMAGE = Path("data") / "pidray" / "easy" / "xray_easy09283.png"

def load_and_deploy_model(version):
    print(f"[INFO] Loading and deploying model version {version}...")
    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    print(f"[INFO] Model version {version} successfully loaded and deployed.")
    return model

def check_for_new_version(current_version):
    print("[INFO] Checking for new model version...")
    # Get the latest version of the model from the registry
    latest_version_info = mlflow.client.MlflowClient().get_latest_versions(MODEL_NAME, stages=["None"])
    latest_version = max(int(version.version) for version in latest_version_info)
    print(f"[INFO] Latest version in the registry is: {latest_version}")

    if current_version is None or latest_version > current_version:
        print(f"[INFO] New version detected: {latest_version}. Current version: {current_version}")
        return latest_version
    
    print(f"[INFO] No new version found. Current version is still: {current_version}")
    return None

def run_inference(model):
    print("[INFO] Running inference on test image...")
    image = Image.open(TEST_IMAGE).convert("RGB")
    image = image.resize((400, 448))  # Resize to expected dimensions
    transform = F.to_tensor
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        prediction = model(image_tensor)
    
    print("[INFO] Inference completed. Prediction:")
    print(prediction)

# Main loop to watch for new versions
while True:
    new_version = check_for_new_version(CURRENT_VERSION)
    
    if new_version:
        model = load_and_deploy_model(new_version)
        CURRENT_VERSION = new_version
        
        # Example deployment: run inference with the new model
        run_inference(model)
    
    print("[INFO] Waiting for the next check...")
    # Wait for a while before checking again
    time.sleep(5)  # Check every 60 seconds
