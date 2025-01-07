import os
from ultralytics import YOLO
import torch
from datetime import datetime

# Set environment variable for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main():
    # Paths and Configuration
    data_yaml = 'datasets/data.yaml'  # Path to your dataset YAML file
    pretrained_weights = 'yolov9t.pt'  # Path to your pre-trained weights
    output_dir = f'runs/train/{datetime.now().strftime("%Y%m%d-%H%M%S")}'  # Output directory for training results

    os.makedirs(output_dir, exist_ok=True)

    # Manually specify the CUDA device to GPU 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the YOLO model with pre-trained weights
    model = YOLO(pretrained_weights)
    model.to(device)

    # Simplified training parameters with memory management enhancements
    training_params = {
        'data': data_yaml,               # Dataset YAML file
        'device': device,                # Device to run on ('cuda' or 'cpu')
        'epochs': 50,                    # Number of epochs
        'imgsz': 512,                    # Reduced image size for memory efficiency
        'batch': 8,                      # Reduced batch size
        'project': output_dir,           # Output directory
        'name': 'simple_model',          # Name for the run
        'amp': True,                     # Enable Automatic Mixed Precision
    }

    # Start model training
    print("Starting model training...")
    model.train(**training_params)

    print("Training complete.")

if __name__ == '__main__':
    main()
