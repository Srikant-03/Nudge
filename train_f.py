import os
import torch
from yolov5.train import run  # Adjust the import based on your YOLOv5 installation

def main():
    # Paths
    dataset_yaml_path = "F:/Drowsiness_Iot/datasets/data.yaml"  # Path to your data.yaml file
    pre_trained_weights = "F:/Drowsiness_Iot/yolov9t.pt"  # Path to pre-trained weights
    
    # Verify file paths
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(f"data.yaml not found at {dataset_yaml_path}")
    
    if not os.path.exists(pre_trained_weights):
        raise FileNotFoundError(f"Pre-trained weights file not found at {pre_trained_weights}")
    
    # Check for GPU availability
    if torch.cuda.is_available():
        device = "cuda:0"  # Use the first CUDA device
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA not available. Training will use the CPU.")

    # Train the model
    print("Starting training...")
    run(
        data=dataset_yaml_path,  # Path to data.yaml
        weights=pre_trained_weights,  # Pre-trained model weights
        batch_size=16,  # Adjust based on your GPU memory
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image resolution
        device=device  # Specify device (GPU or CPU)
    )

    print("Training complete!")

    # Path to the best weights file
    best_weights_path = "runs/train/exp/weights/best.pt"
    
    # Verify if training produced a best weights file
    if os.path.exists(best_weights_path):
        print(f"Training successful. Best weights saved at: {best_weights_path}")
    else:
        print("Training completed, but best weights file was not found!")

if __name__ == "__main__":
    main()
