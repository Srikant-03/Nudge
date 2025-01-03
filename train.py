import os
from ultralytics import YOLO
from ultralytics.utils.patches import imread
from datetime import datetime
import torch

def load_image(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} not found. Skipping.")
        return None
    return imread(file_path)

def clear_cuda_cache():
    """Clears the CUDA memory cache."""
    torch.cuda.empty_cache()
    print("CUDA cache cleared.")

def main():
    # Paths and Configuration
    data_yaml = 'datasets/data.yaml' 
    pretrained_weights = 'yolov9t.pt'  
    output_dir = f'runs/train/{datetime.now().strftime("%Y%m%d-%H%M%S")}'  

    os.makedirs(output_dir, exist_ok=True)

    # Set device to 'cuda:0' explicitly
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Print the device being used
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = YOLO(pretrained_weights)

    training_params = {
        'data': data_yaml,               
        'device': device,                
        'epochs': 120,                   
        'imgsz': 640,                    
        'batch': 32,                      
        'workers': 8,                    
        'optimizer': 'AdamW',            
        'lr0': 0.001,                    
        'lrf': 0.01,                     
        'momentum': 0.937,               
        'weight_decay': 0.0001,          
        'project': output_dir,           
        'name': 'ultimate_model',        
        'patience': 20,                  
        'augment': True,                 
        'cache': 'disk',                 
        'dropout': 0.5,                  
        'half': True,                    
        'save_period': 5,                
        'seed': 42,                      
        'verbose': True,                 
        'amp': True                      
    }

    checkpoint_path = os.path.join(output_dir, 'weights/last.pt')
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        training_params['resume'] = True

    # Clear CUDA cache before starting training
    clear_cuda_cache()

    print("Starting model training...")
    model.train(**training_params)

    # Clear CUDA cache before hyperparameter tuning
    clear_cuda_cache()

    print("\nStarting hyperparameter tuning...")
    model.tune(data=data_yaml, epochs=50, imgsz=1280)

    # Clear CUDA cache before validation
    clear_cuda_cache()

    print("\nValidating the model on validation set...")
    validation_results = model.val(data=data_yaml, imgsz=1280, iou=0.6)
    print("Validation Results:", validation_results)

    test_images = 'path/to/test/images'
    print("\nRunning inference on test data...")

    # Clear CUDA cache before inference
    clear_cuda_cache()

    inference_results = model.predict(source=test_images, conf=0.5, save=True)
    print("Inference complete. Check saved results.")

    print("\nGenerating advanced training metrics...")
    model.export(format='csv')  
    print(f"Training and validation logs saved to {output_dir}.")

if __name__ == '__main__':
    main()
