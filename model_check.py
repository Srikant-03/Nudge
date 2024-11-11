import os
from ultralytics import YOLO
import time


model_dir = r'best_models/'  
test_source = r"F:\Drowsiness_Iot\datasets\test\images" 
img_size = 512
device = 0  


if not os.path.exists(test_source):
    raise FileNotFoundError(f"Test source directory does not exist: {test_source}")


model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]


model_results = {}


for model_path in model_files:
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path) 


    start_time = time.time()
    results = model.predict(source=test_source, imgsz=img_size, device=device, conf=0.25, save_txt=True)
    end_time = time.time()


    total_detections = sum(len(result.boxes) for result in results)


    model_results[model_path] = {
        'Total Detections': total_detections,
        'Inference Time (s)': end_time - start_time,
    }


print("\nModel Evaluation Results:")
for model, metrics in model_results.items():
    print(f"\nModel: {model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
