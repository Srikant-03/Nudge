import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Configuration
model_dir = r'best_models/'
test_source = r"F:\Drowsiness_Iot\datasets\test\images"
ground_truth = r"F:\Drowsiness_Iot\datasets\test\labels"  # Ground truth labels
img_size = 512
device = 0

# Check paths
if not os.path.exists(test_source):
    raise FileNotFoundError(f"Test source directory does not exist: {test_source}")
if not os.path.exists(ground_truth):
    raise FileNotFoundError(f"Ground truth labels directory does not exist: {ground_truth}")

# Get list of model files
model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]

# Store results
model_results = {}

# Evaluate each model
for model_path in model_files:
    print(f"Evaluating model: {model_path}")
    model = YOLO(model_path)

    # Perform inference
    start_time = time.time()
    results = model.predict(source=test_source, imgsz=img_size, device=device, conf=0.25, save_txt=True)
    end_time = time.time()

    # Calculate evaluation metrics
    all_predictions = []
    all_ground_truths = []

    for result in results:
        # Convert YOLO predictions to usable format
        predictions = result.boxes.xywh.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy()

        all_predictions.extend(labels)
        # Load corresponding ground truth labels (assumes same naming convention)
        label_path = os.path.join(ground_truth, os.path.basename(result.path).replace('.jpg', '.txt'))
        if os.path.exists(label_path):
            gt_labels = np.loadtxt(label_path, usecols=0)  # Assuming YOLO format
            all_ground_truths.extend(gt_labels)

    # Compute metrics
    if all_predictions and all_ground_truths:
        precision = precision_score(all_ground_truths, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_ground_truths, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_ground_truths, all_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_ground_truths, all_predictions)
    else:
        precision, recall, f1, conf_matrix = 0, 0, 0, None

    # Store results
    model_results[model_path] = {
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Inference Time (s)': end_time - start_time,
        'Confusion Matrix': conf_matrix.tolist() if conf_matrix is not None else None,
    }

# Save results to JSON
results_file = 'model_evaluation_results.json'
with open(results_file, 'w') as f:
    json.dump(model_results, f, indent=4)
print(f"\nResults saved to {results_file}")

# Visualize results
# Bar chart for Precision, Recall, and F1-score
metrics = ['Precision', 'Recall', 'F1-Score']
models = list(model_results.keys())
values = {metric: [model_results[model][metric] for model in models] for metric in metrics}

plt.figure(figsize=(12, 6))
for i, metric in enumerate(metrics):
    plt.bar(np.arange(len(models)) + i * 0.2, values[metric], width=0.2, label=metric)

plt.xticks(np.arange(len(models)) + 0.3, [os.path.basename(model) for model in models], rotation=45)
plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Evaluation Metrics')
plt.legend()
plt.tight_layout()
plt.show()
