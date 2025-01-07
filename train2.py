import os
from yolov5 import train


data_yaml_path = r'F:\Drowsiness_Iot\datasets\data.yaml'
model_path = 'yolov9t.pt'
epochs = 50
batch_size = 4
img_size = 512
device = 0
half_precision = True


train_command = f'yolo task=detect mode=train model={model_path} data={data_yaml_path} epochs={epochs} batch={batch_size} imgsz={img_size} device={device} half={str(half_precision).lower()}'


os.system(train_command)
