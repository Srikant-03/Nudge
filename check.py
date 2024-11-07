import torch
import torchvision
print(torch.__version__)  # Should match torchvision's version
print(torchvision.__version__)
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should print your GPU name
