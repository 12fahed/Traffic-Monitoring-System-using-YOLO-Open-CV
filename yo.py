import torch
import torchvision
import tensorflow as tf

print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())