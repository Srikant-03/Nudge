import torch

def check_gpus():
    # Check if CUDA is available
    if torch.cuda.is_available():
        # Get the number of GPUs available
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")
        
        # List the GPUs
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available. Using CPU.")

if __name__ == '__main__':
    check_gpus()
