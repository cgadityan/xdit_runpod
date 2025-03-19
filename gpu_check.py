import torch

def check_gpu_health():
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No NVIDIA GPUs detected.")
        return
    
    print(f"Detected {num_gpus} NVIDIA GPU(s).\n")
    gpu_status = []

    # Check each GPU
    for gpu_id in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(gpu_id)
        try:
            print(f"Checking GPU {gpu_id}: {gpu_name}")
            
            # Set the GPU device
            device = torch.device(f"cuda:{gpu_id}")
            
            # Allocate a sample tensor on the GPU
            sample_tensor = torch.randn((100, 100)).to(device)
            
            # Perform a simple computation on the tensor
            result_tensor = sample_tensor * 2
            
            # Check if the computation succeeded
            if result_tensor is not None:
                print(f"GPU {gpu_id} is healthy.\n")
                gpu_status.append((gpu_id, gpu_name, "Healthy"))
            else:
                print(f"GPU {gpu_id} failed to load the tensor.\n")
                gpu_status.append((gpu_id, gpu_name, "Failed"))
        except Exception as e:
            print(f"GPU {gpu_id} encountered an error: {e}\n")
            gpu_status.append((gpu_id, gpu_name, f"Error: {e}"))

    # Print final status summary
    print("GPU Health Status Summary:")
    for gpu_id, gpu_name, status in gpu_status:
        print(f"GPU {gpu_id}: {gpu_name} - {status}")

if __name__ == "__main__":
    check_gpu_health()
