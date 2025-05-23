import gc
import os
import torch
import wandb
import yaml
import datetime

dt = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def read_config(config_path):
    """Read configuration from a file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def clear_cuda_memory():
    """Clear CUDA memory cache and force garbage collection."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

def print_cuda_memory_stats():
    """Print detailed CUDA memory statistics."""
    if torch.cuda.is_available():
        # Get memory stats for current device
        current_device = torch.cuda.current_device()
        
        # Get allocated and reserved memory (in bytes)
        allocated = torch.cuda.memory_allocated(current_device)
        reserved = torch.cuda.memory_reserved(current_device)
        
        # Get maximum allocated and reserved memory
        max_allocated = torch.cuda.max_memory_allocated(current_device)
        max_reserved = torch.cuda.max_memory_reserved(current_device)
        
        # Convert to more readable format (GB)
        def bytes_to_gb(x):
            return x / (1024 ** 3)
        
        print(f"\n----- CUDA Memory Stats (Device {current_device}) -----")
        print(f"  Allocated: {bytes_to_gb(allocated):.2f} GB (Max: {bytes_to_gb(max_allocated):.2f} GB)")
        print(f"  Reserved:  {bytes_to_gb(reserved):.2f} GB (Max: {bytes_to_gb(max_reserved):.2f} GB)")
        print(f"  Utilization: {(allocated / max(reserved, 1) * 100):.1f}%")
        
        if torch.cuda.device_count() > 1:
            print("Memory stats for all devices:")
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i)
                res = torch.cuda.memory_reserved(i)
                print(f"  Device {i}: {bytes_to_gb(alloc):.2f} GB allocated, "
                      f"{bytes_to_gb(res):.2f} GB reserved")
        print("-------------------------------------------------\n")
    else:
        print("CUDA not available")

def wandb_init(stage=None):
    """Initialize Weights & Biases (wandb) for experiment tracking."""
    wandb.init(
        project="accfiy",
        name = f"Stage{stage}Experiment_{dt}",
    )
    wandb.run.name = "experiment_name"
    wandb.run.save()