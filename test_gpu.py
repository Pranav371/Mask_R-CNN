import torch
import torchvision
import time

def test_gpu():
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("\n✓ CUDA is available! GPU acceleration can be used.")
        print("CUDA version:", torch.version.cuda)
        
        # Get number of GPUs
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}")
        
        # Print information for each GPU
        for i in range(device_count):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            
            # Get device properties for more detailed information
            props = torch.cuda.get_device_properties(i)
            print(f"- Total memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"- Multi processors: {props.multi_processor_count}")
            print(f"- CUDA Capability: {props.major}.{props.minor}")
            
        # Set device to CUDA
        device = torch.device("cuda")
        print("\nSelected device:", device)
        
        # Perform a simple test to see if operations work
        print("\nRunning a simple test...")
        
        # Create random tensor on CPU
        cpu_tensor = torch.randn(5000, 5000)
        
        # Measure time to move tensor to GPU
        start_time = time.time()
        gpu_tensor = cpu_tensor.to(device)
        cuda_sync_time = time.time() - start_time
        print(f"Time to move tensor to GPU: {cuda_sync_time:.4f} seconds")
        
        # Measure matrix multiplication time on GPU
        start_time = time.time()
        result = torch.matmul(gpu_tensor, gpu_tensor)
        torch.cuda.synchronize()  # Wait for operation to complete
        gpu_time = time.time() - start_time
        print(f"GPU matrix multiplication time: {gpu_time:.4f} seconds")
        
        # For comparison, measure matrix multiplication time on CPU
        start_time = time.time()
        result_cpu = torch.matmul(cpu_tensor, cpu_tensor)
        cpu_time = time.time() - start_time
        print(f"CPU matrix multiplication time: {cpu_time:.4f} seconds")
        
        print(f"GPU speedup: {cpu_time/gpu_time:.2f}x faster")
        
        # Test loading the Mask R-CNN model
        print("\nLoading Mask R-CNN model...")
        start_time = time.time()
        try:
            from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
            model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        except (ImportError, TypeError):
            model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        model = model.to(device)
        model_load_time = time.time() - start_time
        print(f"Model loaded in {model_load_time:.2f} seconds")
        
        # Get current GPU memory usage
        print(f"\nCurrent GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")
        
        print("\n✓ GPU test completed successfully!")
    else:
        print("\n✗ CUDA is not available. Running on CPU only.")
        print("Please check your NVIDIA drivers and PyTorch installation.")

if __name__ == "__main__":
    test_gpu() 