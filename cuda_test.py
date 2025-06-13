import torch

# Check if CUDA is available
if torch.cuda.is_available():
    # Create a tensor and move it to the GPU
    tensor = torch.randn(3, 3).to('cuda')
    print(f"Tensor on CUDA: \n{tensor}")
else:
    print("CUDA is not available!")
