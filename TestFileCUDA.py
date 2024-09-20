import torch

# Check CUDA availability and initialization status
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Ensure that it is properly installed and configured.")

try:
    torch.cuda.current_device()
except Exception as e:
    print(f"Error during CUDA initialization: {e}")
    exit(1)

print("CUDA is available and properly initialized.")
print(f"Current device: {torch.cuda.get_device_name()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Device index: {torch.cuda.current_device()}")
print(f"CUDA version: {torch.version.cuda}")

