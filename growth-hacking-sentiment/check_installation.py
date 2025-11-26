import torch

print(f"Torch version: {torch.__version__}")
print(f"Torch cuda version: {torch.version.cuda}")  # should show 12.x
print(f"Torch cuda available?: {torch.cuda.is_available()}")  # should be True
