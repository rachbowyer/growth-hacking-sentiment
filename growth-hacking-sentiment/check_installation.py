import torch

print(torch.__version__)
print(torch.version.cuda)  # should show 12.x
print(torch.cuda.is_available())  # should be True
