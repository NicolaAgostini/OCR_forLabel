import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    # Prova a fare un calcolo banale in CUDA
    x = torch.randn(1).cuda()
    print("Calcolo test su GPU: RIUSCITO")