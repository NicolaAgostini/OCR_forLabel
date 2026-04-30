# check_gpu.py


import torch
import sys

print(f"Versione Python: {sys.version}")
print(f"Versione PyTorch: {torch.__version__}")
print(f"CUDA Disponibile: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    try:
        import torchao
        print(f"Versione Torchao: {torchao.__version__}")
        # Test per l'errore int1
        x = torch.int1
        print("Test int1: SUPERATO")
    except Exception as e:
        print(f"Test int1: FALLITO ({e})")
else:
    print("ERRORE: La GPU non è rilevata. Controlla l'installazione di PyTorch CUDA.")


