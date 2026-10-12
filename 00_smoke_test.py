import torch

print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
print("GPU    :", torch.cuda.get_device_name(0))
print("VRAM   :", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), "GB")
print("bf16   :", torch.cuda.is_bf16_supported())

# cálculo real na GPU, para provar que ela funciona e não só que foi detectada
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
for _ in range(50):
    c = a @ b
torch.cuda.synchronize()
print("Pico de VRAM no teste:", round(torch.cuda.max_memory_allocated() / 1024**2), "MB")
print("=== AMBIENTE OK ===")