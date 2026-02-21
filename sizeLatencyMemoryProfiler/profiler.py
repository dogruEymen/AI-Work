# Profiling Size, Latency, and Memory

import time
import torch
import torch.nn as nn
torch.manual_seed(0)

class SimpleMLP(nn.Module):
    def __init__(self, input_size=128, hidden_size=516, output_size=1):
        super().__init__()
        self.fc1   = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Move model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleMLP().to(device)

"""OBSERVING  MODEL SIZE """

def model_size_bytes(model):
    """Return model size in bytes (params + buffers)."""
    model_params = list(model.parameters()) + list(model.buffers())
    size = 0
    for tensor in model_params:
        size += tensor.numel() * tensor.element_size()
    return size

size_bytes = model_size_bytes(model)
print(f"[Model Size] {size_bytes:,} bytes, {(size_bytes / 1024 ** 2):.3f} MB.")

"""OBSERVING  MODEL LATENCY """
# Create test batches
x_base = torch.randn(1, 128, device=device)
batch_size1 = 64
x1 = x_base.expand(batch_size1, -1).contiguous()
batch_size2 = 8
x2 = x_base.expand(batch_size2, -1).contiguous()


def measure_latency(model, x, iters=50):
    """Return average latency (ms) per forward pass."""
    model.eval()
    start = time.perf_counter()
    iters = 50
    with torch.inference_mode():
        for _ in range(iters):
            _ = model(x)
            if x.device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency = elapsed / iters

    return latency

latency_x1 = measure_latency(model, x1, iters=50)
latency_x2 = measure_latency(model, x2, iters=50)


print(f"[Latency] {latency_x1:.3f} ms per forward pass (batch size={batch_size1})")
print(f"[Latency] {latency_x2:.3f} ms per forward pass (batch size={batch_size2})")

"""OBSERVING  MODEL MEMORY USAGE """
# Create test batches 
x_base = torch.randn(1, 128, device=device)
batch_size1 = 64
x1 = x_base.expand(batch_size1, -1).contiguous()
batch_size2 = 8
x2 = x_base.expand(batch_size2, -1).contiguous()

def measure_gpu_memory(model, x):
    """Return current and peak GPU memory in MB after one forward."""
    torch.cuda.empty_cache()            
    torch.cuda.reset_peak_memory_stats()
    
    model.eval()
    with torch.inference_mode():
        y = model(x)
        
    torch.cuda.synchronize()
    current = torch.cuda.memory_allocated() / (1024 ** 2)
    peak    = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return current, peak

mem1 = measure_gpu_memory(model, x1)
mem2 = measure_gpu_memory(model, x2)


print(f"[GPU Memory] Current allocated: {mem1[0]:.2f} MB | Peak during forward: {mem1[1]:.2f} MB (batch size={batch_size1})")
print(f"[GPU Memory] Current allocated: {mem2[0]:.2f} MB | Peak during forward: {mem2[1]:.2f} MB (batch size={batch_size2})")


""" ---Clean up session---"""
import gc, torch

del model, x1, x2 
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()