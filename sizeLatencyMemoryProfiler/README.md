# PyTorch Model Profiler

This tool helps engineers understand the trade-offs between batch size, inference latency, and GPU memory consumption.

## Features

- **Model Size Calculation:** Computes the footprint of parameters and buffers.
- **Stabilized Latency Benchmarking:** Uses `time.perf_counter` and `torch.cuda.synchronize()` to ensure accurate measurements.
- **Memory Tracking:** Monitors allocated and peak VRAM usage during the forward pass.
