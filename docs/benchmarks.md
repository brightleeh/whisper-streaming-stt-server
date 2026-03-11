# Hardware Performance & Sizing Guide

> **Last updated**: 2026-03-12
> 
> **Audio asset**: `stt_client/assets/hello.wav` (Korean, ~2s)

## Overview

This document provides measured performance data for the Whisper Streaming STT Server
across tested hardware configurations, and sizing guidelines for production deployments.

All benchmarks use the **small** Whisper model with realtime streaming mode,
VAD-continue, and `partial=true` + `emit_final_on_vad=true`.

## Test Environment

### Hardware

| | macOS ARM64 | Ubuntu x86_64 |
|---|---|---|
| CPU | Apple M4 Pro (12-core: 4E+8P) | AMD Ryzen 7 8845HS (8-core Zen 4, 3.8 GHz) |
| RAM | 24 GB (unified) | 32 GB |
| GPU | M4 Pro 16-core (integrated, shared memory) | NVIDIA RTX 4060 Laptop (8 GB VRAM, 100W TGP) |
| OS | macOS Tahoe 26.3 | Ubuntu 24.04 |

### Software

| Component | Version |
|---|---|
| Server | whisper-streaming-stt-server 1.0.0 |
| Python | 3.12 |
| faster-whisper | 1.1.1 |
| torch | 2.6.0 |
| mlx-whisper | 0.4.2 |
| CUDA / cuDNN | N/A (macOS) / 12.6 / 9.6 (Ubuntu) |

### Test Matrix

All configurations use model=**small**, iterations=3, warmup=1, realtime mode, chunk=100ms.

**Multi-pool matrix** (macos-cpu, macos-gpu, ubuntu-cpu, ubuntu-gpu):

| pool_size | channels |
|-----------|----------|
| 1 | 1, 3, 5, 10 |
| 2 | 1, 5, 10, 15 |
| 4 | 5, 10, 15, 20, 30 |

**Single-pool matrix** (macos-gpu-mlx, macos-gpu-1pool, ubuntu-gpu-1pool):

GPU backends were re-tested with pool=1 after confirming that GPU environments
perform optimally with a single model instance (see [GPU Pool Sizing](#gpu-pool-sizing-pool1-is-optimal)).

| pool_size | channels |
|-----------|----------|
| 1 | 1, 3, 5, 10, 15, 20, 30 |

### Backend Configurations (5 cases)

| Case | Hardware | Backend | Device | Compute Type |
|------|----------|---------|--------|-------------|
| macos-cpu | macOS ARM64 (M4 Pro, 24GB RAM) | faster_whisper | cpu | int8 |
| macos-gpu | macOS ARM64 (M4 Pro, 24GB RAM) | torch_whisper | mps | float32 ¹ |
| macos-gpu-mlx | macOS ARM64 (M4 Pro, 24GB RAM) | mlx_whisper | mps | float16 |
| ubuntu-cpu | Ubuntu x86_64 (Ryzen 7 8845HS, 32GB RAM) | faster_whisper | cpu | int8 |
| ubuntu-gpu | Ubuntu x86_64 (RTX 4060 8GB, 32GB RAM) | faster_whisper | cuda | float16 |

> ¹ torch_whisper forces float32 fallback on MPS due to PyTorch fp16 instability on Apple Silicon.

---

## Key Metrics

Each table reports the following from the load test output:

| Metric | Description | What to watch |
|--------|-------------|---------------|
| **RTF P50** | Median Real-Time Factor (decode time / audio duration) | Lower = faster. <1.0 means faster than realtime. |
| **Queue Wait P50/P95** | Time waiting for an available worker | Spikes when channels exceed pool capacity. |
| **Inference P50/P95** | Model execution time per decode | Stable regardless of load (hardware-bound). |
| **Decode Total P50/P95** | End-to-end: queue wait + inference + emit | User-perceived latency. |
| **Bottleneck** | Dominant phase (buffer_wait vs queue_wait vs inference) | Shifts to queue_wait when overloaded. |
| **Error Rate** | Failures / total sessions | Should be 0% within capacity. |

---

## Results: macOS ARM64 (M4 Pro, 24GB RAM)

### macos-cpu — faster_whisper (CPU, int8)

**Baseline RTF (pool=1, ch=1):** 0.646

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.646 | 0.389s | 0.396s | 4.617s | 4.844s | 13.696s | 13.923s | buffer_wait (63% of total) | 0/3 |
| 1 | 3 | 0.429 | 5.111s | 6.438s | 3.080s | 3.161s | 13.842s | 15.275s | buffer_wait (41% of total) | 0/9 |
| 1 | 5 | 0.259 | 14.217s | 14.383s | 3.063s | 3.076s | 22.936s | 23.100s | queue_wait (62% of total) | 0/15 |
| 1 | 10 | 0.129 | 37.233s | 37.590s | 3.062s | 3.071s | 45.954s | 46.297s | queue_wait (81% of total) | 0/30 |
| 2 | 1 | 0.647 | 0.387s | 0.389s | 4.614s | 4.616s | 13.680s | 13.689s | buffer_wait (63% of total) | 0/3 |
| 2 | 5 | 0.414 | 5.731s | 12.540s | 3.822s | 5.749s | 15.187s | 26.935s | queue_wait (38% of total) | 0/15 |
| 2 | 10 | 0.214 | 18.456s | 18.718s | 3.697s | 3.734s | 27.744s | 28.032s | queue_wait (67% of total) | 0/30 |
| 2 | 15 | 0.147 | 31.446s | 34.120s | 3.589s | 3.628s | 40.797s | 43.355s | queue_wait (77% of total) | 0/45 |
| 4 | 5 | 0.397 | 3.739s | 6.716s | 7.755s | 8.366s | 20.607s | 23.579s | buffer_wait (43% of total) | 0/15 |
| 4 | 10 | 0.273 | 10.859s | 11.381s | 5.798s | 5.941s | 22.186s | 22.894s | queue_wait (49% of total) | 0/30 |
| 4 | 15 | 0.183 | 21.480s | 23.149s | 5.828s | 5.970s | 32.985s | 34.684s | queue_wait (65% of total) | 0/45 |
| 4 | 20 | 0.135 | 32.379s | 35.414s | 5.848s | 5.967s | 43.718s | 47.067s | queue_wait (74% of total) | 0/60 |
| 4 | 30 | 0.089 | 55.134s | 56.585s | 5.908s | 6.046s | 66.652s | 68.157s | queue_wait (83% of total) | 0/90 |

### macos-gpu — torch_whisper (MPS, fp32)

Multi-pool results (pool=1,2,4). For optimized pool=1 sweep, see [macos-gpu (pool=1 sweep)](#macos-gpu-pool1-sweep--torch_whisper-mps-fp32).

**Baseline RTF (pool=1, ch=1):** 0.851

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.851 | 0.001s | 0.001s | 1.391s | 1.393s | 10.052s | 10.054s | buffer_wait (86% of total) | 0/3 |
| 1 | 3 | 0.839 | 0.302s | 0.615s | 1.342s | 1.433s | 10.292s | 10.528s | buffer_wait (84% of total) | 0/9 |
| 1 | 5 | 0.625 | 3.697s | 4.949s | 1.269s | 1.315s | 13.662s | 14.823s | buffer_wait (63% of total) | 0/15 |
| 1 | 10 | 0.454 | 6.950s | 8.186s | 0.871s | 0.903s | 13.450s | 14.679s | queue_wait (52% of total) | 0/30 |
| 2 | 1 | 0.846 | 0.001s | 0.002s | 1.402s | 1.535s | 10.087s | 10.187s | buffer_wait (86% of total) | 0/3 |
| 2 | 5 | 0.621 | 2.754s | 3.288s | 2.569s | 2.772s | 13.856s | 14.521s | buffer_wait (62% of total) | 0/15 |
| 2 | 10 | 0.436 | 6.179s | 7.204s | 1.816s | 1.827s | 13.614s | 14.644s | queue_wait (45% of total) | 0/30 |
| 2 | 15 | 0.295 | 12.810s | 14.191s | 1.789s | 2.008s | 20.186s | 21.404s | queue_wait (63% of total) | 0/45 |
| 4 | 5 | 0.560 | 2.002s | 3.551s | 4.527s | 5.274s | 15.288s | 16.218s | buffer_wait (56% of total) | 0/15 |
| 4 | 10 | 0.432 | 5.044s | 11.376s | 3.623s | 5.385s | 14.207s | 25.428s | buffer_wait (39% of total) | 0/30 |
| 4 | 15 | 0.275 | 12.407s | 15.107s | 3.812s | 4.299s | 21.715s | 24.877s | queue_wait (57% of total) | 0/45 |
| 4 | 20 | 0.219 | 17.946s | 18.864s | 3.615s | 3.695s | 27.107s | 28.140s | queue_wait (66% of total) | 0/60 |
| 4 | 30 | 0.146 | 31.420s | 32.412s | 3.591s | 3.730s | 40.622s | 41.532s | queue_wait (77% of total) | 0/90 |

### macos-gpu (pool=1 sweep) — torch_whisper (MPS, fp32)

Re-tested with pool=1 only across 1–30 channels, after confirming GPU pool=1 is optimal.

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.446 | 0.002s | 0.002s | 1.434s | 1.453s | 18.742s | 19.220s | buffer_wait (92% of total) | 0/3 |
| 1 | 3 | 0.462 | 0.002s | 0.244s | 1.386s | 1.452s | 18.137s | 19.410s | buffer_wait (92% of total) | 0/9 |
| 1 | 5 | 0.442 | 0.429s | 0.699s | 1.334s | 1.446s | 19.045s | 19.578s | buffer_wait (91% of total) | 0/15 |
| 1 | 10 | 0.324 | 7.783s | 10.121s | 1.261s | 1.263s | 26.664s | 28.769s | buffer_wait (66% of total) | 0/30 |
| 1 | 15 | 0.212 | 18.603s | 22.803s | 1.274s | 1.287s | 38.232s | 42.416s | queue_wait (49% of total) | 0/45 |
| 1 | 20 | 0.224 | 14.981s | 31.279s | 0.884s | 1.299s | 27.844s | 51.327s | queue_wait (54% of total) | 0/60 |
| 1 | 30 | 0.151 | 26.518s | 29.065s | 0.875s | 0.884s | 39.635s | 41.982s | queue_wait (67% of total) | 0/90 |

### macos-gpu-mlx — mlx_whisper (MPS, fp16)

MLX enforces pool=1 due to thread-safety constraints. Tested across 1–30 channels.

**Baseline RTF (pool=1, ch=1):** 0.900

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.900 | 0.001s | 0.001s | 0.706s | 0.713s | 9.331s | 9.340s | buffer_wait (92% of total) | 0/3 |
| 1 | 3 | 0.898 | 0.026s | 0.131s | 0.674s | 0.685s | 9.349s | 9.405s | buffer_wait (92% of total) | 0/9 |
| 1 | 5 | 0.899 | 0.098s | 0.540s | 0.638s | 0.675s | 9.407s | 9.791s | buffer_wait (92% of total) | 0/15 |
| 1 | 10 | 0.619 | 4.631s | 5.863s | 0.662s | 0.674s | 13.917s | 15.149s | buffer_wait (62% of total) | 0/30 |
| 1 | 15 | 0.517 | 7.030s | 12.099s | 0.668s | 0.679s | 16.298s | 21.386s | buffer_wait (53% of total) | 0/45 |
| 1 | 20 | 0.440 | 7.705s | 8.758s | 0.452s | 0.457s | 13.776s | 14.829s | queue_wait (56% of total) | 0/60 |
| 1 | 30 | 0.298 | 14.074s | 15.272s | 0.447s | 0.456s | 20.124s | 21.294s | queue_wait (70% of total) | 0/90 |

---

## Results: Ubuntu x86_64 (Ryzen 7 8845HS, RTX 4060 8GB, 32GB RAM)

### ubuntu-cpu — faster_whisper (CPU, int8)

**Baseline RTF (pool=1, ch=1):** 0.475

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.475 | 2.772s | 2.869s | 6.923s | 7.039s | 18.079s | 18.336s | buffer_wait (47% of total) | 0/3 |
| 1 | 3 | 0.288 | 10.480s | 10.738s | 4.584s | 4.642s | 20.517s | 20.842s | queue_wait (51% of total) | 0/9 |
| 1 | 5 | 0.175 | 23.842s | 24.183s | 4.503s | 4.584s | 33.786s | 34.127s | queue_wait (71% of total) | 0/15 |
| 1 | 10 | 0.087 | 58.274s | 58.731s | 4.560s | 4.671s | 68.194s | 68.705s | queue_wait (85% of total) | 0/30 |
| 2 | 1 | 0.470 | 2.765s | 2.886s | 7.010s | 7.131s | 18.145s | 18.400s | buffer_wait (46% of total) | 0/3 |
| 2 | 5 | 0.297 | 9.917s | 10.773s | 5.307s | 5.485s | 20.710s | 21.611s | queue_wait (48% of total) | 0/15 |
| 2 | 10 | 0.150 | 28.742s | 29.483s | 5.251s | 5.406s | 39.438s | 40.160s | queue_wait (73% of total) | 0/30 |
| 2 | 15 | 0.099 | 49.167s | 50.432s | 5.325s | 5.451s | 59.926s | 61.137s | queue_wait (82% of total) | 0/45 |
| 4 | 5 | 0.379 | 2.252s | 5.316s | 8.304s | 8.732s | 16.074s | 18.913s | inference (52% of total) | 0/15 |
| 4 | 10 | 0.180 | 19.822s | 20.633s | 8.748s | 9.210s | 33.705s | 34.865s | queue_wait (59% of total) | 0/30 |
| 4 | 15 | 0.124 | 34.387s | 36.827s | 8.616s | 8.969s | 48.351s | 50.934s | queue_wait (71% of total) | 0/45 |
| 4 | 20 | 0.090 | 51.536s | 52.755s | 8.658s | 9.302s | 65.794s | 67.129s | queue_wait (78% of total) | 0/60 |
| 4 | 30 | 0.063 | 82.138s | 83.452s | 8.438s | 8.694s | 95.800s | 97.218s | queue_wait (86%) | **75/15** ¹ |

> ¹ **ubuntu-cpu pool=4/ch=30:** System exceeded `decode_timeout_sec`, causing 75 session failures
> out of 90 total (only 15 succeeded). This demonstrates the CPU scaling wall under extreme load.

### ubuntu-gpu — faster_whisper (CUDA, fp16)

Multi-pool results (pool=1,2,4). For optimized pool=1 sweep, see [ubuntu-gpu (pool=1 sweep)](#ubuntu-gpu-pool1-sweep--faster_whisper-cuda-fp16).

**Baseline RTF (pool=1, ch=1):** 0.907

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.907 | 0.002s | 0.002s | 0.798s | 0.862s | 9.190s | 9.266s | buffer_wait (91% of total) | 0/3 |
| 1 | 3 | 0.895 | 0.072s | 0.184s | 0.709s | 0.794s | 9.287s | 9.410s | buffer_wait (91% of total) | 0/9 |
| 1 | 5 | 0.878 | 0.238s | 0.646s | 0.676s | 0.762s | 9.312s | 9.783s | buffer_wait (91% of total) | 0/15 |
| 1 | 10 | 0.709 | 3.120s | 3.821s | 0.568s | 0.629s | 12.205s | 12.775s | buffer_wait (69% of total) | 0/30 |
| 2 | 1 | 0.913 | 0.002s | 0.002s | 0.812s | 0.952s | 9.199s | 9.339s | buffer_wait (91% of total) | 0/3 |
| 2 | 5 | 0.784 | 0.334s | 1.278s | 1.556s | 1.794s | 10.368s | 10.891s | buffer_wait (82% of total) | 0/15 |
| 2 | 10 | 0.518 | 6.280s | 7.731s | 1.562s | 1.843s | 16.452s | 17.618s | buffer_wait (51% of total) | 0/30 |
| 2 | 15 | 0.477 | 6.400s | 12.590s | 1.069s | 1.689s | 12.980s | 22.703s | queue_wait (49% of total) | 0/45 |
| 4 | 5 | 0.689 | 0.228s | 1.738s | 3.424s | 3.751s | 12.197s | 13.261s | buffer_wait (69% of total) | 0/15 |
| 4 | 10 | 0.464 | 5.812s | 7.675s | 3.428s | 3.780s | 17.678s | 19.520s | buffer_wait (47% of total) | 0/30 |
| 4 | 15 | 0.471 | 5.342s | 10.546s | 2.229s | 2.826s | 13.066s | 21.782s | buffer_wait (42% of total) | 0/45 |
| 4 | 20 | 0.359 | 9.118s | 10.727s | 2.246s | 2.451s | 16.816s | 18.480s | queue_wait (54% of total) | 0/60 |
| 4 | 30 | 0.241 | 16.859s | 18.743s | 2.211s | 2.507s | 24.407s | 26.445s | queue_wait (69% of total) | 0/90 |

### ubuntu-gpu (pool=1 sweep) — faster_whisper (CUDA, fp16)

Re-tested with pool=1 only across 1–30 channels, after confirming GPU pool=1 is optimal.

| pool | ch | RTF P50 | Queue P50 | Queue P95 | Infer P50 | Infer P95 | Total P50 | Total P95 | Bottleneck | Errors |
|------|-----|---------|-----------|-----------|-----------|-----------|-----------|-----------|------------|--------|
| 1 | 1 | 0.916 | 0.002s | 0.002s | 0.740s | 0.766s | 9.105s | 9.130s | buffer_wait (92% of total) | 0/3 |
| 1 | 3 | 0.915 | 0.043s | 0.123s | 0.650s | 0.789s | 9.150s | 9.450s | buffer_wait (92% of total) | 0/9 |
| 1 | 5 | 0.895 | 0.252s | 0.610s | 0.619s | 0.726s | 9.370s | 9.668s | buffer_wait (91% of total) | 0/15 |
| 1 | 10 | 0.676 | 3.667s | 4.998s | 0.587s | 0.664s | 12.747s | 14.036s | buffer_wait (66% of total) | 0/30 |
| 1 | 15 | 0.456 | 9.412s | 10.848s | 0.592s | 0.697s | 18.664s | 19.966s | queue_wait (50% of total) | 0/45 |
| 1 | 20 | 0.486 | 6.920s | 14.656s | 0.416s | 0.645s | 12.894s | 23.678s | queue_wait (54% of total) | 0/60 |
| 1 | 30 | 0.327 | 12.493s | 13.710s | 0.400s | 0.469s | 18.393s | 19.593s | queue_wait (68% of total) | 0/90 |

---

## Cross-Hardware Comparison

### Baseline Performance (pool=1, ch=1, model=small)

| Case | RTF P50 | Inference P50 | Total P50 | Notes |
|------|---------|---------------|-----------|-------|
| ubuntu-gpu | 0.907 | 0.798s | 9.190s | CUDA fp16 — fastest inference |
| macos-gpu-mlx | 0.900 | 0.706s | 9.331s | MLX fp16 — comparable to CUDA |
| macos-gpu | 0.851 | 1.391s | 10.052s | MPS fp32 fallback — ~2x slower inference |
| macos-cpu | 0.646 | 4.617s | 13.696s | int8 on Apple Silicon — moderate |
| ubuntu-cpu | 0.475 | 6.923s | 18.079s | int8 on Zen 4 — slowest |

**Key observations:**

- MLX on Apple Silicon achieves inference parity with CUDA on RTX 4060 (~0.7s vs ~0.8s).
- torch_whisper on MPS is ~2x slower than MLX due to forced fp32 fallback.
- CPU backends are 6–10x slower on inference than GPU backends.
- Higher baseline RTF (closer to 1.0) indicates the system spends more time waiting
  for audio to arrive (buffer_wait dominant), which is the ideal low-load state.

### GPU Pool=1 Scaling Comparison (1–30 channels)

After confirming that GPU backends perform best with pool=1, the three GPU configurations
were tested across an identical 1–30 channel sweep:

| ch | ubuntu-gpu RTF | ubuntu-gpu Total P50 | macos-gpu-mlx RTF | macos-gpu-mlx Total P50 | macos-gpu RTF | macos-gpu Total P50 |
|----|----------------|----------------------|-------------------|-------------------------|---------------|---------------------|
| 1 | 0.916 | 9.105s | 0.900 | 9.331s | 0.446 | 18.742s |
| 5 | 0.895 | 9.370s | 0.899 | 9.407s | 0.442 | 19.045s |
| 10 | 0.676 | 12.747s | 0.619 | 13.917s | 0.324 | 26.664s |
| 15 | 0.456 | 18.664s | 0.517 | 16.298s | 0.212 | 38.232s |
| 20 | 0.486 | 12.894s | 0.440 | 13.776s | 0.224 | 27.844s |
| 30 | 0.327 | 18.393s | 0.298 | 20.124s | 0.151 | 39.635s |

CUDA and MLX track closely at low channel counts, with CUDA maintaining a slight edge
in total latency at higher concurrency. torch_whisper (MPS fp32) shows consistently
higher total latency due to the fp32 overhead, despite its lower RTF numbers.

### Saturation Point (Queue Wait P95 > 1.0s)

The "saturation point" is the channel count where queue wait becomes significant
and P95 total latency begins to degrade noticeably.

| Case | pool=1 | pool=2 | pool=4 |
|------|--------|--------|--------|
| macos-cpu | ~3 ch | ~5 ch | ~10 ch |
| macos-gpu | ~5 ch | ~5 ch | ~10 ch |
| macos-gpu-mlx | ~10 ch | N/A (pool=1 only) | N/A |
| ubuntu-cpu | ~1 ch ¹ | ~5 ch | ~5 ch |
| ubuntu-gpu | ~5 ch | ~5 ch | ~5 ch |

> ¹ ubuntu-cpu shows 2.8s queue wait even at ch=1 due to slow CPU inference (~7s per decode),
> meaning the queue begins accumulating from the very first concurrent session.

### GPU Pool Sizing: pool=1 Is Optimal

Multi-pool benchmarks (pool=2, pool=4) consistently show that increasing pool size
on GPU backends causes severe inference time degradation due to hardware resource contention:

| Config | pool=1 Infer P50 | pool=4 Infer P50 | Degradation |
|--------|------------------|------------------|-------------|
| ubuntu-gpu (CUDA) | 0.57–0.80s | 2.2–3.4s | **3–5x slower** |
| macos-gpu (MPS) | 0.87–1.39s | 3.6–4.5s | **3–4x slower** |

GPU environments (CUDA, MPS, MLX) perform optimally with `pool=1` because multiple
model instances compete for limited VRAM bandwidth and streaming multiprocessors (CUDA)
or the unified memory bus (MPS/MLX), causing context switching overhead that negates
the benefit of additional workers.

CPU environments benefit from larger pool sizes (up to physical core count) since
each worker can leverage independent CPU cores, though L3 cache and memory bandwidth
contention still produces diminishing returns beyond pool=2–4.

**Recommendation:** For GPU backends, always use `pool=1`. For CPU backends,
use `pool=2` for moderate concurrency or `pool=4` for high-throughput batch workloads,
keeping in mind the inference degradation trade-off.

---

## Sizing Guide

### Formula

```
pool_size >= ceil(concurrent_sessions × RTF)
```

Where:
- `concurrent_sessions`: number of channels that may VAD-trigger simultaneously
- `RTF`: Real-Time Factor from baseline measurement (pool=1, ch=1)

> **Note for GPU backends:** Since pool=1 is optimal, scaling beyond single-worker capacity
> requires horizontal scaling (multiple server instances) rather than increasing pool_size.

### Quick Reference

Based on measured RTF values (pool=1 baseline):

| Target Channels | macos-cpu pool | macos-gpu pool | macos-gpu-mlx pool | ubuntu-cpu pool | ubuntu-gpu pool |
|----------------|---------------|---------------|-------------------|----------------|----------------|
| 5 | 4 | 1 | 1 | 3 | 1 |
| 10 | 4+ ¹ | 1 ² | 1 | 4+ ¹ | 1 |
| 20 | scale out | 1 ² | 1 ² | scale out | 1 ² |
| 30 | scale out | scale out | 1 ³ | scale out | scale out |

> ¹ Queue wait becomes dominant; consider horizontal scaling.
> ² Functional but queue wait P95 > 10s; evaluate SLO requirements.
> ³ MLX pool=1 handles 30ch with 21s total P95 and 0% errors, but queue wait is significant.

### Memory Budget

| Config | Per-Instance Memory | pool=1 | pool=2 | pool=4 |
|--------|-------------------|--------|--------|--------|
| faster_whisper small (CPU, int8) | ~500 MB | ~500 MB | ~1 GB | ~2 GB |
| torch_whisper small (MPS, fp32) | ~1 GB | ~1 GB | ~2 GB | ~4 GB |
| mlx_whisper small (MPS, fp16) | ~600 MB | ~600 MB | N/A | N/A |
| faster_whisper small (CUDA, fp16) | ~500 MB VRAM | ~500 MB | ~1 GB | ~2 GB |

Add OS overhead (~2–4 GB) and VAD model pool (~50 MB per model × pool_size) to the totals.

### Production Recommendations

**Small deployment (1–5 concurrent channels)**

- pool_size: 1
- Recommended: any backend; GPU backends will be mostly idle (buffer_wait dominant)
- Memory: 4–8 GB total

**Medium deployment (5–15 concurrent channels)**

- pool_size: 1 (GPU) or 2–4 (CPU)
- Recommended: GPU backend for sub-second inference and lower queue pressure
- Memory: 8–16 GB total (or 2+ GB VRAM for CUDA)

**Large deployment (15–30+ concurrent channels)**

- pool_size: 1 per instance + horizontal scaling (multiple server instances)
- Recommended: CUDA (fastest at scale) or MLX (single-node champion on Apple Silicon)
- Memory: 16+ GB total per instance (or 4+ GB VRAM for CUDA)
- Consider Kubernetes HPA with `stt_decode_pending` metric for auto-scaling

### Tuning Tips

1. **Watch `queue_wait`**: if Queue Wait P95 > Inference P50, add more instances (not more workers).
2. **GPU pool=1 always**: multi-pool GPU causes 3–5x inference degradation; scale horizontally instead.
3. **Partials are expendable**: the server automatically drops partials under load. Only finals block.
4. **Shorter utterances help**: lower `vad.silence` (e.g. 0.3s) produces shorter segments with faster queue rotation.
5. **Smaller models trade accuracy for throughput**: `tiny` or `base` models have 5–10x lower RTF than `small`.
6. **RMS gating reduces noise decodes**: set `safety.speech_rms_threshold` to 0.005–0.02 to skip silent segments.
7. **Server warns on mismatch**: when `max_sessions > pool_size × 3`, the server logs a sizing warning at startup.

---

## Bottleneck Transition Dynamics

All configurations exhibit a predictable two-phase behavior as concurrent channels increase:

**Phase 1 — Buffer-Wait Dominant (healthy):**
At low concurrency (channels ≤ pool capacity), 80–92% of total latency is spent waiting
for audio data to accumulate in the buffer before VAD triggers a decode. The hardware
is underutilized and inference completes near-instantly. This is the ideal operating state.

**Phase 2 — Queue-Wait Dominant (saturated):**
When channel count exceeds the pool's processing capacity, the bottleneck shifts to
queue wait. Workers are 100% occupied, and incoming decode requests queue up behind
active tasks. Queue wait grows linearly (or worse) with channel count and eventually
dominates 60–85% of total latency.

**Tipping point indicators:**

- Queue Wait P50 exceeds Inference P50
- Bottleneck label shifts from `buffer_wait` to `queue_wait`
- Queue Wait P95 / Total P95 ratio exceeds 50%

When the system reaches Phase 2, adding more pool workers (on GPU) worsens per-request
latency due to contention. The correct response is horizontal scaling: deploy additional
server instances behind a load balancer and use `stt_decode_pending` as the HPA trigger metric.

---

## Reproducing These Results

```bash
# macOS CPU
./tools/bench/run_benchmark_matrix.sh macos-cpu

# macOS GPU (torch_whisper, pool=1)
./tools/bench/run_benchmark_matrix.sh macos-gpu

# macOS GPU (mlx_whisper, pool=1)
./tools/bench/run_benchmark_matrix.sh macos-gpu-mlx

# Ubuntu GPU (faster_whisper CUDA, pool=1)
./tools/bench/run_benchmark_matrix.sh ubuntu-gpu

# Ubuntu CPU
./tools/bench/run_benchmark_matrix.sh ubuntu-cpu
```

Results are saved to `bench_results/<profile>/<timestamp>/`.
See `tools/bench/run_benchmark_matrix.sh` for all options and environment variables.

For the full test methodology and load test tool documentation, see
[`docs/development.md`](development.md) and [`docs/slo.md`](slo.md).
