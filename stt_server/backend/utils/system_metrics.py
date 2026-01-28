"""System and process metrics helpers."""

from __future__ import annotations

import os
import platform
import resource
import time
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None
try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None


def _get_load_avg() -> Optional[List[float]]:
    try:
        return list(os.getloadavg())
    except (AttributeError, OSError):
        return None


def _normalize_rss(value: int) -> int:
    if platform.system() == "Darwin":
        return int(value)
    return int(value) * 1024


def _collect_psutil_metrics() -> Dict[str, Any]:
    if psutil is None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "process": {
                "pid": os.getpid(),
                "cpu_percent": None,
                "rss_bytes": _normalize_rss(usage.ru_maxrss),
                "vms_bytes": None,
                "threads": None,
            },
            "system": {
                "memory_total_bytes": None,
                "memory_available_bytes": None,
                "memory_percent": None,
            },
        }
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.0)
    system_mem = psutil.virtual_memory()
    return {
        "process": {
            "pid": process.pid,
            "cpu_percent": cpu_percent,
            "rss_bytes": mem_info.rss,
            "vms_bytes": mem_info.vms,
            "threads": process.num_threads(),
        },
        "system": {
            "memory_total_bytes": system_mem.total,
            "memory_available_bytes": system_mem.available,
            "memory_percent": system_mem.percent,
        },
    }


def _collect_nvidia_metrics() -> List[Dict[str, Any]]:
    if pynvml is None:
        return []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus: List[Dict[str, Any]] = []
        for index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            gpus.append(
                {
                    "index": index,
                    "name": name,
                    "utilization_gpu": int(utilization.gpu),
                    "utilization_memory": int(utilization.memory),
                    "memory_total_mib": int(memory.total / (1024 * 1024)),
                    "memory_used_mib": int(memory.used / (1024 * 1024)),
                    "temperature_gpu": int(temperature),
                }
            )
        return gpus
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gpu_metrics_enabled() -> bool:
    value = os.getenv("STT_ENABLE_GPU_METRICS", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def collect_system_metrics() -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "load_avg": _get_load_avg(),
    }
    metrics.update(_collect_psutil_metrics())
    metrics["gpus"] = _collect_nvidia_metrics() if _gpu_metrics_enabled() else []
    return metrics
