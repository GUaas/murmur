from __future__ import annotations

import ctypes
from ctypes import wintypes
import locale
import os
import platform
from pathlib import Path
from typing import Any

import torch


def process_memory_bytes() -> dict[str, int | None]:
    if os.name != "nt":
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
            return {"working_set": None, "peak_working_set": peak, "private_usage": None}
        except Exception:
            return {"working_set": None, "peak_working_set": None, "private_usage": None}

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return {"working_set": None, "peak_working_set": None, "private_usage": None}
    return {
        "working_set": int(counters.WorkingSetSize),
        "peak_working_set": int(counters.PeakWorkingSetSize),
        "private_usage": int(counters.PrivateUsage),
    }


def collect_system_info(project_root: Path) -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory": int(props.total_memory),
                }
            )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": os.path.realpath(os.sys.executable),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "locale": locale.getlocale(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_devices": cuda_devices,
        "project_root": str(project_root.resolve()),
    }
