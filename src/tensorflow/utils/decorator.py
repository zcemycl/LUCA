import argparse
import time
from typing import Any, Callable

import nvidia_smi
import psutil


def ramit_timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        nvidia_smi.nvmlInit()
        d_Count = nvidia_smi.nvmlDeviceGetCount()
        sram = psutil.virtual_memory().used
        svram = 0
        for i in range(d_Count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            svram += info.used
        res = func(*args, **kwargs)
        end = time.time()
        eram = psutil.virtual_memory().used
        evram = 0
        for i in range(d_Count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            evram += info.used

        # has config in self
        if hasattr(args[0], "config"):
            # config is Namespace
            if isinstance(args[0].config, argparse.Namespace):
                # check if config has debug model
                if (
                    hasattr(args[0].config, "debug_model")
                    and args[0].config.debug_model
                ):
                    print(
                        f"""\
[DEBUG] Func {func.__name__} takes: {end-start:.2f} s,\
{(eram-sram)/2**20:.2f} Mb RAM,\
{(evram-svram)/2**20:.2f} Mb VRAM"""
                    )
        nvidia_smi.nvmlShutdown()
        return res

    return wrapper
