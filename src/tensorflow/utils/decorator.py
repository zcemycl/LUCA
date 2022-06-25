import argparse
import time
from typing import Any, Callable

import nvidia_smi
import psutil

import tensorflow as tf


def ramit_timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        gpu_available = tf.test.is_gpu_available()
        start = time.time()
        svram, evram = 0, 0
        if gpu_available:
            nvidia_smi.nvmlInit()
            d_Count = nvidia_smi.nvmlDeviceGetCount()
            for i in range(d_Count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                print("[INFO] ------", info.used)
                svram += info.used
        sram = psutil.virtual_memory().used

        res = func(*args, **kwargs)
        end = time.time()
        eram = psutil.virtual_memory().used
        if gpu_available:
            for i in range(d_Count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                evram += info.used
            nvidia_smi.nvmlShutdown()

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

        return res

    return wrapper
