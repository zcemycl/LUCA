import argparse
import time
from typing import Any, Callable

import psutil


def ramit_timeit(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        sram = psutil.virtual_memory().used
        res = func(*args, **kwargs)
        end = time.time()
        eram = psutil.virtual_memory().used

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
{(eram-sram)/2**20} Mb
                        """
                    )

        return res

    return wrapper
