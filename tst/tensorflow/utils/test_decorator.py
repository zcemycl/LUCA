import io
import sys
from argparse import Namespace

import pytest
from pytest_mock import MockFixture

from src.tensorflow.utils.decorator import ramit_timeit


class fakeClass:
    def __init__(self, config: Namespace):
        self.config = config

    @ramit_timeit
    def fakeMethod(self):
        pass


class fakeNvid:
    @property
    def used(self):
        return 0


@pytest.fixture
def return_class() -> fakeClass:
    args = Namespace(debug_model=True)
    fake = fakeClass(args)
    return fake


def test_ramtime_class_cpu(return_class: fakeClass, mocker: MockFixture):
    mocker.patch(
        "src.tensorflow.utils.decorator.tf.test.is_gpu_available",
        return_value=False,
    )
    capOut = io.StringIO()
    sys.stdout = capOut
    return_class.fakeMethod()
    sys.stdout = sys.__stdout__
    assert "[DEBUG] Func" in capOut.getvalue()


def test_ramtime_class_gpu(return_class: fakeClass, mocker: MockFixture):
    nvd = fakeNvid()
    mocker.patch(
        "src.tensorflow.utils.decorator.tf.test.is_gpu_available",
        return_value=True,
    )
    mocker.patch(
        "src.tensorflow.utils.decorator.nvidia_smi.nvmlInit",
        return_value=None,
    )
    mocker.patch(
        "src.tensorflow.utils.decorator.nvidia_smi.nvmlShutdown",
        return_value=None,
    )
    mocker.patch(
        "src.tensorflow.utils.decorator.nvidia_smi.nvmlDeviceGetCount",
        return_value=1,
    )
    mocker.patch(
        "src.tensorflow.utils.decorator.nvidia_smi.nvmlDeviceGetHandleByIndex",
        return_value=None,
    )
    mocker.patch(
        "src.tensorflow.utils.decorator.nvidia_smi.nvmlDeviceGetMemoryInfo",
        return_value=nvd,
    )
    capOut = io.StringIO()
    sys.stdout = capOut
    return_class.fakeMethod()
    sys.stdout = sys.__stdout__
    assert "[DEBUG] Func" in capOut.getvalue()
