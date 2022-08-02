import os

import numpy as np
import pytest

import tensorflow as tf
from src.tensorflow.loss.voc.loss import compute_giou

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")


@pytest.mark.parametrize("mode", [("iou"), ("giou")])
def test_compute_giou(mode: str):
    a = np.array([[1, 1, 2, 2], [2, 2, 3, 3]])
    b = tf.convert_to_tensor(a, tf.float32)
    res = compute_giou(b, b, mode=mode)
    assert res.shape == (2,)
