import argparse
import io
import os
import sys
from typing import Tuple

import pytest

import tensorflow as tf
from src.tensorflow.model.mobilenetv3_yolov3.model import MobileNetV3_YoloV3


@pytest.fixture
def inp_net() -> Tuple[tf.Tensor, tf.keras.Sequential]:
    inp = tf.keras.layers.Input([416, 416, 3])
    args = argparse.Namespace(debug_model=True)
    net = MobileNetV3_YoloV3(args)
    return inp, net


def test_backbone(inp_net: Tuple[tf.Tensor, tf.keras.Sequential]):
    inp, net = inp_net
    capOut = io.StringIO()
    sys.stdout = capOut
    backbone = net.Backbone(inp)
    sys.stdout = sys.__stdout__
    print(capOut.getvalue())
    assert "[DEBUG] Func" in capOut.getvalue()
    layers = backbone.layers
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(TEST_DIR + "/output_shape.txt", "r") as f:
        lines = f.readlines()
    assert len(layers) == len(lines)
    for layer, line in zip(layers, lines):
        nameDim = line.split(",")
        nameDim[-1] = nameDim[-1].replace("\n", "")
        dim = list(map(int, nameDim[1:]))
        if "input" in layer.name:
            continue
        assert layer.name == nameDim[0].replace(" ", "")
        assert list(layer.output_shape)[1:] == dim


def test_network(inp_net: Tuple[tf.Tensor, tf.keras.Sequential]):
    _, net = inp_net
    model = net.Network()
    x = tf.random.normal([1, 416, 416, 3])
    y = model(x)
    assert y.shape == (1, 13, 13, 960)
