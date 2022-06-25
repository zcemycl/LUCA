import argparse
import os
from typing import Tuple

import pytest

import tensorflow as tf
from src.tensorflow.model.mobilenetv3_yolov3.model import (
    MobileNetV3_YoloV3,
    main,
    parse_args,
)


@pytest.fixture
def inp_net() -> Tuple[tf.Tensor, tf.keras.Sequential]:
    inp = tf.keras.layers.Input([416, 416, 3])
    args = argparse.Namespace()
    net = MobileNetV3_YoloV3(args)
    return inp, net


def test_backbone(inp_net: Tuple[tf.Tensor, tf.keras.Sequential]):
    inp, net = inp_net
    backbone = net.Backbone(inp)
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


def test_parse_args():
    args = parse_args(["--debug_model"])
    assert args.debug_model is True


def test_main():
    args = parse_args([])
    main(args)
