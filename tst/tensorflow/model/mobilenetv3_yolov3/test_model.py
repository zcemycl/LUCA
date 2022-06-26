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
def inp_net_model() -> Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]:
    inp = tf.keras.layers.Input([416, 416, 3])
    args = parse_args([])
    net = MobileNetV3_YoloV3(args)
    model = net.Network()
    return inp, net, model


@pytest.fixture
def inp_empty_net_model() -> Tuple[
    tf.Tensor, tf.keras.Sequential, tf.keras.Model
]:
    inp = tf.keras.layers.Input([416, 416, 3])
    args = parse_args(["--anchor_mask", ",".join(["3"] * 9)])
    net = MobileNetV3_YoloV3(args)
    model = net.Network()
    return inp, net, model


def test_backbone(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    inp, net, _ = inp_net_model
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
        # assert layer.name == nameDim[0].replace(" ", "")
        assert list(layer.output_shape)[1:] == dim


def test_network(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, model = inp_net_model
    batch = 2
    x = tf.zeros([batch, 416, 416, 3])
    y = model(x)
    assert y[0].shape == (batch, 13, 13, net.num_anchors_layers[0], 25)
    assert y[1].shape == (batch, 26, 26, net.num_anchors_layers[1], 25)
    assert y[2].shape == (batch, 52, 52, net.num_anchors_layers[2], 25)


def test_empty_network(
    inp_empty_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, model = inp_empty_net_model
    batch = 2
    x = tf.zeros([batch, 416, 416, 3])
    y = model(x)
    assert y[0].shape == (batch, 13, 13, net.num_anchors_layers[0], 25)
    assert y[1].shape == (batch, 26, 26, net.num_anchors_layers[1], 25)
    assert y[2].shape == (batch, 52, 52, net.num_anchors_layers[2], 25)


def test_separableConv2D(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, _ = inp_net_model
    x = tf.zeros([1, 13, 13, 960])
    channels = 2
    layer = net.SeparableConv2D(
        channels, (3, 3), padding="same", use_bias=False
    )
    y = layer(x)
    assert y.shape == (1, 13, 13, channels)


def test_pyramidLayer(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, _ = inp_net_model
    batch, channels = 2, 4
    x = tf.zeros([batch, 13, 13, 960])
    layer = net.PyramidLayer(15, channels)
    y = layer(x)
    assert y.shape == (batch, 13, 13, channels)


def test_emptyLayer(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, _ = inp_net_model
    batch, channels = 2, 4
    shape = [13, 13, 0, 25]
    x = tf.zeros([batch, 13, 13, channels])
    layer = net.EmptyLayer([*shape])
    y = layer(x)
    assert y.shape == (batch, *shape)


def test_glueLayer(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, _ = inp_net_model
    batch = 2
    x = tf.zeros([batch, 13, 13, 512])
    layer = net.GlueLayer(18, 256)
    y = layer(x)
    assert y.shape == (batch, 26, 26, 256)


def test_parse_args():
    args = parse_args(["--debug_model"])
    assert args.debug_model is True


def test_main():
    args = parse_args([])
    y = main(args)
    assert len(y) == 3
    batch = 1
    assert y[0].shape == (batch, 13, 13, 3, 25)
    assert y[1].shape == (batch, 26, 26, 3, 25)
    assert y[2].shape == (batch, 52, 52, 3, 25)


@pytest.mark.parametrize("channels,expected", [(384, 384), (128, 128)])
def test_make_divisible(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model],
    channels: int,
    expected: int,
):
    _, net, _ = inp_net_model
    res = net._make_divisible(channels * 1, 8)
    assert res == expected


def test_skipConv2D(
    inp_net_model: Tuple[tf.Tensor, tf.keras.Sequential, tf.keras.Model]
):
    _, net, _ = inp_net_model
    batch = 2
    x = tf.zeros([batch, 52, 52, 40])
    layer = net.SkipConv2D((1, 1), 1, 128)
    y = layer(x)
    assert y.shape == (batch, 52, 52, 128)
