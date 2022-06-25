import argparse
import os

import tensorflow as tf
from src.tensorflow.model.mobilenetv3_yolov3.model import MobileNetV3_YoloV3


def test_backbone():
    inp = tf.keras.layers.Input([416, 416, 3])
    args = argparse.Namespace(a=1, b=2)
    net = MobileNetV3_YoloV3(args)
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
