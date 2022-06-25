import argparse

import pytest

import tensorflow as tf
from src.tensorflow.model.interface.network_bbox import Network_Bbox


def test_backbone_NotImplementedError():
    x = tf.keras.layers.Input([416, 416, 3])
    args = argparse.Namespace(a=1, b=2)
    interfaceNet = Network_Bbox(args)
    with pytest.raises(NotImplementedError) as _:
        interfaceNet.Backbone(x)


def test_network_NotImplementedError():
    args = argparse.Namespace(a=1, b=2)
    interfaceNet = Network_Bbox(args)
    with pytest.raises(NotImplementedError) as _:
        interfaceNet.Network(args)
