import tensorflow as tf
from src.tensorflow.model.mobilenetv3_yolov3.model import MobileNetV3_YoloV3


def test_backbone():
    inp = tf.keras.layers.Input([416, 416, 3])
    backbone = MobileNetV3_YoloV3.Backbone(inp)
    assert backbone.layers[-1].name == "multiply_19"
