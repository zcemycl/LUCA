import argparse

import src.tensorflow.model.mobilenetv3_yolov3._paths as _paths
import tensorflow as tf
from src.tensorflow.model.interface.network_bbox import Network_Bbox
from src.tensorflow.utils.decorator import ramit_timeit

print(_paths)


class MobileNetV3_YoloV3(Network_Bbox):
    config: argparse.Namespace

    def __init__(self, config: argparse.Namespace):
        super().__init__(config)

    @ramit_timeit
    def Backbone(
        self,
        x: tf.Tensor,
        alpha: float = 1.0,
        include_top: bool = False,
        weights: str = "imagenet",
    ) -> tf.keras.Sequential:
        return tf.keras.applications.MobileNetV3Large(
            input_tensor=x,
            alpha=alpha,
            include_top=include_top,
            weights=weights,
        )

    @ramit_timeit
    def Network(self) -> tf.keras.Model:
        inp = tf.keras.layers.Input([416, 416, 3])
        backbone = self.Backbone(inp)
        for layer in backbone.layers:
            layer.trainable = False
        y = backbone.output

        return tf.keras.Model(inputs=inp, outputs=y)


if __name__ == "__main__":
    pass
