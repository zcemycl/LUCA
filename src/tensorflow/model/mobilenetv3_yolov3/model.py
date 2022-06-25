import argparse
import sys
from typing import List

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

    def Network(self) -> tf.keras.Model:
        inp = tf.keras.layers.Input([416, 416, 3])
        backbone = self.Backbone(inp)
        for layer in backbone.layers:
            layer.trainable = False
        y = backbone.output

        return tf.keras.Model(inputs=inp, outputs=y)


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument(
        "--anchors",
        type=str,
        default="""\
10,13,16,30,33,23,\
30,61,62,45,59,119,\
116,90,156,198,373,326""",
    )
    p.add_argument("--debug_model", action="store_true")
    return p.parse_args(args)


def main(args: argparse.Namespace):
    net = MobileNetV3_YoloV3(args)
    model = net.Network()
    x = tf.random.normal([1, 416, 416, 3])
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
