import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

import src.tensorflow.model.mobilenetv3_yolov3._paths as _paths
import tensorflow as tf
from src.tensorflow.model.interface.network_bbox import Network_Bbox
from src.tensorflow.utils.decorator import ramit_timeit
from tensorflow.keras.layers import (  # Concatenate,; Reshape,; UpSampling2D,
    BatchNormalization,
    Conv2D,
    DepthwiseConv2D,
    Input,
    Lambda,
    ReLU,
)

print(_paths)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MobileNetV3_YoloV3(Network_Bbox):
    config: argparse.Namespace

    def __init__(self, config: argparse.Namespace):
        super().__init__(config)
        self.num_layers = 3
        self.anchors = [float(x) for x in self.config.anchors.split(",")]
        self.anchors = np.array(self.anchors, np.float32).reshape(-1, 2)
        self.num_anchors = len(self.anchors)
        self.anchor_mask = [int(x) for x in self.config.anchor_mask.split(",")]
        self.anchor_mask = np.array(self.anchor_mask)
        self.anchor_mask = [
            np.where(self.anchor_mask == i)[0].tolist()
            for i in range(self.num_layers)
        ]

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
    def SeparableConv2D(
        self,
        channels: int,
        kernel: Tuple[int, int],
        strides: Tuple[int, int] = (1, 1),
        padding: str = "valid",
        use_bias: bool = True,
    ) -> tf.keras.Sequential:
        res = tf.keras.Sequential(
            [
                DepthwiseConv2D(
                    kernel, padding=padding, use_bias=use_bias, strides=strides
                ),
                BatchNormalization(),
                ReLU(6.0),
                Conv2D(
                    channels, 1, padding="same", use_bias=use_bias, strides=1
                ),
                BatchNormalization(),
                ReLU(6.0),
            ]
        )
        return res

    def EmptyLayer(self, shape: List[int]) -> tf.keras.Sequential:
        return tf.keras.Sequential(
            [Lambda(lambda x: tf.zeros([tf.shape(x)[0], *shape]))]
        )

    def PyramidLayer(self, id: int, channels: int) -> tf.keras.Sequential:
        res = tf.keras.Sequential(
            [
                Conv2D(
                    channels,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    name=f"block_{id}_conv",
                ),
                BatchNormalization(momentum=0.9, name=f"block_{id}_BN"),
                ReLU(6.0, name=f"block_{id}_relu6"),
                self.SeparableConv2D(
                    2 * channels, (3, 3), padding="same", use_bias=False
                ),
                Conv2D(
                    channels,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    name=f"block_{id+1}_conv",
                ),
                BatchNormalization(momentum=0.9, name=f"block_{id+1}_BN"),
                ReLU(6.0, name=f"block_{id+1}_relu6"),
                self.SeparableConv2D(
                    2 * channels, (3, 3), padding="same", use_bias=False
                ),
                Conv2D(
                    channels,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    name=f"block_{id+2}_conv",
                ),
                BatchNormalization(momentum=0.9, name=f"block_{id+2}_BN"),
                ReLU(6.0, name=f"block_{id+2}_relu6"),
            ]
        )
        return res

    def Network(self) -> tf.keras.Model:
        inp = Input([416, 416, 3])
        backbone = self.Backbone(inp)
        for layer in backbone.layers:
            layer.trainable = False
        # choose from multiply_1 / expanded_conv_5/project/BatchNorm
        skipBot = backbone.get_layer(
            "expanded_conv_5/project/BatchNorm"
        ).output
        # choose from multiply_13 / expanded_conv_11/project/BatchNorm
        skipMid = backbone.get_layer(
            "expanded_conv_11/project/BatchNorm"
        ).output
        y = backbone.output  # multiply_19
        nullLay = self.EmptyLayer([13, 13, 0, 25])
        y1 = nullLay(y)

        return tf.keras.Model(inputs=inp, outputs=[skipBot, skipMid, y1, y])


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--num_classes", type=int, default=20)
    p.add_argument("--anchor_mask", type=str, default="2,2,2,1,1,1,0,0,0")
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
    y1 = net.PyramidLayer(15, 512)(y[-1])
    print(y[-1].shape)
    print(y1.shape)
    print(net.anchors, net.num_anchors)
    print(net.anchor_mask)
    # pdb.set_trace()


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
