import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

import src.tensorflow.model.mobilenetv3_yolov3._paths as _paths
import tensorflow as tf
from src.tensorflow.model.interface.network_bbox import Network_Bbox
from src.tensorflow.utils.decorator import ramit_timeit
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    DepthwiseConv2D,
    Input,
    Lambda,
    ReLU,
    Reshape,
    UpSampling2D,
)

print(_paths)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class MobileNetV3_YoloV3(Network_Bbox):
    config: argparse.Namespace
    """
    Backbone -> 2 Feature Maps -------------------------
    |                                                  |
    PyramidLayer                                       |
    |          |                                       |
    GlueLayer  PyramidLeak (1st Detection Layer)       |
    |                                                  |
    Concatenate <-Skip from Backbone -------------------
    |                                                  |
    PyramidLayer                                       |
    |          |                                       |
    GlueLayer  PyramidLeak (2nd Detection Layer)       |
    |                                                  |
    Concatenate <-Skip from Backbone -------------------
    |
    PyramidLayer
               |
               PyramidLeak (3rd Detection Layer)
    """

    def __init__(self, config: argparse.Namespace):
        super().__init__(config)
        self.num_layers = 3
        self.dim = 416
        self.input_shape = tf.TensorShape([self.dim, self.dim])
        self.anchors = [float(x) for x in self.config.anchors.split(",")]
        self.anchors = np.array(self.anchors, np.float32).reshape(-1, 2)
        self.num_anchors = len(self.anchors)
        self.anchor_layer_ind = [
            int(x) for x in self.config.anchor_mask.split(",")
        ]
        self.anchor_mask = np.array(self.anchor_layer_ind)
        self.anchor_mask = [
            np.where(self.anchor_mask == i)[0].tolist()
            for i in range(self.num_layers)
        ]
        self.num_anchors_layers = [len(i) for i in self.anchor_mask]

    @ramit_timeit
    def Backbone(
        self,
        x: tf.Tensor,
        alpha: float = 1.0,
        include_top: bool = False,
        weights: str = "imagenet",
    ) -> tf.keras.Sequential:
        """
        Pretrained Backbone Network
        - Replaceable with other imagenet network.
        - Requires multiple scale feature maps (416->52->26->13).
        - Different input shapes for different backbones.
        """
        return tf.keras.applications.MobileNetV3Large(
            input_tensor=x,
            alpha=alpha,
            include_top=include_top,
            weights=weights,
        )

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

    @ramit_timeit
    def EmptyLayer(self, shape: List[int]) -> tf.keras.Sequential:
        """
        Empty Layer is for detection layer with 0 anchors
        """
        return tf.keras.Sequential(
            [Lambda(lambda x: tf.zeros([tf.shape(x)[0], *shape]))]
        )

    @ramit_timeit
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

    @ramit_timeit
    def PyramidLeak(
        self, channels: int, num_anchors: int, dim: int
    ) -> tf.keras.Sequential:
        """
        Detection Layer with output shapes
        - (13,13,3,25)
        - (26,26,3,25)
        - (52,52,3,25)
        - 3: number of anchors for that layer
        - 25: xywh+objectness+20 class scores
        """
        out_filters = num_anchors * (self.config.num_classes + 5)
        res = tf.keras.Sequential()
        res.add(
            self.SeparableConv2D(
                2 * channels, (3, 3), padding="same", use_bias=False
            )
        )
        res.add(
            Conv2D(out_filters, kernel_size=1, padding="same", use_bias=False)
        )
        res.add(Reshape((dim, dim, num_anchors, self.config.num_classes + 5)))
        return res

    @ramit_timeit
    def GlueLayer(self, id: int, channels: int) -> tf.keras.Sequential:
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
                UpSampling2D(2),
            ]
        )
        return res

    def _make_divisible(
        self, v: int, divisor: int, min_value: Optional[int] = None
    ) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    @ramit_timeit
    def SkipConv2D(
        self, kernel: int, alpha: int, channels: int
    ) -> tf.keras.Sequential:
        last_block_filters = self._make_divisible(channels * alpha, 8)
        res = tf.keras.Sequential(
            [
                Conv2D(
                    last_block_filters, kernel, padding="same", use_bias=False
                ),
                BatchNormalization(),
                ReLU(6.0),
            ]
        )
        return res

    def Network(self) -> tf.keras.Model:
        """
        Functional Yolo Model
        - MobileNetV3 (416,416,3)->(13/26/52,13/26/52,3,25)
        """
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
        x = backbone.output  # multiply_19
        top = self.PyramidLayer(15, 512)
        if self.num_anchors_layers[0] == 0:
            topLeak = self.EmptyLayer([13, 13, 0, 5 + self.config.num_classes])
        else:
            topLeak = self.PyramidLeak(512, self.num_anchors_layers[0], 13)
        glueTopMid = self.GlueLayer(18, 256)
        mid = self.PyramidLayer(19, 256)
        if self.num_anchors_layers[1] == 0:
            midLeak = self.EmptyLayer([26, 26, 0, 5 + self.config.num_classes])
        else:
            midLeak = self.PyramidLeak(256, self.num_anchors_layers[1], 26)
        skipConvMid = self.SkipConv2D((1, 1), self.config.alpha, 384)
        glueMidBot = self.GlueLayer(22, 128)
        bot = self.PyramidLayer(23, 128)
        if self.num_anchors_layers[2] == 0:
            botLeak = self.EmptyLayer([52, 52, 0, 5 + self.config.num_classes])
        else:
            botLeak = self.PyramidLeak(128, self.num_anchors_layers[2], 52)
        skipConvBot = self.SkipConv2D((1, 1), self.config.alpha, 128)

        x1 = top(x)
        y1 = topLeak(x1)
        x2 = glueTopMid(x1)
        x2 = Concatenate()([x2, skipConvMid(skipMid)])
        x2 = mid(x2)
        y2 = midLeak(x2)
        x3 = glueMidBot(x2)
        x3 = Concatenate()([x3, skipConvBot(skipBot)])
        x3 = bot(x3)
        y3 = botLeak(x3)

        return tf.keras.Model(inputs=inp, outputs=[y1, y2, y3])

    def head(
        self, layer_id: int, feats: tf.Tensor, calc_loss: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Head of detection layer
        - Returns uncorrected xywh, objectness, class confidences
        - x'(x-center) = x+sigmoid(px)
        - w'(box width) = exp(pw)*anchor_size_w
        - confidence/objectness = sigmoid(?)
        - With calc_loss == True
        - - grid is metadata combining all detections in one variable
        """
        num_anchors = self.num_anchors_layers[layer_id]
        if not calc_loss and num_anchors == 0:
            return (
                feats[..., :2],
                feats[..., 2:4],
                feats[..., 4:5],
                feats[..., 5:],
            )
        anchors_tensor = tf.reshape(
            tf.constant(self.anchors[self.anchor_mask[layer_id]]),
            [1, 1, 1, num_anchors, 2],
        )
        grid_shape = tf.shape(feats)[1:3]
        grid_y = tf.tile(
            tf.reshape(tf.range(0, grid_shape[0]), [-1, 1, 1, 1]),
            [1, grid_shape[1], 1, 1],
        )
        grid_x = tf.tile(
            tf.reshape(tf.range(0, grid_shape[1]), [1, -1, 1, 1]),
            [grid_shape[0], 1, 1, 1],
        )
        grid = tf.concat([grid_x, grid_y], -1)
        grid = tf.cast(grid, feats.dtype)
        if calc_loss and num_anchors == 0:
            return grid, feats[..., :2], feats[..., 2:4], feats[..., 4:5]
        box_xy = (tf.sigmoid(feats[..., :2]) + grid) / tf.cast(
            grid_shape[..., ::-1], feats.dtype
        )

        box_wh = (
            tf.exp(feats[..., 2:4])
            * tf.cast(anchors_tensor, feats.dtype)
            / tf.cast(self.input_shape, feats.dtype)
        )

        box_confidence = tf.sigmoid(feats[..., 4:5])
        if calc_loss:
            return grid, box_xy, box_wh, box_confidence
        box_class_probs = tf.sigmoid(feats[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs

    def correct_boxes(
        self,
        box_xy: tf.Tensor,
        box_wh: tf.Tensor,
        input_shape: tf.Tensor,
        image_shape: List[int],
    ) -> tf.Tensor:
        """
        Get corrected boxes
        - Output shape: [batch, 13, 13, num_anchors, 4]
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, box_yx.dtype)
        image_shape = tf.cast(image_shape, box_yx.dtype)
        max_shape = tf.maximum(image_shape[0], image_shape[1])
        ratio = image_shape / max_shape
        boxed_shape = input_shape * ratio
        offset = (input_shape - boxed_shape) / 2.0
        scale = image_shape / boxed_shape
        box_yx = (box_yx * input_shape - offset) * scale
        box_hw *= input_shape * scale
        box_mins = box_yx - (box_hw / 2.0)
        box_maxes = box_yx + (box_hw / 2.0)
        # ymin,xmin,ymax,xmax
        boxes = tf.concat(
            [
                tf.clip_by_value(box_mins[..., 0:1], 0, image_shape[0]),
                tf.clip_by_value(box_mins[..., 1:2], 0, image_shape[1]),
                tf.clip_by_value(box_maxes[..., 0:1], 0, image_shape[0]),
                tf.clip_by_value(box_maxes[..., 1:2], 0, image_shape[1]),
            ],
            axis=-1,
        )
        return boxes

    def boxes_and_scores(
        self, layer_id: int, feats: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get Output from detection layer
        """
        box_xy, box_wh, box_confidence, box_class_probs = self.head(
            layer_id, feats
        )
        boxes = self.correct_boxes(
            box_xy, box_wh, self.input_shape, [self.dim, self.dim]
        )
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, self.config.num_classes])
        return boxes, box_scores

    def group_boxes_and_scores(
        self,
        yolo_outputs: List[tf.Tensor],
        # image_shape: List[int]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        num_layers = len(yolo_outputs)
        boxes, box_scores = [], []
        for i in range(num_layers):
            _boxes, _box_scores = self.boxes_and_scores(
                i,
                yolo_outputs[i],
            )
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis=0)
        box_scores = tf.concat(box_scores, axis=0)
        return boxes, box_scores

    def filter_boxes(
        self,
        boxes: tf.Tensor,
        box_scores: tf.Tensor,
        iou_threshold: float,
        score_threshold: float,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        filter boxes with non-maximum suppression
        """
        boxes_, scores_, classes_ = [], [], []
        max_boxes_tensor = tf.constant(self.config.max_boxes, dtype=tf.int32)
        for c in range(self.config.num_classes):
            nms_index = tf.image.non_max_suppression(
                boxes,
                box_scores[:, c],
                max_boxes_tensor,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )
            class_boxes = tf.gather(boxes, nms_index)
            class_box_scores = tf.gather(box_scores[:, c], nms_index)
            classes = tf.ones_like(class_box_scores, tf.int32) * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis=0)
        scores_ = tf.concat(scores_, axis=0, name="scores")
        classes_ = tf.concat(classes_, axis=0, name="classes")
        boxes_ = tf.cast(boxes_, tf.int32, name="boxes")
        return boxes_, scores_, classes_


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--num_classes",
        type=int,
        default=20,
        help="Number of object classes",
    )
    p.add_argument(
        "--anchor_mask",
        type=str,
        default="2,2,2,1,1,1,1,0,0",
        help="Allocations of anchors for each detection layer",
    )
    p.add_argument(
        "--anchors",
        type=str,
        default="""\
10,13,16,30,33,23,\
30,61,62,45,59,119,\
116,90,156,198,373,326""",
    )
    p.add_argument("--debug_model", action="store_true")
    p.add_argument("--alpha", type=int, default=1)
    p.add_argument(
        "--max_boxes",
        type=int,
        default=20,
        help="Maximum number of boxes for each class",
    )
    p.add_argument("--score_threshold", type=float, default=0.6)
    p.add_argument("--iou_threshold", type=float, default=0.5)
    return p.parse_args(args)


def main(args: argparse.Namespace):
    net = MobileNetV3_YoloV3(args)
    model = net.Network()
    x = tf.random.normal([1, 416, 416, 3])
    y = model(x)
    return y


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
