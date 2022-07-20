import argparse
import sys

import matplotlib.patches as ptc
import matplotlib.pyplot as plt

import tensorflow as tf
from src.tensorflow.model.mobilenetv3_yolov3.model import (
    MobileNetV3_YoloV3,
    parse_args,
)


def plotAnchors(
    img: tf.Tensor,
    boxes: tf.Tensor,
    start: int = 0,
    end: int = 1,
    skip: int = 300,
):
    box1 = tf.reshape(boxes, [-1, 4])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img[0])
    for y1, x1, y2, x2 in box1[start:end:skip]:
        rect = ptc.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
        _ = ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()


def visualizeAnchors(args: argparse.Namespace):
    netObj = MobileNetV3_YoloV3(args)
    print(args)
    model = netObj.Network()
    x = tf.random.normal([1, 416, 416, 3])
    y = model(x)
    layer_id = 2
    box_xy, box_wh, box_conf, box_class = netObj.head(layer_id, y[layer_id])
    boxes = netObj.correct_boxes(
        box_xy, box_wh, netObj.input_shape, [416, 416]
    )
    print(boxes.shape)
    x = tf.zeros([1, 416, 416, 3])
    plotAnchors(x, boxes[:, :, :, 0, :], start=0, end=2704, skip=20)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    visualizeAnchors(args)
