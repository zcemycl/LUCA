import argparse
import sys
from typing import List

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
    end: int = None,
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
    box_xy, box_wh, box_conf, box_class = netObj.head(
        args.layer_id, y[args.layer_id]
    )
    boxes = netObj.correct_boxes(
        box_xy, box_wh, netObj.input_shape, [416, 416]
    )
    print(boxes.shape)
    x = tf.zeros([1, 416, 416, 3])
    plotAnchors(
        x, boxes[:, :, :, args.relative_anchor_id, :], start=0, skip=args.skip
    )


def patch_parse_args(args: List[str]) -> argparse.Namespace:
    ns = parse_args(args)
    p = argparse.ArgumentParser()
    p.add_argument("--layer_id", type=int, default=2)
    p.add_argument("--relative_anchor_id", type=int, default=0)
    p.add_argument("--skip", type=int, default=20)
    args_, _ = p.parse_known_args(args)
    return argparse.Namespace(**vars(ns), **vars(args_))


if __name__ == "__main__":
    args = patch_parse_args(sys.argv[1:])
    visualizeAnchors(args)
