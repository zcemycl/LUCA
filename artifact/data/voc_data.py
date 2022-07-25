import argparse
import pdb
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import matplotlib.image as mpimg
import matplotlib.patches as ptc
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from src.tensorflow.data.voc.data import (
    decode_fn,
    encode_fn,
    extractXml,
    parse_args,
    voc_dataloader,
)
from src.tensorflow.utils.fixmypy import mypy_xmlTree

find = mypy_xmlTree.find
getText = mypy_xmlTree.getText


def rectangle(
    xmin: Union[int, float],
    ymin: Union[int, float],
    xmax: Union[int, float],
    ymax: Union[int, float],
) -> ptc.Rectangle:
    rect = ptc.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        linewidth=1,
        edgecolor="g",
        facecolor="none",
    )
    return rect


def plotBboxAndImg(img: Union[np.ndarray, tf.Tensor], info: Dict[str, Any]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    print(img.shape)
    ax.imshow(img)
    for (xmin, ymin, xmax, ymax) in info["bbox"]:
        rect = rectangle(xmin, ymin, xmax, ymax)
        _ = ax.add_patch(rect)
    ax.set_axis_off()
    plt.show()


def visualizeBbox(args: argparse.Namespace):
    pathann = Path(args.vocroot) / Path("Annotations")
    pathjpg = Path(args.vocroot) / Path("JPEGImages")
    img = mpimg.imread(pathjpg / Path(args.vocid + ".jpg"))
    tree = ET.parse(pathann / Path(args.vocid + ".xml"))
    root = tree.getroot()
    xmlinfo = extractXml(root)
    print(xmlinfo)
    plotBboxAndImg(img, xmlinfo)


@tf.function
def load(
    record: Dict[str, Any],
    mode: str = "train",
    target_dims: tf.Tensor = tf.constant([416, 416], tf.float32),
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(record["path"])
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img, tf.float32) / 255.0
    xb = tf.cast(record["image/x"], tf.float32)
    yb = tf.cast(record["image/y"], tf.float32)
    wb = tf.cast(record["image/w"], tf.float32)
    hb = tf.cast(record["image/h"], tf.float32)
    labels = record["labels"]

    if mode == "train":
        tar_aspect_ratio = target_dims[0] / target_dims[1]
        dims = tf.shape(img)
        h = tf.cast(dims[0], tf.float32)
        w = tf.cast(dims[1], tf.float32)
        aspect_ratio = h / w
        print(h, w, aspect_ratio)

        aspect_cond = tf.less(aspect_ratio, tar_aspect_ratio)
        r = tf.cond(
            aspect_cond, lambda: target_dims[1] / w, lambda: target_dims[0] / h
        )
        tar_w = r * w
        tar_h = r * h
        img = tf.image.resize(img, [tar_h, tar_w])
        wb *= r
        hb *= r
        xb *= r
        yb *= r
        dw = tf.cast(target_dims[1] - tar_w, tf.int64)
        dh = tf.cast(target_dims[0] - tar_h, tf.int64)
        dx = tf.cond(
            aspect_cond,
            lambda: tf.constant(0, tf.int64),
            lambda: tf.random.uniform([], 0, dw, tf.int64),
        )
        dy = tf.cond(
            aspect_cond,
            lambda: tf.random.uniform([], 0, dh, tf.int64),
            lambda: tf.constant(0, tf.int64),
        )
        xb = tf.cond(
            aspect_cond, lambda: xb, lambda: xb + tf.cast(dx, tf.float32)
        )
        yb = tf.cond(
            aspect_cond, lambda: yb + tf.cast(dy, tf.float32), lambda: yb
        )
        # # pdb.set_trace()
        img = tf.pad(
            img, tf.convert_to_tensor([[dy, dh - dy], [dx, dw - dx], [0, 0]])
        )

        flipx = tf.random.uniform([], 0, 1)
        flipy = tf.random.uniform([], 0, 1)
        img, xb = tf.cond(
            tf.less(flipx, 0.5),
            lambda: (img, xb),
            lambda: (tf.image.flip_left_right(img), target_dims[1] - xb),
        )
        img, yb = tf.cond(
            tf.less(flipy, 0.5),
            lambda: (img, yb),
            lambda: (tf.image.flip_up_down(img), target_dims[0] - yb),
        )

        img = tf.image.random_hue(img, 0.5)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        img = tf.image.random_brightness(img, 0.1)
        val = tf.random.uniform([], 0.8, 2)
        image = tf.image.adjust_gamma(img, val)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.image.random_jpeg_quality(img, 80, 100)
        # add gaussian noise
        img += tf.cast(
            tf.random.uniform(shape=tf.shape(img), minval=0, maxval=0.05),
            tf.float32,
        )
    return img, xb, yb, wb, hb, labels


if __name__ == "__main__":
    # args = parse_args(sys.argv[1:])
    args = parse_args(["--tfrecord", "tst/tensorflow/data/voc/voc.tfrecord"])
    visualizeBbox(args)
    ds = voc_dataloader(args)
    tf_ds = tf.data.TFRecordDataset([ds.config.tfrecord])
    record = next(iter(tf_ds))
    record = decode_fn(record)
    img, xb, yb, wb, hb, labels = load(record)
    print(xb, yb, wb, hb, labels)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(3):
        xmin = xb[i] - wb[i] // 2
        xmax = xb[i] + wb[i] // 2
        ymin = yb[i] - hb[i] // 2
        ymax = yb[i] + hb[i] // 2
        rect = rectangle(xmin, ymin, xmax, ymax)
        _ = ax.add_patch(rect)
    plt.show()
    pdb.set_trace()
