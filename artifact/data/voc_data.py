import argparse
import pdb
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Union

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


def plotBboxAndImg(img: Union[np.ndarray, tf.Tensor], info: Dict[str, Any]):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for (xmin, ymin, xmax, ymax) in info["bbox"]:
        rect = ptc.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
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
def load(record: Dict[str, Any], mode: str = "train"):
    img = tf.io.read_file(record["path"])
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img, tf.float32) / 255.0

    if mode == "train":
        dims = tf.shape(img)
        h = tf.cast(dims[0], tf.float32)
        w = tf.cast(dims[1], tf.float32)
        aspect_ratio = h / w
        print(h, w, aspect_ratio)
        flipx = tf.random.uniform([], 0, 1)
        flipy = tf.random.uniform([], 0, 1)
        img = tf.cond(
            tf.less(flipx, 0.5),
            lambda: img,
            lambda: tf.image.flip_left_right(img),
        )
        img = tf.cond(
            tf.less(flipy, 0.5),
            lambda: img,
            lambda: tf.image.flip_up_down(img),
        )

        xb = record["image/x"]

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
    return img, xb


if __name__ == "__main__":
    # args = parse_args(sys.argv[1:])
    args = parse_args(["--tfrecord", "tst/tensorflow/data/voc/voc.tfrecord"])
    visualizeBbox(args)
    ds = voc_dataloader(args)
    tf_ds = tf.data.TFRecordDataset([ds.config.tfrecord])
    record = next(iter(tf_ds))
    record = decode_fn(record)
    img, xb = load(record)
    plt.imshow(img)
    plt.show()
    pdb.set_trace()
