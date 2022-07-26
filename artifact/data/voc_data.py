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
    load,
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


if __name__ == "__main__":
    # args = parse_args(sys.argv[1:])
    args = parse_args(["--tfrecord", "tst/tensorflow/data/voc/voc.tfrecord"])
    visualizeBbox(args)
    ds = voc_dataloader(args)
    tf_ds = tf.data.TFRecordDataset([ds.config.tfrecord])
    record = next(iter(tf_ds))
    record = decode_fn(record)
    img, xb, yb, wb, hb, labels = load(record, resize_mode=args.resize_mode)
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
    ax.set_axis_off()
    plt.show()
    pdb.set_trace()
