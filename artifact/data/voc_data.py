import argparse
import pdb
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as ptc
import matplotlib.pyplot as plt

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


def visualizeBbox(args: argparse.Namespace):
    pathann = Path(args.vocroot) / Path("Annotations")
    pathjpg = Path(args.vocroot) / Path("JPEGImages")
    img = mpimg.imread(pathjpg / Path(args.vocid + ".jpg"))
    tree = ET.parse(pathann / Path(args.vocid + ".xml"))
    root = tree.getroot()
    xmlinfo = extractXml(root)
    print(xmlinfo)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for (xmin, ymin, xmax, ymax) in xmlinfo["bbox"]:
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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    visualizeBbox(args)
    ds = voc_dataloader(args)
    tf_ds = tf.data.TFRecordDataset([ds.config.tfrecord])
    record = next(iter(tf_ds))
    record = decode_fn(record)
    img = tf.io.read_file(record["path"])
    img = tf.io.decode_jpeg(img)
    plt.imshow(img)
    plt.show()
    pdb.set_trace()
