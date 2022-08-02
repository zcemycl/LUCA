import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from xml.etree.ElementTree import Element

import numpy as np

import tensorflow as tf
from src.tensorflow.loss.voc.loss import compute_giou
from src.tensorflow.utils.fixmypy import mypy_xmlTree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
find = mypy_xmlTree.find
getText = mypy_xmlTree.getText


def parse_args(args: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vocroot",
        type=str,
        default="/media/yui/8TB_partition2/data/cv/VOCdevkit/VOC2012",
        help="Path which contains JPEGImages and Annotations (xml).",
    )
    p.add_argument(
        "--vocid",
        type=str,
        default="2010_002107",
        help="""Specific filename in voc dataset\
excluding extension like xml and jpg.""",
    )
    p.add_argument(
        "--tfrecord",
        type=str,
        default="artifact/data/voc.tfrecord",
        help="Where to save/load TFRecord.",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=["none", "convert", "loader"],
        help="""dataloader mode ... \
1. None: No mode. \
2. Convert: Convert xml files to TFRecord. \
""",
    )
    p.add_argument(
        "--resize_mode",
        type=str,
        default="karrp",
        choices=["normal", "karrp", "karcp"],
        help="""1. normal: not keep aspect ratio. \
2. karrp: keep aspect ratio random pad. \
3. karcp: keep aspect ratio center pad. \
""",
    )
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
    p.add_argument("--batch_size", type=int, default=16, help="Batch size")
    p.add_argument(
        "--buffer_size", type=int, default=400, help="Shuffle buffer size"
    )
    args_, _ = p.parse_known_args(args)
    return args_


def path2XmlRoot(path: Union[str, Path]) -> Element:
    tree = ET.parse(path)
    root = tree.getroot()
    return root


def extractXml(root: Element) -> Dict[str, Any]:
    """
    filename: str
    label: List[str]
    bbox: List[List[int]] (x1,y1,x2,y2)
    """
    res, label, bbox = {}, [], []
    res["filename"] = getText(find(root, "filename"))
    for obj in root.findall("object"):
        label.append(getText(find(obj, "name")))
        bndbox = find(obj, "bndbox")
        xmin = int(float(getText(find(bndbox, "xmin"))))
        ymin = int(float(getText(find(bndbox, "ymin"))))
        xmax = int(float(getText(find(bndbox, "xmax"))))
        ymax = int(float(getText(find(bndbox, "ymax"))))
        bbox.append([xmin, ymin, xmax, ymax])
    res["label"] = label
    res["bbox"] = bbox
    return res


def encode_fn(
    path: str,
    x: List[int],
    y: List[int],
    w: List[int],
    h: List[int],
    label: List[int],
) -> Any:
    return tf.train.Example(
        features=tf.train.Features(
            feature={
                "path": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[bytes(path, "utf-8")])
                ),
                "image/x": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=x)
                ),
                "image/y": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=y)
                ),
                "image/w": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=w)
                ),
                "image/h": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=h)
                ),
                "labels": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=label)
                ),
            }
        )
    )


def decode_fn(record_bytes: tf.Tensor):
    return tf.io.parse_single_example(
        record_bytes,
        {
            "path": tf.io.FixedLenFeature([], dtype=tf.string),
            "image/x": tf.io.FixedLenSequenceFeature(
                [], dtype=tf.int64, allow_missing=True
            ),
            "image/y": tf.io.FixedLenSequenceFeature(
                [], dtype=tf.int64, allow_missing=True
            ),
            "image/w": tf.io.FixedLenSequenceFeature(
                [], dtype=tf.int64, allow_missing=True
            ),
            "image/h": tf.io.FixedLenSequenceFeature(
                [], dtype=tf.int64, allow_missing=True
            ),
            "labels": tf.io.FixedLenSequenceFeature(
                [], dtype=tf.int64, allow_missing=True
            ),
        },
    )


def random_flip_bbox(
    img: tf.Tensor, target_dims: tf.Tensor, xb: tf.Tensor, yb: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
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
    return img, xb, yb


def random_image_colormap(img: tf.Tensor) -> tf.Tensor:
    img = tf.image.random_hue(img, 0.5)
    img = tf.image.random_saturation(img, 0.5, 1.5)
    img = tf.image.random_brightness(img, 0.1)
    # val = tf.random.uniform([], 0.8, 2)
    # img = tf.image.adjust_gamma(img, val)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_jpeg_quality(img, 80, 100)
    # add gaussian noise
    img += tf.cast(
        tf.random.uniform(shape=tf.shape(img), minval=0, maxval=0.05),
        tf.float32,
    )
    return img


def resize_bbox(
    img: tf.Tensor,
    xb: tf.Tensor,
    yb: tf.Tensor,
    wb: tf.Tensor,
    hb: tf.Tensor,
    target_dims: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    dims = tf.cast(tf.shape(img), tf.float32)
    rw = target_dims[1] / dims[1]
    rh = target_dims[0] / dims[0]
    img = tf.image.resize(img, [target_dims[0], target_dims[1]])
    xb *= rw
    yb *= rh
    wb *= rw
    hb *= rh
    return img, xb, yb, wb, hb


def resize_bbox_keep_aspect(
    img: tf.Tensor,
    xb: tf.Tensor,
    yb: tf.Tensor,
    wb: tf.Tensor,
    hb: tf.Tensor,
    target_dims: tf.Tensor,
    center_pad: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    zero = tf.constant(0, tf.int64)
    tar_aspect_ratio = target_dims[0] / target_dims[1]
    dims = tf.shape(img)
    h = tf.cast(dims[0], tf.float32)
    w = tf.cast(dims[1], tf.float32)
    aspect_ratio = h / w

    aspect_cond = tf.less(aspect_ratio, tar_aspect_ratio)
    r = tf.cond(
        aspect_cond, lambda: target_dims[1] / w, lambda: target_dims[0] / h
    )
    tar_w = tf.cond(aspect_cond, lambda: target_dims[1], lambda: r * w)
    tar_h = tf.cond(aspect_cond, lambda: r * h, lambda: target_dims[0])
    img = tf.image.resize(img, [tar_h, tar_w])
    wb *= r
    hb *= r
    xb *= r
    yb *= r
    dw = tf.cast(target_dims[1] - tf.floor(tar_w), tf.int64)
    dh = tf.cast(target_dims[0] - tf.floor(tar_h), tf.int64)
    if not center_pad:
        cond_dx_eq0 = tf.equal(zero, dw)
        cond_dy_eq0 = tf.equal(zero, dh)
        dx = tf.cond(
            aspect_cond,
            lambda: zero,
            lambda: tf.cond(
                cond_dx_eq0,
                lambda: zero,
                lambda: tf.random.uniform([], 0, dw, tf.int64),
            ),
        )
        dy = tf.cond(
            aspect_cond,
            lambda: tf.cond(
                cond_dy_eq0,
                lambda: zero,
                lambda: tf.random.uniform([], 0, dh, tf.int64),
            ),
            lambda: zero,
        )
    else:
        dx = dw // 2
        dy = dh // 2
    xb, yb = tf.cond(
        aspect_cond,
        lambda: (xb, yb + tf.cast(dy, tf.float32)),
        lambda: (xb + tf.cast(dx, tf.float32), yb),
    )
    img = tf.pad(
        img, tf.convert_to_tensor([[dy, dh - dy], [dx, dw - dx], [0, 0]])
    )
    return img, xb, yb, wb, hb


@tf.function
def load(
    record: Dict[str, Any],
    target_dims: tf.Tensor = tf.constant([416, 416], tf.float32),
    mode: str = "train",
    resize_mode: str = "karrp",  # normal,karrp,karcp
) -> Tuple[tf.Tensor, tf.Tensor]:
    img = tf.io.read_file(record["path"])
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img, tf.float32) / 255.0
    xb = tf.cast(record["image/x"], tf.float32)
    yb = tf.cast(record["image/y"], tf.float32)
    wb = tf.cast(record["image/w"], tf.float32)
    hb = tf.cast(record["image/h"], tf.float32)
    labels = tf.cast(record["labels"], tf.float32)

    if mode == "train":
        if resize_mode == "normal":
            img, xb, yb, wb, hb = resize_bbox(img, xb, yb, wb, hb, target_dims)
        elif resize_mode == "karrp":
            img, xb, yb, wb, hb = resize_bbox_keep_aspect(
                img, xb, yb, wb, hb, target_dims
            )
        elif resize_mode == "karcp":
            img, xb, yb, wb, hb = resize_bbox_keep_aspect(
                img, xb, yb, wb, hb, target_dims, center_pad=True
            )
        img, xb, yb = random_flip_bbox(img, target_dims, xb, yb)
        img = random_image_colormap(img)

    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)
    bbox = tf.stack([xb, yb, wb, hb, labels])
    # (number of boxes, 5)
    bbox = tf.transpose(bbox, [1, 0])
    return img, bbox


class voc_dataloader:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.pathann = Path(config.vocroot) / Path("Annotations")
        self.pathjpg = Path(config.vocroot) / Path("JPEGImages")
        self.xmlfiles = os.listdir(self.pathann)
        self.length = len(self.xmlfiles)
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.data = self.extractAllFiles()
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

    def extractAllFiles(self) -> List[Dict[str, Any]]:
        data = []
        for xmlfile in self.xmlfiles:
            fullpath = self.pathann / Path(xmlfile)
            root = path2XmlRoot(fullpath)
            info = extractXml(root)
            # x1y1wh xmin ymin width height
            # to xywh x-center y-center width height
            info["path"] = str(self.pathjpg / Path(info["filename"]))
            ids, xs, ys, ws, hs = [], [], [], [], []
            for i, label in enumerate(info["label"]):
                if label not in self.label2id:
                    id_ = len(self.label2id)
                    self.label2id[label] = id_
                    self.id2label[id_] = label
                bbox = info["bbox"]
                ids.append(id_)
                xs.append((bbox[i][0] + bbox[i][2]) // 2)
                ys.append((bbox[i][1] + bbox[i][3]) // 2)
                ws.append(bbox[i][2] - bbox[i][0])
                hs.append(bbox[i][3] - bbox[i][1])
            info["xs"] = xs
            info["ys"] = ys
            info["ws"] = ws
            info["hs"] = hs
            info["ids"] = ids
            data.append(info)
        return data

    def writeTFRecord(self):
        os.system(f"rm {self.config.tfrecord}")
        with tf.io.TFRecordWriter(self.config.tfrecord) as f_write:
            for datum in self.data:
                record = encode_fn(
                    datum["path"],
                    datum["xs"],
                    datum["ys"],
                    datum["ws"],
                    datum["hs"],
                    datum["ids"],
                ).SerializeToString()
                f_write.write(record)

    def preprocess_true_boxes(
        self, true_boxes: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Preprocess true boxes to training input format
        --------------

        --------------
        """
        num_layers = len(self.num_anchors_layers)
        true_boxes = np.array(true_boxes, dtype="float32")
        input_shape = np.array(self.input_shape, dtype="int32")
        boxes_xy = true_boxes[..., 0:2]
        boxes_wh = true_boxes[..., 2:4]
        # because height width in shape
        true_boxes[..., 0:2] = boxes_xy / input_shape[..., ::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[..., ::-1]

        grid_shapes = [
            np.round(input_shape / [32, 16, 8][layer_id]).astype(np.int32)
            for layer_id in range(num_layers)
        ]
        y_true = [
            np.zeros(
                (
                    grid_shapes[layer_id][0],
                    grid_shapes[layer_id][1],
                    len(self.anchor_mask[layer_id]),
                    5 + self.config.num_classes,
                ),
                dtype="float32",
            )
            for layer_id in range(num_layers)
        ]

        anchors = np.expand_dims(self.anchors, 0)
        anchor_maxes = anchors / 2.0
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0
        wh = boxes_wh[valid_mask]
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.0
        box_mins = -box_maxes
        anchor_box = tf.stack(
            [
                anchor_mins[..., 1],
                anchor_mins[..., 0],
                anchor_maxes[..., 1],
                anchor_maxes[..., 0],
            ],
            axis=-1,
        )
        bbox = tf.stack(
            [
                box_mins[..., 1],
                box_mins[..., 0],
                box_maxes[..., 1],
                box_maxes[..., 0],
            ],
            axis=-1,
        )
        iou = compute_giou(anchor_box, bbox, mode="iou").numpy()
        best_anchor = np.argmax(iou, axis=-1)
        for t, n in enumerate(best_anchor):
            for layer_id in range(num_layers):
                if n in self.anchor_mask[layer_id]:
                    i = np.floor(
                        true_boxes[t, 0] * grid_shapes[layer_id][1]
                    ).astype("int32")
                    j = np.floor(
                        true_boxes[t, 1] * grid_shapes[layer_id][0]
                    ).astype("int32")
                    k = self.anchor_mask[layer_id].index(n)
                    c = true_boxes[t, 4].astype("int32")
                    y_true[layer_id][j, i, k, 0:4] = true_boxes[t, 0:4]
                    y_true[layer_id][j, i, k, 4] = 1
                    y_true[layer_id][j, i, k, 5 + c] = 1

        return y_true[0], y_true[1], y_true[2]

    @tf.function
    def loadGroundTruth(
        self,
        record: Dict[str, Any],
        target_dims: tf.Tensor = tf.constant([416, 416], tf.float32),
        mode: str = "train",
        resize_mode: str = "karrp",  # normal,karrp,karcp
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        img, bbox = load(
            record, target_dims=target_dims, mode=mode, resize_mode=resize_mode
        )
        y1, y2, y3 = tf.py_function(
            self.preprocess_true_boxes,
            [bbox],
            [tf.float32, tf.float32, tf.float32],
        )
        return img, y1, y2, y3

    def get_data(self, mode: str = "train", resize_mode: str = "karrp"):
        ds = tf.data.TFRecordDataset([self.config.tfrecord])
        ds = ds.map(
            lambda x: self.loadGroundTruth(
                decode_fn(x), mode=mode, resize_mode=resize_mode
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # ds = ds.shuffle(self.config.buffer_size)
        # ds = ds.batch(self.config.batch_size)
        return ds


def main(args: argparse.Namespace):
    ds = voc_dataloader(args)
    if args.mode == "convert":
        ds.writeTFRecord()
    elif args.mode == "loader":
        _ = ds.get_data()
    return ds


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
