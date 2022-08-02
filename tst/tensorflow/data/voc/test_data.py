import os
from xml.etree.ElementTree import Element

import pytest

import tensorflow as tf
from src.tensorflow.data.voc.data import (
    decode_fn,
    encode_fn,
    extractXml,
    load,
    main,
    parse_args,
    path2XmlRoot,
    voc_dataloader,
)
from src.tensorflow.utils.fixmypy import mypy_xmlTree

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel("ERROR")
find = mypy_xmlTree.find
getText = mypy_xmlTree.getText


@pytest.fixture
def xml_root() -> Element:
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    print(TEST_DIR)
    root = path2XmlRoot(TEST_DIR + "/Annotations/2010_002107.xml")
    return root


def test_parse_args():
    args = parse_args([])
    assert isinstance(args.vocroot, str)


def test_extractXml(xml_root: Element):
    root = xml_root
    res = extractXml(root)
    assert res["label"] == ["car", "car", "person"]
    assert res["filename"] == "2010_002107.jpg"
    assert isinstance(res["bbox"], list)


def test_decode_encode():
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    args = parse_args(
        ["--vocroot", TEST_DIR, "--tfrecord", TEST_DIR + "/voc.tfrecord"]
    )
    ds = main(args)
    datum = ds.data[0]
    bytesInfo = encode_fn(
        datum["path"],
        datum["xs"],
        datum["ys"],
        datum["ws"],
        datum["hs"],
        datum["ids"],
    ).SerializeToString()
    tmp = decode_fn(bytesInfo)
    assert ds.data[0]["xs"] == tmp["image/x"].numpy().tolist()
    assert ds.data[0]["ys"] == tmp["image/y"].numpy().tolist()
    assert ds.data[0]["ws"] == tmp["image/w"].numpy().tolist()
    assert ds.data[0]["hs"] == tmp["image/h"].numpy().tolist()
    assert ds.data[0]["ids"] == tmp["labels"].numpy().tolist()
    assert ds.data[0]["path"] == tmp["path"].numpy().decode("utf-8")


class TestLoadData(tf.test.TestCase):
    def setUp(self):
        super(TestLoadData, self).setUp()
        TEST_DIR = os.path.dirname(os.path.abspath(__file__))
        self.tfrecordPath = TEST_DIR + "/voc.tfrecord"
        self.tf_ds = tf.data.TFRecordDataset([self.tfrecordPath])
        self.record = next(iter(self.tf_ds))
        self.record = decode_fn(self.record)
        self.record["path"] = tf.constant(
            TEST_DIR + "/JPEGImages/2010_002107.jpg"
        )
        self.ds = voc_dataloader(
            parse_args(
                [
                    "--vocroot",
                    TEST_DIR,
                    "--tfrecord",
                    TEST_DIR + "/voc.tfrecord",
                ]
            )
        )

    def testLoad(self):
        for i in ["normal", "karrp", "karcp"]:
            with self.subTest("custom message", i=i):
                img, bbox = load.__wrapped__(self.record, resize_mode=i)
                self.assertAllEqual(img.numpy().shape, (416, 416, 3))

    # def testLoadGT(self):
    #     img, y1, y2, y3 = self.ds.loadGroundTruth.__wrapped__(self.record)
    #     self.assertAllEqual(img.numpy().shape, (416, 416, 3))

    def test_preprocess_true_boxes(self):
        xy = tf.random.uniform([2, 2], 100, 200, tf.float32)
        wh = tf.random.uniform([2, 2], 10, 20, tf.float32)
        labels = tf.random.uniform([2, 1], 0, 10, tf.float32)
        bbox = tf.concat([xy, wh, labels], axis=-1)
        res = self.ds.preprocess_true_boxes(bbox)
        feature_map_dims = [13, 26, 52]
        for i in range(self.ds.num_layers):
            dim = feature_map_dims[i]
            with self.subTest("custom message", i=i):
                self.assertAllEqual(
                    res[i].shape, (dim, dim, self.ds.num_anchors_layers[i], 25)
                )


def test_main():
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    args = parse_args(
        ["--vocroot", TEST_DIR, "--tfrecord", TEST_DIR + "/voc.tfrecord"]
    )
    ds = main(args)
    assert len(ds.label2id) == 2
