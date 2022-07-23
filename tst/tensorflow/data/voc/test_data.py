import os
from xml.etree.ElementTree import Element

import pytest

from src.tensorflow.data.voc.data import (
    extractXml,
    main,
    parse_args,
    path2XmlRoot,
)
from src.tensorflow.utils.fixmypy import mypy_xmlTree

find = mypy_xmlTree.find
getText = mypy_xmlTree.getText


@pytest.fixture
def xml_root() -> Element:
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
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


def test_main():
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    args = parse_args(["--vocroot", TEST_DIR])
    main(args)
