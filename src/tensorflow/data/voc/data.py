import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Union
from xml.etree.ElementTree import Element

from src.tensorflow.utils.fixmypy import mypy_xmlTree

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
        xmin = int(getText(find(bndbox, "xmin")))
        ymin = int(getText(find(bndbox, "ymin")))
        xmax = int(getText(find(bndbox, "xmax")))
        ymax = int(getText(find(bndbox, "ymax")))
        bbox.append([xmin, ymin, xmax, ymax])
    res["label"] = label
    res["bbox"] = bbox
    return res


class voc_dataloader:
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.pathann = Path(config.vocroot) / Path("Annotations")
        self.pathjpg = Path(config.vocroot) / Path("JPEGImages")
        self.xmlfiles = os.listdir(self.pathann)


def main(args: argparse.Namespace):
    _ = voc_dataloader(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
