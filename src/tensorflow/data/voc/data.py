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
        xmin = int(float(getText(find(bndbox, "xmin"))))
        ymin = int(float(getText(find(bndbox, "ymin"))))
        xmax = int(float(getText(find(bndbox, "xmax"))))
        ymax = int(float(getText(find(bndbox, "ymax"))))
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
        self.label2id: Dict[str, int] = {}
        self.id2label: Dict[int, str] = {}
        self.data = self.extractAllFiles()

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


def main(args: argparse.Namespace):
    ds = voc_dataloader(args)
    print(ds.label2id, ds.id2label, ds.data[:2])
    return ds


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
