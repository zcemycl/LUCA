from typing import Any
from xml.etree.ElementTree import Element


class mypy_xmlTree:
    @staticmethod
    def find(element: Element, tag: str) -> Element:
        result = element.find(tag)
        assert (
            result is not None
        ), f"""Nog tag {tag} found\
            in element {element}"""
        assert isinstance(result.text, str)
        return result

    @staticmethod
    def getText(element: Element) -> Any:
        result: Any = element.text
        return result
