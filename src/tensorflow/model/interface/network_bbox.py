import argparse
from abc import abstractmethod

import tensorflow as tf


class Network_Bbox:
    config: argparse.Namespace

    def __init__(self, config: argparse.Namespace):
        self.config = config

    @abstractmethod
    def Backbone(
        self,
        x: tf.Tensor,
        alpha: float = 1.0,
        include_top: bool = False,
        weights: str = "imagenet",
    ) -> tf.keras.Sequential:
        raise NotImplementedError

    @abstractmethod
    def Network(self) -> tf.keras.Model:
        raise NotImplementedError
