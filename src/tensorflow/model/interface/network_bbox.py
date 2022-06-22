from abc import abstractmethod

import tensorflow as tf


class Network_Bbox:
    @staticmethod
    @abstractmethod
    def Backbone(
        x: tf.Tensor,
        alpha: float = 1.0,
        include_top: bool = False,
        weights: str = "imagenet",
    ) -> tf.keras.Sequential:
        raise NotImplementedError
