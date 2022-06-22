import pytest

import tensorflow as tf
from src.tensorflow.model.interface.network_bbox import Network_Bbox


def test_backbone_NotImplementedError():
    x = tf.keras.layers.Input([416, 416, 3])
    with pytest.raises(NotImplementedError) as _:
        Network_Bbox.Backbone(x)
