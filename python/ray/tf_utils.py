from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import os

logger = logging.getLogger(__name__)


def unflatten(vector, shapes):
    i = 0
    arrays = []
    for shape in shapes:
        size = np.prod(shape, dtype=np.int)
        array = vector[i:(i + size)].reshape(shape)
        arrays.append(array)
        i += size
    assert len(vector) == i, "Passed weight does not have the correct shape."
    return arrays


def try_import_tf():
    """Attempt to import TensorFlow.
    """
    logger = logging.getLogger("ray")
    if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
        logger.warning("Not importing TensorFlow for test purposes")
        return None

    try:
        if "TF_CPP_MIN_LOG_LEVEL" not in os.environ:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow.compat.v1 as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.disable_v2_behavior()
        return tf
    except ImportError:
        try:
            import tensorflow as tf
            return tf
        except ImportError:
            return None


def try_import_tfp():
    """Attempt to import TensorFlow probability.
    """
    logger = logging.getLogger("ray")
    if "RLLIB_TEST_NO_TF_IMPORT" in os.environ:
        logger.warning(
            "Not importing TensorFlow Probability for test purposes.")
        return None

    try:
        import tensorflow_probability as tfp
        return tfp
    except ImportError:
        return None
