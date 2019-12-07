from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random
from ray.tf_utils import try_import_tf

tf = try_import_tf()


def seed(np_seed=0, random_seed=0, tf_seed=0):
    np.random.seed(np_seed)
    random.seed(random_seed)
    tf.set_random_seed(tf_seed)
