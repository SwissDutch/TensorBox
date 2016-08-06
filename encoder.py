from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from utils import googlenet_load


def inference(hypes, images, encoder_net, train=True):
    # Load googlenet and returns the cnn_codes

    # encoder_net = googlenet_load.init(hypes, config=config)

    # grid_size = hypes['grid_width'] * hypes['grid_height']
    input_mean = 117.
    images -= input_mean
    cnn, early_feat, _ = googlenet_load.model(images, encoder_net, hypes)

    return cnn, early_feat, _
