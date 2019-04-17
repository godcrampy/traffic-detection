
"""Imports"""
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from utils import label_map_util
from utils import visualization_utils as vis_util

"""Load a Frozen Graph"""
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
PATH_TO_FROZEN_GRAPH = 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')