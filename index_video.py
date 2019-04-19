
"""Imports"""
import numpy as np
import os
import tensorflow as tf
import requests
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

"""Load a Frozen Graph"""
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
PATH_TO_FROZEN_GRAPH = 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

"""Create the category index"""
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

""" Prediction"""
path = 'samples/footage.avi'
cap = cv2.VideoCapture('samples/footage2.mp4')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   while (True):
      # Get the image
      ret, image_np = cap.read()
      key = cv2.waitKey(1) & 0xff
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      
     
      # Visualization of the results of a detection.
      image_np = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
         line_thickness=4)
     

      cv2.imshow('image',cv2.resize(image_np,(720,480)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          #cap.release()
          break
