
"""Imports"""
import numpy as np
import os
import tensorflow as tf
import requests
import cv2

from utils import label_map_util
from utils import visualization_utils as vis_util

boxes_temp = 0
classes_temp = 0

"""Load a Frozen Graph"""
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
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
cap = cv2.VideoCapture('samples/footage.mp4')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
   while (True):
      # Get the image
      ret, image_np = cap.read()
      #image_np = image_np[200:359, 75:639]
      tl = [220, 160]
      tr = [448, 160]
      bl = [64, 359]
      br = [560, 359]
      
      pts1 = np.float32([tl, tr, bl, br])
      pts2 = np.float32([[0, 0], [500, 0], [0, 200], [500, 200]])
      
      matrix = cv2.getPerspectiveTransform(pts1, pts2)
      image_np = cv2.warpPerspective(image_np, matrix, (500, 200))
      
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
      
      
      boxes_prev = boxes_temp
      boxes_temp = boxes
      
      classes_prev = classes_temp
      classes_temp = classes
      
      # Visualization of the results of a detection.
      image_np = vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
         line_thickness=4)
     

      cv2.imshow('image',cv2.resize(image_np,(640,360)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          #cap.release()
          break
