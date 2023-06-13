import numpy as np
import cv2

# Models: https://github.com/richzhang/colorization/tree/caffe/colorization/models
# Points: https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy

prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'lion.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)