import os
from typing import Text
import cv2
import numpy as np
import onnx
import app.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
# onnx runtime
import onnxruntime as ort
#---------------------
import dlib
import numpy as np
from imutils import face_utils
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
import app.facenet as facenet
import threading
#----------------------
onnx_path = "app/models/onnx/version-RFB-640.onnx"

predictor = onnx.load(onnx_path)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name
required_shape = (160,160)

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,iou_threshold=iou_threshold,top_k=top_k)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

shape_predictor = dlib.shape_predictor('app/models/facial_landmarks/shape_predictor_5_face_landmarks.dat')
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

TRAINING_BASE = 'app/faces/training/'

class Train_Model_Thread(threading.Thread):
    def __init__(self,):
        threading.Thread.__init__(self)

    def start(self):
        self.text = "Start Training"
        dirs = os.listdir(TRAINING_BASE)
        images = []
        names = []
        print(self.text)
        for label in dirs:
            for i, fn in enumerate(os.listdir(os.path.join(TRAINING_BASE, label))):
                self.text = f"Start collecting faces from {label}'s data"
                print(self.text)
                cap = cv2.VideoCapture(os.path.join(TRAINING_BASE, label, fn))
                frame_count = 0
                while True:
                    # read video frame
                    ret, raw_img = cap.read()
                    # process every 5 frames
                    if frame_count % 5 == 0 and raw_img is not None:
                        h, w, _ = raw_img.shape
                        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (640, 480))
                        img_mean = np.array([127, 127, 127])
                        img = (img - img_mean) / 128
                        img = np.transpose(img, [2, 0, 1])
                        img = np.expand_dims(img, axis=0)
                        img = img.astype(np.float32)

                        confidences, boxes = ort_session.run(None, {input_name: img})
                        boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)

                        # if face detected
                        if boxes.shape[0] > 0:
                            x1, y1, x2, y2 = boxes[0,:]
                            gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
                            aligned_face = fa.align(raw_img, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                            aligned_face = cv2.resize(aligned_face, required_shape)
                            cv2.imwrite(f'app/faces/tmp/{label}_{frame_count}_{i}.jpg', aligned_face)
                            aligned_face = aligned_face - 127.5
                            aligned_face = aligned_face * 0.0078125
                            images.append(aligned_face)
                            names.append(label)
                    frame_count += 1
                    if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        break

        with tf.Graph().as_default():
            with tf.Session() as sess:
                facenet.load_model('app/models/facenet/20180402-114759.pb')
                self.text = "Start extracting face features"
                print(self.text)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                embeds = sess.run(embeddings, feed_dict=feed_dict)
                with open("app/embeddings/facemodel.pkl", "wb") as f:
                    pickle.dump((embeds, names), f)
                self.text = "Done!"
                print(self.text)

    def get_process(self,):
        return self.text