"""
This code uses the onnx model to detect faces from live video or cameras.
"""
from app.facenet import train
import cv2, os
import numpy as np
import onnx
import app.vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
# onnx runtime
import onnxruntime as ort
#-----------------------------------
import dlib
from imutils import face_utils
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
import pickle
#-----------------------------------
from configparser import ConfigParser
config = ConfigParser()
config.read('app/config.ini')
from app.webcam import Webcam
from app.camera import VideoCamera
from threading import Thread
#-----------------------------------
onnx_path = config.get('path','onnx_path')
predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

shape_predictor = dlib.shape_predictor(config.get('path','dlib_5_landmarks'))
fa = face_utils.facealigner.FaceAligner(shape_predictor, desiredFaceWidth=112, desiredLeftEye=(0.3, 0.3))

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
        box_probs = box_utils.hard_nms(box_probs,iou_threshold=iou_threshold,top_k=top_k,)
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

def nothing(x):
    pass

FACENET_MODEL_PATH = config.get('path','facenet_model_path')
with open(config.get('path','pickle_path'), "rb") as f:
    (saved_embeds, names) = pickle.load(f)

#---------------------------------------------
class Processing:
    def __init__(self,threshold_value):
        self.webcam = VideoCamera()
        self.get_threshold(threshold_value)
        self.flag = True
        self.recording = False
        self.ret = self.webcam.get_frame()[0]
        self.frame_ver = self.frame_raw = self.frame_detect = self.webcam.get_frame()[1]
        self.name_list =[]
        self.state = "verification"
        
    def get_threshold(self,threshold_value):
        self.face_detect_threshold = int(threshold_value['face_detect_threshold'])/100
        self.small_face_delete_threshold = int(threshold_value['small_face_delete_threshold'])
        self.face_verification_threshold = int(threshold_value['face_verification_threshold'])/100

    def start(self):
        Thread(target=self._processing,args=()).start()

    def get_fps(self,timer,frame):
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if fps>60: myColor = (20,230,20)
        elif fps>20: myColor = (230,20,20)
        else: myColor = (20,20,230)
        cv2.putText(frame,str(int(fps)), (75, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, myColor, 2)
        return frame

    def _processing(self):        
        with tf.Graph().as_default():
            with tf.compat.v1.Session() as sess:
                with tf.gfile.GFile(FACENET_MODEL_PATH,'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map=None, name='')
                images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]
                font = cv2.FONT_HERSHEY_DUPLEX
                while (self.flag):
                    self.ret, frame = self.webcam.get_frame()
                    if not self.ret:
                        break
                    timer = cv2.getTickCount()

                    # preprocess faces
                    h, w, _ = frame.shape
                    if eval(config.get('camera', 'flip')):
                        frame = cv2.flip(frame,1)
                    if self.recording == True:
                        self.out.write(frame)
                        cv2.waitKey(10)
                        cv2.putText(frame, "RECORD", (w-150, 40), font, 1.2, (0,0,256), 3)
                    self.frame_raw = frame.copy()
                    self.frame_raw = self.get_fps(timer,self.frame_raw)

                    if self.state == "detection" or self.state == "verification":
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (640, 480))
                        image_mean = np.array([127, 127, 127])
                        image = (image - image_mean) / 128
                        image = np.transpose(image, [2, 0, 1])
                        image = np.expand_dims(image, axis=0)
                        image = image.astype(np.float32)

                        # detect faces
                        confidences, boxes = ort_session.run(None, {input_name: image})
                        boxes, labels, probs = predict(frame.shape[1], frame.shape[0], confidences, boxes, self.face_detect_threshold)

                        # locate faces
                        faces = []
                        box_idx=[]
                        name_list = []
                        boxes[boxes<0] = 0
                        
                        for i in range(boxes.shape[0]):
                            box = boxes[i, :]
                            x1, y1, x2, y2 = box
                            #delete small face
                            if abs (x1-x2) > self.small_face_delete_threshold:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                aligned_face = fa.align(frame, gray, dlib.rectangle(left = x1, top=y1, right=x2, bottom=y2))
                                aligned_face = cv2.resize(aligned_face, (160,160))
                                aligned_face = aligned_face - 127.5
                                aligned_face = aligned_face * 0.0078125
                                faces.append(aligned_face)
                                box_idx.append(i)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
                        self.frame_detect = frame.copy()
                        self.frame_detect = self.get_fps(timer,self.frame_detect)

                        if self.state == "verification":
                            if len(faces)>0:
                                predictions = []
                                faces = np.array(faces)
                                feed_dict = { images_placeholder: faces, phase_train_placeholder:False }
                                embeds = sess.run(embeddings, feed_dict=feed_dict)

                                # prediciton using distance
                                for i , embedding in enumerate(embeds):
                                    diff = np.subtract(saved_embeds, embedding)
                                    dist = np.sum(np.square(diff), 1)
                                    idx = np.argmin(dist)
                                    # print(f"face{i}",names[idx]," ",dist[idx])
                                    if dist[idx] < self.face_verification_threshold:
                                        predictions.append(names[idx])
                                        name_list.append(names[idx])
                                        # export_csvy.markAttendance(names[idx])
                                    else:
                                        predictions.append("unknown")
                                    box = boxes[box_idx[i], :]
                                    text = f"{predictions[i]}"
                                    x1, y1, x2, y2 = box
                                    # Draw a label with a name below the face
                                    cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
                                    cv2.putText(frame, text, (x1 + 6, (y2 - 6)), font, 0.3, (255, 255, 255), 1)
                            self.frame_ver = frame.copy()
                            self.frame_ver = self.get_fps(timer,self.frame_ver)
                            self.name_list = name_list.copy()
        # self.webcam.close()

    def get_name_list(self):
        return self.name_list

    def get_frame(self,state):
        self.state = state
        if state == "verification":
            return self.ret,self.frame_ver
        elif state == "detection":
            return self.ret, self.frame_detect
        else:
            return self.ret, self.frame_raw

    def start_record(self,train_name):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        train_name = train_name.title()
        if not os.path.exists('./app/faces/training/'+ train_name):
            os.makedirs('./app/faces/training/'+ train_name)
        num_file = len(os.listdir('./app/faces/training/'+ train_name))
        self.out = cv2.VideoWriter('./app/faces/training/'+ train_name + f'/{train_name}_{num_file+1}.avi',fourcc, 20.0, (640,480))
        self.recording = True

    def stop_record(self):
        self.recording = False
        self.out.release()

    def close(self):
        threshold = np.array(list([self.face_detect_threshold*100,
                                    self.small_face_delete_threshold,
                                    self.face_verification_threshold*100]))
        np.savetxt(config.get('path','threshold'),threshold)
        self.flag = False
        return self.flag  
