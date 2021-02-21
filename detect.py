import cv2 as cv
import tensorflow as tf
import numpy as np
import dlib
import face_recognition
import threading

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class MaskDetect:
    def __init__(self):
        self.mask_count = 0
        self.mask_is_wearing = True
        self.mask_detect_completed = True
        self.face_location = []
        self.model = tf.keras.models.load_model('models/mask_detector.model')
        self.face_detector = cv.dnn.readNetFromCaffe("models/deploy.prototxt", "models/res10_300x300_ssd_iter_140000.caffemodel")
        self.face_detected = False
        self.chunjie_encoding = np.loadtxt("chunjie_encoding", delimiter=',')
        self.name = "Unknown"
        self.known_face_encoding = [self.chunjie_encoding]
        self.known_face_name = ["Chunjie Shan"]
        self.left = 0
        self.right = 0
        self.bottom = 0
        self.top = 0

    def detect_mask(self, frame):
        resized = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        resized = cv.resize(resized, (224, 224))

        resized = tf.keras.preprocessing.image.img_to_array(resized)
        resized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)

        resized = np.expand_dims(resized, axis=0)

        mask, without_mask = self.model.predict([resized])[0]
        print(mask)

        return mask

    def recognize(self, img_src, thread_name=None):
        global left, right, top, bottom
        self.name = "Unknown"

        # face_location = [(top, right, bottom, left)]
        face_location = [(self.top, self.right, self.bottom, self.left)]
        # face_location = face_recognition.face_locations(img_gray)
        face_encoding = face_recognition.face_encodings(img_src, face_location)
        if face_location:
            for (top, right, bottom, left), face_encoding in zip(face_location, face_encoding):
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encoding, face_encoding)

                # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encoding, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    self.name = self.known_face_name[best_match_index]
                print(self.name)

                # img_src = cv.rectangle(img_src, (left, top), (right, bottom), (255, 0, 0), thickness=2, lineType=8)
                img_src = cv.rectangle(img_src, (left, bottom), (right, bottom + 25), (255, 0, 0), thickness=-1, lineType=8)
                img_src = cv.putText(img_src, self.name, (left, bottom + 22), cv.FONT_ITALIC, 0.7, (255, 255, 255), thickness=2)

        return img_src

    def draw_landmarks(self, img_src, img_gray):
        landmark_list = face_recognition.face_landmarks(img_gray)
        landmark_frame = np.zeros((480, 640))

        if landmark_list:
            self.face_detected = True
            for landmark in landmark_list:
                for facial_feature in landmark.keys():
                    for position in landmark[facial_feature]:
                        landmark_frame = cv.circle(img_src, position, radius=3, color=(0, 0, 255), thickness=-1, lineType=8)
            return landmark_frame
        else:
            self.face_detected = False
            return img_src

    def face_detection(self, frame):
        rows = frame.shape[0]
        cols = frame.shape[1]
        scores = []
        # if frame.shape > (300, 300):
        #     crop = cv.resize(frame, (300, 300))
        self.face_detector.setInput(cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
        cv_out = self.face_detector.forward()
        for detection in cv_out[0, 0, :, :]:
            # print(detection)
            class_id = int(detection[1])
            score = float(detection[2])
            if score > 0.5:
                self.left = int(detection[3] * cols)
                self.top = int(detection[4] * rows)
                self.right = int(detection[5] * cols)
                self.bottom = int(detection[6] * rows)
                (start_x, start_y) = (max(0, self.left), max(0, self.top))
                (end_x, end_y) = (min(640 - 1, self.right), min(480 - 1, self.bottom))
                face_image = frame[start_y: end_y, start_x: end_x]
                result = self.detect_mask(face_image)
                self.face_detected = True
                if result > 0.5:
                    color = (0, 255, 0)
                    label = "Mask"
                    cv.putText(frame, label, (self.left, self.top + 22), cv.FONT_ITALIC, 0.8, color, thickness=2)
                    self.mask_is_wearing = True
                else:
                    color = (0, 0, 255)
                    label = "No Mask"
                    cv.putText(frame, label, (self.left, self.top + 22), cv.FONT_ITALIC, 0.8, color, thickness=2)
                    self.mask_is_wearing = False

                cv.rectangle(frame, (self.left, self.top), (self.right, self.bottom), color, thickness=2)

                if self.mask_is_wearing is not True:
                    self.recognize(frame)
        return frame

    def detect_mask_with_camera(self, thread_name=None):
        capture = cv.VideoCapture(0)
        while True:
            ret, frame = capture.read()
            if ret is True:
                frame = self.face_detection(frame)

                cv.imshow("video-input", frame)
                c = cv.waitKey(50)
                if c == 27:
                    break
            else:
                break
        capture.release()


class MyThread(threading.Thread):
    def __init__(self, thread_id, name):
        threading.Thread.__init__(self)
        self.thread_id = thread_id
        self.name = name

    def run(self):
        print("Thread beginning")
        thread_lock.acquire()
        mask_detect = MaskDetect()
        mask_detect.detect_mask_with_camera(thread_name=self.name)
        thread_lock.release()


if __name__ == '__main__':
    # result = detect_mask("88.jpg")
    # thread_1 = MyThread(1, "Thread-1")
    # thread_2 = MyThread(2, "Thread-2")
    #
    # thread_1.start()
    # thread_2.start()
    # thread_1.join()
    # thread_2.join()
    video_detect = MaskDetect()
    video_detect.detect_mask_with_camera()
