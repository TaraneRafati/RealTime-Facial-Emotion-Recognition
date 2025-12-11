import numpy as np
import cv2
import tensorflow as tf
from face_detector import YoloDetector
import glob

def prep(frame):
    face = cv2.resize(frame, (48,48))
    face = np.array(face)
    face = face / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face

def pred(frame, model):
    labels = ['Angry', 'Happy', 'Sad', 'Surprised', 'Poker']
    face = prep(frame)
    pred = model.predict(face, verbose = 0)
    pred = np.argmax(pred, axis=1)[0]
    frame = cv2.resize(frame, (480,480))
    frame = cv2.putText(frame, labels[pred], (240, 240), fontFace=1, fontScale=1, color=(255,255,255))
    return frame



yolo = YoloDetector(device="cpu", min_face=20)
model = tf.keras.models.load_model('./output/face_classification_model.keras')
path = './dataset/test/1/'
while True:
    files = glob.glob(path + '*.jpg')
    print(files)
    for image in files:
        frame = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        frame = pred(frame, model)
        cv2.imshow("my stream", frame)
        cv2.waitKey(1000)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  
cv2.destroyAllWindows()