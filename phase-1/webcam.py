import numpy as np
import cv2
import tensorflow as tf
from face_detector import YoloDetector

def prep(frame, box):
    x1,y1,x2,y2 = box
    face = frame[y1:y2, x1:x2]
    face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (48,48))
    face = np.array(face)
    face = face / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face


def draw_emoji(frame, box, pred):
    emojis = ['angry.png', 'happy.png', 'sad.png', 'surprised.png', 'neutral.png']
    emoji = cv2.imread('./emojis/' + emojis[pred], cv2.IMREAD_UNCHANGED)
    x1,y1,x2,y2 = box
    points1 = np.array([(0,0),(emoji.shape[1],0),(emoji.shape[1],emoji.shape[0]),(0,emoji.shape[0])], dtype=np.float32)
    points2 = np.array([(x1,y1),(x2,y1),(x2,y2),(x1,y2)], dtype=np.float32)
    H = cv2.getPerspectiveTransform(points1, points2)
    emoji = cv2.warpPerspective(emoji, H, (frame.shape[1], frame.shape[0]))
    mask = (emoji[:, :, 3] == 0)
    mask = ~np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    frame = np.where(mask, emoji[:, :, :3], frame)
    return frame

def pred(frame, model, box, emoji):
    labels = ['Angry', 'Happy', 'Sad', 'Surprised', 'Neutral']
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = prep(frame, box)
    x1,y1,x2,y2 = box
    pred = model.predict(face, verbose = 0)
    pred = np.argmax(pred, axis=1)[0]
    frame = cv2.rectangle(frame, (x1,y1), (x2,y2), color=(255,255,255), thickness=1)
    frame = cv2.putText(frame, labels[pred], ((x1 + x2)// 2, y1), fontFace=1, fontScale=1, color=(255,255,255))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if emoji == True:
        return draw_emoji(frame, box, pred)
    return frame





yolo = YoloDetector(device="cpu", min_face=20)
model = tf.keras.models.load_model('./output/face_classification_model.keras')
cap = cv2.VideoCapture(0)
emoji = False
while True:
    ret, frame = cap.read()
    if ret == False: # end of video (perhaps)
        continue
    bboxes, points = yolo.predict(frame)
    for box in bboxes[0]:
        frame = pred(frame, model, box, emoji)
    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('e'):
        emoji = not emoji
cap.release()
cv2.destroyAllWindows()