import cv2
import tensorflow as tf
from face_detector import YoloDetector
import glob
from os import path
from  webcam import pred

yolo = YoloDetector(device="cpu", min_face=20)
model = tf.keras.models.load_model('./output/face_classification_model.keras')
images = glob.glob('./dataset/Fam4a/*.jpg')
for filename in images:
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image_with_emoji = image.copy()
    image_without_emoji = image.copy()
    bboxes, points = yolo.predict(image)
    for box in bboxes[0]:
        image_with_emoji = pred(image_with_emoji, model, box, True)
        image_without_emoji = pred(image_without_emoji, model, box, False)
    cv2.imwrite('./output/Fam4a/with-emoji/' + path.basename(filename), image_with_emoji)
    cv2.imwrite('./output/Fam4a/without-emoji/' + path.basename(filename), image_without_emoji)
cv2.destroyAllWindows()