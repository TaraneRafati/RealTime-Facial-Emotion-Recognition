import cv2
import numpy as np
import csv
from face_detector import YoloDetector
from matplotlib import pyplot as plt

num_of_pics_before = [0] * 7
num_of_pics_after = [0] * 7
axis = [0, 1, 2, 3, 4, 5, 6]
with open('dataset.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    model = YoloDetector(device="cpu", min_face=20)
    for line in csv_reader:
        img = np.fromstring(line[1], np.uint8, sep=' ').reshape((48,48))
        num_of_pics_before[int(line[0])] += 1
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        bboxes, points = model.predict(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
if len(bboxes[0]) == 1:
    num_of_pics_after[int(line[0])] += 1
    x1,y1,x2,y2 = bboxes[0][0]
    cropped_image = img[y1:y2, x1:x2]
    cv2.imwrite(f'./images/{line[0]}/{csv_reader.line_num}.jpg', cropped_image)
# plot
plt.figure()
plt.bar(axis, num_of_pics_before)
plt.title('Number of pictures before face detection and crop')
plt.savefig('before.jpg')

plt.figure()
plt.bar(axis, num_of_pics_after)
plt.title('Number of pictures after face detection and crop')
plt.savefig('after.jpg')

plt.figure()
plt.bar(['Before', 'After'], [sum(num_of_pics_before), sum(num_of_pics_after)])
plt.title('Total number of pictures')
plt.savefig('total.jpg')

print(f'total before:{sum(num_of_pics_before)}\ntotal after:{sum(num_of_pics_after)}')