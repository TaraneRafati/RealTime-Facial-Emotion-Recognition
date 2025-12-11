import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = tf.keras.models.load_model('./output/face_classification_model.keras')
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_dataset = test_datagen.flow_from_directory(
    './dataset/test',
    target_size=(48, 48),
    color_mode='grayscale',
    batch_size=1,
    class_mode='sparse',
    shuffle=False
)

true_labels = test_dataset.classes
pred_probs = model.predict(test_dataset)
pred_labels = tf.argmax(pred_probs, axis=1)

acc = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='weighted')
recall = recall_score(true_labels, pred_labels, average='weighted')
f1 = f1_score(true_labels, pred_labels, average='weighted')
conf = confusion_matrix(true_labels, pred_labels, normalize='true')

metrics = {'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
plt.figure(figsize=(10, 5))
bars = plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange', 'red'])
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.ylim([0, 1])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', ha='center', va='bottom')
plt.savefig('metrics.png')
plt.show()

labels = ['Anger', 'Happiness', 'Sadness', 'Surprise', 'Neutrality']
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='.3f', cmap='Blues', cbar=False,
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()