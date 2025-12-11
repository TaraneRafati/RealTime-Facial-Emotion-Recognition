from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_dataset = train_datagen.flow_from_directory(
    './dataset/train',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse'
)
validation_dataset = validation_datagen.flow_from_directory(
    './dataset/train',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse'
)

print(train_dataset)

model = Sequential([
    Conv2D(64, (3, 3),padding = 'same', input_shape=(48, 48, 1), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3),padding = 'same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3),padding = 'same', activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3, 3),padding = 'same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(256, (3, 3),padding = 'same', activation='relu'),
    BatchNormalization(),
    Conv2D(256, (3, 3),padding = 'same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    Flatten(),
    
    Dense(1024, activation='relu'),

    Dropout(0.2),
    
    Dense(5, activation='softmax')
])

model.compile(
    optimizer= Adam(learning_rate= 0.001),  
    loss= SparseCategoricalCrossentropy(),  
    metrics=['accuracy']  
)

model.summary()

history = model.fit(
    train_dataset,
    batch_size=32,
    epochs=15,
)


test_generator = test_datagen.flow_from_directory(
    './dataset/train',
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='sparse'

)

test_loss, test_acc = model.evaluate(test_generator, batch_size=32)
print(f"Test accuracy: {test_acc} test loss: {test_loss}")
model.save('./output/face_classification_model.keras')