import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Parâmetros
batch_size = 32
epochs = 10

# Pré-processamento
train_datagen = ImageDataGenerator(rescale=1./255,
                                  rotation_range=15,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Arquitetura da Rede
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')
])

# Treinamento da Rede
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

history = model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epochs,
                    validation_data=test_generator, validation_steps=len(test_generator))

# Avaliação da Rede
test_loss, test_acc = model.evaluate(test_generator)
print('Acurácia na validação:', test_acc)

# Reconhecimento em Tempo Real
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0

    prediction = model.predict(frame[np.newaxis, ...])
    predicted_letter = np.argmax(prediction[0])

    cv2.put