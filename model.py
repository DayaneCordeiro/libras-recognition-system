from keras.models import Sequential
from keras.layers import Conv2D, Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import BatchNormalization, MaxPool2D, Dropout

# Preparação das imagens
trainDataAugumentation = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False
)

testDataAugumentation = ImageDataGenerator(
    rescale=1. / 255
)

imagesWith = 224
imagesHeight = 224

trainImages = trainDataAugumentation.flow_from_directory(
    "dataset/train",
    target_size=(imagesWith, imagesHeight),
    batch_size=16,
    class_mode="categorical"
)

testImages = trainDataAugumentation.flow_from_directory(
    "dataset/test",
    target_size=(imagesWith, imagesHeight),
    batch_size=16,
    class_mode="categorical"
)

classes = list(trainImages.class_indices.keys())

# Implementando o modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=(imagesWith, imagesHeight, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(24, (3, 3), strides=1, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Conv2D(18, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=24, activation='softmax'))

# Otimizando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treinando o modelo
history = model.fit(
    trainImages, steps_per_epoch=128, epochs=20,
    validation_data=testImages, validation_steps=128
)

# Salvando o modelo
model.save("model.h5")
