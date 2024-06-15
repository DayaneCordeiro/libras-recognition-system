from keras.models import Sequential
from keras.layers import Conv2D, Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.src.layers import BatchNormalization, MaxPool2D, Dropout
from keras.callbacks import EarlyStopping

IMAGES_WIDTH = 224
IMAGES_HEIGHT = 224

# Preparação das imagens =================
# Inserindo os parâmetros que serão utilizados no data augumentation
trainDataAugumentation = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    vertical_flip=False,
    validation_split=0.25
)

testDataAugumentation = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.05
)

# Executando o data augumentation na base de dados
trainImages = trainDataAugumentation.flow_from_directory(
    "dataset/train",
    target_size=(IMAGES_WIDTH, IMAGES_HEIGHT),
    color_mode='rgb',
    batch_size=16,
    shuffle=True,
    class_mode="categorical"
)

testImages = trainDataAugumentation.flow_from_directory(
    "dataset/test",
    target_size=(IMAGES_WIDTH, IMAGES_HEIGHT),
    color_mode='rgb',
    batch_size=16,
    shuffle=True,
    class_mode="categorical"
)

classes = list(trainImages.class_indices.keys())

# Implementando o modelo

early_stopping_monitor = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
inputShape = (IMAGES_WIDTH, IMAGES_HEIGHT, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu', input_shape=inputShape))
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
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=22, activation='softmax'))

# Otimizando o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Treinando o modelo
history = model.fit(
    trainImages,
    steps_per_epoch=trainImages.n // trainImages.batch_size,
    epochs=10,
    validation_data=testImages,
    validation_steps=testImages.n // testImages.batch_size,
    verbose=2,
    callbacks=[early_stopping_monitor]
)
# Salvando o modelo
model.save("model.h5")
