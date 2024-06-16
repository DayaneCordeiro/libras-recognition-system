import numpy as np
import os

from keras.src.callbacks import Callback, TensorBoard
from keras.src.layers import LSTM
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATA_PATH = os.path.join('dataset')
classes = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o'])
number_of_images = 400


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.90, verbose=1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print(f"Monitor '{self.monitor}' is not available. Available metrics are: {', '.join(logs.keys())}")
            return

        if current >= self.value:
            if self.verbose > 0:
                print(
                    f"Epoch {epoch + 1}: early stopping as {self.monitor} reached {current:.4f} (target was {self.value:.4f})")
            self.model.stop_training = True


# Dicionário com as ações utilizadas no sistema
dictionary = {label: num for num, label in enumerate(classes)}

# sequences - dados de recursos
# labels - rótulos
images, labels = [], []

for libras_class in classes:
    for image_number in range(number_of_images):
        result = np.load(os.path.join(DATA_PATH, libras_class, "{}.npy".format(image_number)))

        images.append(result)
        labels.append(dictionary[libras_class])

x = np.array(images)
y = to_categorical(labels).astype(int)

# Dividindo a base entre treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Obtenção de logs para monitorar o treinamento da rede
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

print(x_train.shape)
print(y_train.shape)

x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

# Construção arquitetura da rede neural
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(1,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(classes.shape[0], activation='softmax'))

# Compilando o modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_by_accuracy = EarlyStoppingByAccuracy(monitor='accuracy', value=0.90, verbose=1)

# Treinando o modelo
model.fit(
    x_train,
    y_train,
    epochs=2000,
    callbacks=[early_stopping_by_accuracy, tb_callback]
)

model.summary()

result = model.predict(x_test)
print("Predito: ", classes[np.argmax(result[0])])
print("Encontrado pela rede: ", classes[np.argmax(y_test[0])])

model.save('model.h5')
