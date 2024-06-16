import os.path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard, Callback
import numpy as np

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['obrigado', 'euteamo', 'a'])
number_of_sequencies = 30
sequence_length = 30


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
label_map = {label: num for num, label in enumerate(actions)}

# sequences - dados de recursos
# labels - rótulos
sequences, labels = [], []

for action in actions:
    for sequence in range(number_of_sequencies):
        window = []

        # concatena os dados coletados em um unico array, formando assim os vídeos de ações
        for frame_number in range(sequence_length):
            result = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_number)))
            window.append(result)

        sequences.append(window)
        labels.append(label_map[action])

x = np.array(sequences)
y = to_categorical(labels).astype(int)

# Dividindo a base entre treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Obtenção de logs para monitorar o treinamento da rede
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Construção arquitetura da rede neural
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compilando o modelo
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_by_accuracy = EarlyStoppingByAccuracy(monitor='accuracy', value=0.90, verbose=1)


# Treinando o modelo
model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=2000,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping_by_accuracy, tb_callback]
)

model.summary()

result = model.predict(x_test)
print("Predito: ", actions[np.argmax(result[0])])
print("Encontrado pela rede: ", actions[np.argmax(y_test[0])])

model.save('model.h5')
