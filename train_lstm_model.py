import os.path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['ola', 'obrigado', 'euteamo', 'a', 'b'])
number_of_sequencies = 30
sequence_length = 30

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
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)

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

# Treinando o modelo
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])

model.summary()

result = model.predict(x_test)
print("Predito: ", actions[np.argmax(result[0])])
print("Encontrado pela rede: ", actions[np.argmax(y_test[0])])

model.save('model.h5')
