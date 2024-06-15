import os.path

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['ola', 'obrigado', 'euteamo', 'a'])
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