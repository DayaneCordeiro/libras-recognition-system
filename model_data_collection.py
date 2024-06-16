import cv2
import numpy as np
import os
import mediapipe as mp

# Inicializando as variáveis das libs do mediapipe que vão ajudar no mapeamento da face e da mão
mp_holistic = mp.solutions.holistic  # Faz as detecções
mp_drawing = mp.solutions.drawing_utils  # faz os desenhos dos pontos


# Função que faz a detecção dos pontos de interesse utilizando as libs do mediapipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB
    image.flags.writeable = False  # A imagem ainda não é gravável

    results = model.process(image)  # Faz a predição

    image.flags.writeable = True  # A imagem passa a ser gravável
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte a imagem de RGB para BGR

    return image, results


# Desenha os pontos-chave da mão direita
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# Extração dos pontos-chave
def extract_key_points(results):
    return np.array([[res.x, res.y, res.z] for res in
                     results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)


# Definição de variáveis que irão coletar os frames para treinar a rede ============================================
DATA_PATH = os.path.join('dataset')

# Ações e gestos que serão detectados
classes = np.array(['a'])

# Número de imagens para cada classe
number_of_images = 400

# cria no sistema operacional uma pasta para cada classe
for libras_class in classes:
    for image_number in range(number_of_images):
        try:
            os.makedirs(os.path.join(DATA_PATH, libras_class))
        except:
            pass

# Acessando a webcam
cap = cv2.VideoCapture(0)

# Define o modelo do mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Percorre as ações definidas
    for libras_class in classes:
        # Coleta os pontos de interesse da mão direita definidos para cada classe
        for image_number in range(number_of_images):
            ret, frame = cap.read()

            # Faz a detecção
            image, results = mediapipe_detection(frame, holistic)

            # Desenhando os pontos de referência
            draw_landmarks(image, results)

            # Escrita na tela
            if number_of_images == 0:
                cv2.putText(image, 'INICIANDO A COLETA DE DADOS', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(image, 'Coletando imagens para a classe {} - imagem {}/{}'.format(libras_class,
                                                                                              image_number,
                                                                                              number_of_images),
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Coletando imagens para a classe {} - imagem {}/{}'.format(libras_class,
                                                                                            image_number,
                                                                                            number_of_images),
                            (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Reconhecimento de LIBRAS com LSTM Deep Learning', image)

            # Exportando pontos-chave
            key_points = extract_key_points(results)
            npy_path = os.path.join(DATA_PATH, libras_class, str(image_number))
            np.save(npy_path, key_points)

            if cv2.waitKey(10) & 0xff == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()