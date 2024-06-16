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


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# Extração dos pontos-chave
def extract_keypoints(results):
    face = np.array([[result.x, result.y, result.z, result.visibility]
                     for result in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(33 * 4)

    pose = np.array([[result.x, result.y, result.z, result.visibility]
                     for result in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(468 * 3)

    left_hand = np.array([[result.x, result.y, result.z, result.visibility]
                          for result in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)

    right_hand = np.array([[result.x, result.y, result.z, result.visibility]
                           for result in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, face, left_hand, right_hand])


# Definição de variáveis que irão coletar os frames para treinar a rede ============================================
DATA_PATH = os.path.join('MP_Data')

# Ações e gestos que serão detectados
actions = np.array(['ola', 'obrigado', 'euteamo', 'a'])

number_of_sequencies = 30

# os vídeos terão como tamanho 30 frames
sequence_length = 30

# cria no sistema operacional uma pasta para cada classe contendo 30 pastas dentro com os frames coletados
for action in actions:
    for sequence in range(number_of_sequencies):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Acessando a webcam
cap = cv2.VideoCapture(0)

# Define o modelo do mediapipe
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Percorre as ações definidas
    for action in actions:
        # Executa 30 vezes para obter 30 vídeos de cada ação
        for sequence in range(number_of_sequencies):
            # Executa 30 vezes para obter 30 frames dos pontos-chave que formarão os vídeos
            for frame_number in range(sequence_length):
                ret, frame = cap.read()

                # Faz a detecção
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Desenhando os pontos de referência
                draw_landmarks(image, results)

                # Lógica que aguarda para salvar os frames
                if frame_number == 0:
                    cv2.putText(image, 'Iniciando a coleta de dados', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Coletando frames para a classe {} - video {}/30'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Reconhecimento de LIBRAS com LSTM Deep Learning', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Coletando frames para a classe {} - video {}/30'.format(action, sequence), (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Reconhecimento de LIBRAS com LSTM Deep Learning', image)

                # Exportando pontos-chave
                key_points = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_number))
                np.save(npy_path, key_points)

                if cv2.waitKey(10) & 0xff == ord('q'):
                    break

cap.release()
cv2.destroyAllWindows()
