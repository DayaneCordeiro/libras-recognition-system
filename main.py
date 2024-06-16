import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

sequence = []
sentence = []
threshold = 0.90

# Inicializando as variáveis das libs do mediapipe que vão ajudar no mapeamento da face e da mão
mp_holistic = mp.solutions.holistic  # Faz as detecções
mp_drawing = mp.solutions.drawing_utils  # faz os desenhos dos pont

model = load_model('model.h5')

actions = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'teamo'])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB
    image.flags.writeable = False  # A imagem ainda não é gravável

    results = model.process(image)  # Faz a predição

    image.flags.writeable = True  # A imagem passa a ser gravável
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Converte a imagem de RGB para BGR

    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


# Extração dos pontos-chave
def extract_keypoints(results):

    right_hand = np.array([[res.x, res.y, res.z] for res in
                                results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([right_hand])


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # draw_landmarks(image, results)

        # Lógica de predição
        key_points = extract_keypoints(results)
        sequence.append(key_points)
        sequence = sequence[-1:]

        if len(sequence) == 1:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            # Verificando se o resultado está acima do limite definido
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 3:
                sentence = sentence[-3:]

            # image = prob_viz(res, actions, image, colors)
            cv2.putText(image, actions[np.argmax(res)], (200, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (1, 1, 1), 2,
                        cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Reconhecimento de LIBRAS com LSTM Deep Learning', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
