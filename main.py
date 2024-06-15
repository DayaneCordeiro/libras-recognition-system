# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
import cv2
# import mediapipe as mp
from keras.models import load_model
import numpy as np

# import time
#
#
model = load_model("model.h5")
#
# mp_hands = mp.solutions.hands
# mediapipeHands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils
#
# classes = {
#     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
#     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'
# }
#
# mp_drawing_styles = mp.solutions.drawing_styles
#
# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)
#
#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_draw.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing_styles.get_default_hand_landmarks_style(),
#             mp_drawing_styles.get_default_hand_connections_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()

# Mapeamento de índices para letras
class_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
               9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S',
               18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'}

# Inicialização da câmera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Pré-processamento da imagem da câmera
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)

    # Predição
    predictions = model.predict(img)
    class_idx = np.argmax(predictions[0])
    class_name = class_names[class_idx]

    # Exibição do resultado
    cv2.putText(frame, class_name, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-time ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('A'):
        break

cap.release()
cv2.destroyAllWindows()
