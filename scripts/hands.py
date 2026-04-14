import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 500)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# verify hand landmark detection
while(True):
  success, frame = cap.read()
  if success:
    RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)
    if result.multi_hand_landmarks:
      for hand_landmarks in result.multi_hand_landmarks:
        print(hand_landmarks)
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv.imshow("capture image", frame)
    if cv.waitKey(1) == ord('q'):
      break