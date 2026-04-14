import cv2
import mediapipe as mp
import numpy as np
import csv

CSV_FILE = 'gesture_data.csv'
GESTURES = {'1': 'go', '2': 'stop', '3': 'left', '4': 'right'}
NUM_SAMPLES = 500 

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    if not landmarks:
        return np.array([])
    
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # translate relative to wrist
    wrist_point = keypoints[0]
    translated_keypoints = keypoints - wrist_point
    
    # scale
    index_mcp_point = keypoints[5]
    scale_factor = np.linalg.norm(wrist_point - index_mcp_point)
    
    if scale_factor == 0:
        return np.array([0] * 63)
    
    scaled_keypoints = translated_keypoints / scale_factor
    
    return scaled_keypoints.flatten()

# collect data for each gesture
def collect_data():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    header = ['label'] + [f'L{i}_{c}' for i in range(21) for c in ['x', 'y', 'z']]
    
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            
            for key, gesture_name in GESTURES.items():
                print(f"\n--- Get ready for '{gesture_name}' ({key}). Press ENTER to start. ---")
                input()

                count = 0
                while count < NUM_SAMPLES:
                    ret, frame = cap.read()
                    if not ret: break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # record gesture datapoint
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            normalized_features = normalize_landmarks(hand_landmarks.landmark)
                            
                            if normalized_features.size > 0:
                                row = [gesture_name] + normalized_features.tolist()
                                writer.writerow(row)
                                count += 1

                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            cv2.putText(image, f"Collecting: {gesture_name} ({count}/{NUM_SAMPLES})", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                    cv2.imshow('Data Collection', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()
            print("\nData collection complete.")

if __name__ == "__main__":
    collect_data()