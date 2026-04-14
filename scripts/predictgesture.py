import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import serial

MODEL_FILE = "gesture_classifier.pkl"
COM_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

robot_serial = None

def normalize_landmarks(landmarks):
    if not landmarks:
        return np.array([])
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = keypoints[0]
    translated = keypoints - wrist
    index_mcp = keypoints[5]
    scale = np.linalg.norm(wrist - index_mcp)
    if scale == 0:
        return np.zeros(63)
    return (translated / scale).flatten()

def predict_gesture_label(hand_landmarks, model_package):
    model = model_package["model"]
    scaler = model_package["scaler"]
    normalized = normalize_landmarks(hand_landmarks.landmark)
    if normalized.size == 0:
        return "No Hand Detected"
    scaled = scaler.transform(normalized.reshape(1, -1))
    return model.predict(scaled)[0]

# initializes serial communication link between script and robot hardware
def init_serial():
    global robot_serial
    try:
        robot_serial = serial.Serial(COM_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"Serial connected on {COM_PORT}")
    except Exception as e:
        print(f"Serial unavailable. Robot disabled. {e}")
        robot_serial = None

# decodes command and sends it over the serial communication link
def send_robot_command(gesture_label):
    command_map = {
        "go": "F",
        "stop": "S",
        "left": "L",
        "right": "R",
    }
    cmd = command_map.get(gesture_label, "S")
    if robot_serial:
        try:
            robot_serial.write(f"{cmd}\n".encode())
            print(f"--> SENT ROBOT: {cmd} (gesture={gesture_label})")
        except Exception as e:
            print(f"Serial write failed: {e}")
    else:
        print(f"[TEST] Gesture={gesture_label} Command={cmd}")

def run_realtime_control():
    # Load model
    try:
        with open(MODEL_FILE, "rb") as f:
            model_package = pickle.load(f)
            print(
                "Model loaded. CV Mean Accuracy:",
                model_package["cv_mean_accuracy"],
            )
    except FileNotFoundError:
        print(f"Missing model file: {MODEL_FILE}")
        return
    init_serial()

    # Qualcomm camera via GStreamer
    gst_pipeline = (
        "qtiqmmfsrc camera=0 ! "
        "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=1 sync=false"
    )
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: Could not open camera through GStreamer pipeline.")
        return
    
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Frame grab failed")
                    continue
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                gesture_label = "Waiting..."

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture_label = predict_gesture_label(
                            hand_landmarks, model_package
                        )
                        send_robot_command(gesture_label)

                print("Gesture:", gesture_label)
        
        except KeyboardInterrupt:
            print("Stopping gesture control...")
    
    cap.release()
    if robot_serial:
        robot_serial.close()
        print("Serial port closed.")

if __name__ == "__main__":
    run_realtime_control()
