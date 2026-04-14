<h1 align="center">Qualcomm-Gesture-Controlled-Robot 👋</h1>

## 1st Place Winner - Qualcomm Career Catalyst Mini-Hackathon

## • Description
This project implements a real-time control system that translates hand gestures into physical robot movements. By utilizing a distributed Edge AI architecture, the system captures a camera feed from a Rubik Pi, performs inference on a local machine, and transmits movement commands to an Arduino-powered robot.

## • Demonstration
![qualcommdemo (1)](https://github.com/user-attachments/assets/b75a557a-781b-4907-9d00-b35bbe8c72e0)  

Vist [this link](https://youtu.be/Ewj8m4dt_m8) for a higher quality demo video of the project, and watch the real-time detection and prediction of gestures.

## • Pipeline

1) **Data Collection**: A custom dataset of 2000 hand gesture images was created for training, covering various angles and orientations for robustness.

   Gestures:  
      a) Closed Fist: "Go Forward"  
      b) Open Palm: "Stop"  
      c) Finger Pointing Right: "Turn Right"  
      d) Finger Pointing Left: "Turn Left"  

   Used Google's MediaPipe Hands to extract (x, y, z) coordinates for the 21 landmarks of a hand, resulting in 63 distinct features per gesture image. This dataset is stored in ```data/gesture_data.csv```.

   ![landmarks](https://github.com/user-attachments/assets/58e43161-203f-4eed-903f-6bbd73fad8db)

   
3) **Model Training**: Used the custom dataset to train an SVM model, and utilized Stratified K-Fold Cross Validation to evaluate model performance. We achieved a cross-validation accuracy of (99 ± 0.2)%, attributed to the highly distinct spatial features of the chosen gestures. The trained model and its scaler are stored in ```model/gesture_classifier.pkl``` for real-time inference.

4) **Real-Time Gesture Inference**: This executes the following loop:
   
   a) **Frame Capture**: Reads a real-time video stream from the webcam mounted on the Rubik Pi, frame by frame.

   b) **Landmark Detection**: Processes each incoming frame via MediaPipe to identify the 21 (x, y, z) landmark coordinates. Normalization is performed to account for the position/orientation of the hand relative to the camera, and scaled for consistency.
   
   c) **Inference and Command Mapping**: The trained SVM model is loaded and the normalized landmarks are provided as input. The model predicts the corresponding gesture, which is mapped to a movement character (eg: "Stop" -> "S").
   
   d) **Feedback Loop**: This movement command character is transmitted back to the robot's Arduino via a serial bridge, and the corresponding motor functions of the robot are triggered to perform the action.  




