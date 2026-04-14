# Qualcomm-Gesture-Controlled-Robot

## 1st Place Winner - Qualcomm Career Catalyst Mini-Hackathon

## • Description
This project implements a low-latency, real-time control system that translates hand gestures into physical robot movements. By utilizing a distributed Edge AI architecture, the system captures a camera feed from a Rubik Pi, performs inference on a local machine, and transmits movement commands to an Arduino-powered robot.

## • Demonstration
[![video_robot_logo](https://github.com/user-attachments/assets/d34e1fc6-8deb-4866-b1cd-a35ac79291a7)](https://youtu.be/Ewj8m4dt_m8)

## • Pipeline

1) **Data Collection**: A custom dataset of 2000 hand gesture images was created for training, covering various angles and orientations for robustness.

   • Gestures:
   a) Closed Fist: "Go Forward"
   b) Open Palm: "Stop"
   c) Finger Pointing Right: "Turn Right"
   d) Finger Pointing Left: "Turn Left"

Used Google's MediaPipe Hands to extract (x, y, z) coordinates for the 21 landmarks of a hand, resulting in 63 distinct features per gesture image.
   





