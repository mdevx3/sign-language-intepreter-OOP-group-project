# collect_landmarks.py
import cv2
import mediapipe as mp
import numpy as np
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Label input
label = input("Enter label (A-Z): ").upper()
save_dir = f"landmark_dataset/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot access the camera")
    exit()
img_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Flatten x, y, z

            if cv2.waitKey(1) & 0xFF == ord('s'):
                npy_path = os.path.join(save_dir, f"{label}_{img_count}.npy")
                np.save(npy_path, np.array(landmarks))
                print(f"Saved {npy_path}")
                img_count += 1

    cv2.putText(frame, f"Label: {label} | Count: {img_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collect Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
