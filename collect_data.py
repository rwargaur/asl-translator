import cv2
import numpy as np
import os
import mediapipe as mp

SIGNS = ["hello", "thanks", "yes", "no", "please"]
SEQUENCES = 30
FRAMES = 30
DATA_PATH = "data"

os.makedirs(DATA_PATH, exist_ok=True)
for sign in SIGNS:
    for seq in range(SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, sign, str(seq)), exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    for sign in SIGNS:
        for seq in range(SEQUENCES):

            # Wait for spacebar to start each sequence
            while True:
                ret, frame = cap.read()
                cv2.putText(frame, f"SPACE to record: '{sign}' seq {seq+1}/30", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

            # Record frames
            for frame_num in range(FRAMES):
                ret, frame = cap.read()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    keypoints = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
                else:
                    keypoints = np.zeros(63)

                np.save(os.path.join(DATA_PATH, sign, str(seq), str(frame_num)), keypoints)

                cv2.putText(frame, f"RECORDING: {sign} | {frame_num+1}/30", (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Collect", frame)
                cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()