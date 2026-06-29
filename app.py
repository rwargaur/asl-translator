import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from collections import deque

SIGNS = ["hello", "thanks", "yes", "no", "please"]
FRAMES = 30
MODEL_PATH = "hand_landmarker.task"

# Colors (BGR)
WHITE      = (255, 255, 255)
LIGHT_GREY = (240, 240, 240)
MID_GREY   = (200, 200, 200)
DARK_GREY  = (80, 80, 80)
BLACK      = (30, 30, 30)
ACCENT     = (100, 100, 100)
GREEN      = (120, 200, 120)

class ASLModel(nn.Module):
    def __init__(self, input_size=63, hidden=128, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ASLModel(num_classes=len(SIGNS))
model.load_state_dict(torch.load("model/lstm_model.pth"))
model.eval()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

sequence  = deque(maxlen=FRAMES)
prediction = ""
confidence = 0.0
history    = deque(maxlen=4)  # last 4 predictions

PANEL_W = 280
CAM_W, CAM_H = 640, 480

def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r),   0, 0, 90, color, thickness)

def draw_panel(canvas, prediction, confidence, history, hand_detected):
    px = CAM_W + 1
    # Panel background
    canvas[0:CAM_H, px:px+PANEL_W] = (245, 245, 245)

    # Title
    cv2.putText(canvas, "ASL", (px + 20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, BLACK, 2)
    cv2.putText(canvas, "Translator", (px + 20, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, DARK_GREY, 1)

    # Divider
    cv2.line(canvas, (px + 20, 88), (px + PANEL_W - 20, 88), MID_GREY, 1)

    # Hand status dot
    dot_color = GREEN if hand_detected else MID_GREY
    cv2.circle(canvas, (px + 30, 112), 7, dot_color, -1)
    status_text = "Hand detected" if hand_detected else "No hand"
    cv2.putText(canvas, status_text, (px + 46, 117),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DARK_GREY, 1)

    # Current prediction box
    draw_rounded_rect(canvas, px + 16, 135, px + PANEL_W - 16, 230, 10, WHITE)
    cv2.putText(canvas, "CURRENT SIGN", (px + 28, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, MID_GREY, 1)

    pred_display = prediction if prediction else "—"
    font_scale = 1.4 if len(pred_display) <= 6 else 0.9
    cv2.putText(canvas, pred_display, (px + 28, 210),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, BLACK, 2)

    # Confidence bar
    bar_x, bar_y = px + 16, 242
    bar_w = PANEL_W - 32
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), MID_GREY, -1)
    fill = int(bar_w * confidence)
    if fill > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + 6), DARK_GREY, -1)
    cv2.putText(canvas, f"{confidence:.0%} confidence", (bar_x, bar_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, DARK_GREY, 1)

    # Divider
    cv2.line(canvas, (px + 20, 278), (px + PANEL_W - 20, 278), MID_GREY, 1)

    # History
    cv2.putText(canvas, "RECENT", (px + 20, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, MID_GREY, 1)

    for i, h in enumerate(reversed(history)):
        alpha = 1.0 - i * 0.22
        grey_val = int(30 + i * 55)
        color = (grey_val, grey_val, grey_val)
        cv2.putText(canvas, h, (px + 20, 328 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    # Footer
    cv2.putText(canvas, "Press Q to quit", (px + 20, CAM_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, MID_GREY, 1)

cap = cv2.VideoCapture(0)
print("Starting ASL Translator — press Q to quit")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        canvas = np.ones((CAM_H, CAM_W + PANEL_W, 3), dtype=np.uint8) * 245

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        hand_detected = bool(result.hand_landmarks)

        if hand_detected:
            lm = result.hand_landmarks[0]
            keypoints = np.array([[p.x, p.y, p.z] for p in lm]).flatten()
            h, w, _ = frame.shape
            for point in lm:
                cx, cy = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (cx, cy), 3, DARK_GREY, -1)
        else:
            keypoints = np.zeros(63)

        sequence.append(keypoints)

        if len(sequence) == FRAMES:
            x = torch.tensor(np.array([list(sequence)]), dtype=torch.float32)
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                confidence = conf.item()
                if confidence > 0.85:
                    new_pred = SIGNS[pred_idx.item()]
                    if new_pred != prediction:
                        history.append(new_pred)
                    prediction = new_pred

        # Thin divider line between cam and panel
        cv2.line(frame, (CAM_W - 1, 0), (CAM_W - 1, CAM_H), MID_GREY, 1)

        canvas[:, :CAM_W] = frame
        draw_panel(canvas, prediction, confidence, history, hand_detected)

        cv2.imshow("ASL Translator", canvas)
        cv2.setWindowProperty("ASL Translator", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()