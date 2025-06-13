import cv2
import mediapipe as mp
import pygame
import os
import math
import time

# Initialize pygame mixer
pygame.mixer.init()

# Load sounds
note_files = {
    'C': 'C.wav',
    'D': 'D.wav',
    'E': 'E.wav',
    'F': 'F.wav',
    'G': 'G.wav',
    'A': 'A.wav',
    'B': 'B.wav'
}

notes = {}
for key, file in note_files.items():
    path = os.path.join('sounds', file)
    notes[key] = pygame.mixer.Sound(path)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Define piano key layout
key_labels = list(note_files.keys())
pressed_key = None
last_play_time = 0
cooldown = 0.5  # seconds

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def draw_keyboard(frame, key_areas, active_key=None):
    for key, (x1, y1, x2, y2) in key_areas.items():
        color = (0, 255, 255) if key == active_key else (255, 255, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        cv2.putText(frame, key, (x1 + 15, y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    key_width = w // len(key_labels)
    key_height = 100
    key_y_start = h - key_height

    # Define virtual keys
    key_areas = {
        label: (i * key_width, key_y_start, (i + 1) * key_width, h)
        for i, label in enumerate(key_labels)
    }

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_key = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index tip and thumb tip
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Visual feedback
            cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 8, (0, 0, 255), -1)

            # Check for pinch
            dist = distance((x1, y1), (x2, y2))
            if dist < 40:
                # Find which key is touched
                for key, (kx1, ky1, kx2, ky2) in key_areas.items():
                    if kx1 <= x1 <= kx2 and ky1 <= y1 <= ky2:
                        current_key = key
                        if time.time() - last_play_time > cooldown:
                            notes[key].play()
                            last_play_time = time.time()
                        break

    # Draw virtual piano keyboard
    draw_keyboard(frame, key_areas, current_key)

    # Show result
    cv2.imshow("Advanced Gesture Piano", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
