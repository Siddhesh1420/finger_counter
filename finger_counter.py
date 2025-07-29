import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands=mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
)
cap= cv2.VideoCapture(0)
tip_ids = [4, 8, 12, 16, 20]
def count_fingers(image, hand_landmarks, hand_label):
    if hand_landmarks:
        landmarks= hand_landmarks.landmark
        fingers = []
        # Thumb
        if(hand_label== 'Right'):
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        # Other fingers
        for id in range(1, 5):
            if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        total_fingers = sum(fingers)
    return total_fingers
def is_palm_facing(hand_landmarks, hand_label):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    if hand_label == "Right":
        return thumb_tip.x < index_mcp.x
    else:
        return thumb_tip.x > index_mcp.x
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'
            palm_facing = is_palm_facing(hand_landmarks, hand_label)
            finger_count = count_fingers(frame,hand_landmarks, hand_label)

            info_text = f"Fingers: {finger_count}"

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow('Finger Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
