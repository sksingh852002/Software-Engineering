import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe hands and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
drawing = False  # Flag to check if drawing is active
last_point = None

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flip and convert the image for MediaPipe processing
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Revert image to BGR for OpenCV processing and display
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get fingertip coordinates for index, middle, and other fingers
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                h, w, _ = image.shape
                ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                mx, my = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                
                # Check if each finger is up (y-coordinate of tips should be lower than other points)
                index_finger_up = index_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
                middle_finger_up = middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
                ring_finger_up = ring_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y
                pinky_up = pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y
                thumb_up = thumb_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
                
                if index_finger_up and middle_finger_up and ring_finger_up and pinky_up and thumb_up:
                    # Erase (rub) with a circle of 25 px radius
                    cv2.circle(canvas, (ix, iy), 25, (0, 0, 0), -1)
                    drawing = False
                elif index_finger_up and not middle_finger_up:
                    drawing = True
                elif index_finger_up and middle_finger_up:
                    drawing = False
                
                if drawing:
                    if last_point:
                        cv2.line(canvas, last_point, (ix, iy), (255, 255, 255), 5)
                    last_point = (ix, iy)
                else:
                    last_point = None

        # Combine canvas and camera image
        combined_image = cv2.addWeighted(image, 0.5, canvas, 0.5, 0)
        
        # Display information about controls
        cv2.putText(combined_image, 'Index Finger: Draw', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined_image, 'Index + Middle Finger: Pause', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined_image, 'All Fingers: Erase', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(combined_image, "'c': Clear Canvas", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the combined image
        cv2.imshow('Draw with Hand', combined_image)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break  # Quit the program when 'q' key is pressed
        elif key == ord('c'):
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # Clear the canvas

cap.release()
cv2.destroyAllWindows()
