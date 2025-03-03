# Eyelids segmentation using MediaPipe with blink detection

# Import libraries
import cv2
import numpy as np
import mediapipe as mp
from utils import fillPolyTrans
import win32api
import win32con
import time

# MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Virtual key codes for keyboard simulation
VK_DOWN = 0x28  # Down arrow key
VK_UP = 0x26    # Up arrow key
# VK_LEFT = 0x25  # Left arrow key
# VK_RIGHT = 0x27 # Right arrow key
# VK_SPACE = 0x20 # Space key

# Eyelid indices
LEFT_EYE_UPPER = [386, 374, 373, 390, 388, 387, 386]
LEFT_EYE_LOWER = [382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386]
RIGHT_EYE_UPPER = [159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_LOWER = [145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173]

# Blink detection parameters
BLINK_CLOSURE_THRESHOLD = 0.5  # 60% eye closure threshold
BLINK_TIME_THRESHOLD = 1.0  # Time window for double blink detection
COOLDOWN_TIME = 2.0  # Cooldown period after triggering an event

# Variables to track maximum eye opening
max_left_opening = 0
max_right_opening = 0
CALIBRATION_FRAMES = 30  # Number of frames to calibrate maximum eye opening

# Capture video
cap = cv2.VideoCapture(0)

def calculate_eye_closure_percentage(eye_upper, eye_lower, max_opening):
    # Calculate the current eye opening
    vertical_distances = [abs(upper[1] - lower[1]) for upper, lower in zip(eye_upper, eye_lower)]
    current_opening = np.mean(vertical_distances)
    
    # Update maximum eye opening if current opening is larger
    if current_opening > max_opening:
        max_opening = current_opening
    
    # Calculate closure percentage (1 - current_ratio)
    if max_opening > 0:  # Avoid division by zero
        closure_percentage = 1.0 - (current_opening / max_opening)
    else:
        closure_percentage = 0.0
        
    return closure_percentage, max_opening

def simulate_keypress(vk_code):
    # Helper function to simulate a key press and release
    win32api.keybd_event(vk_code, 0, 0, 0)  # Press key
    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)  # Release key

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    # Initialize variables for blink detection
    last_blink_time = time.time()
    blink_counter = 0
    last_event_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for selfie view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # Detect facial landmarks
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]
            )

            # Extract eyelid points
            left_upper = mesh_points[LEFT_EYE_UPPER]
            left_lower = mesh_points[LEFT_EYE_LOWER]
            right_upper = mesh_points[RIGHT_EYE_UPPER]
            right_lower = mesh_points[RIGHT_EYE_LOWER]

            # Calculate eye closure percentages
            left_closure, max_left_opening = calculate_eye_closure_percentage(left_upper, left_lower, max_left_opening)
            right_closure, max_right_opening = calculate_eye_closure_percentage(right_upper, right_lower, max_right_opening)

            # Detect blink when either eye is closed beyond threshold
            current_time = time.time()
            if left_closure > BLINK_CLOSURE_THRESHOLD or right_closure > BLINK_CLOSURE_THRESHOLD:
                if current_time - last_blink_time < BLINK_TIME_THRESHOLD:
                    blink_counter += 1
                else:
                    blink_counter = 1
                last_blink_time = current_time

                # Process blink events after cooldown period
                if current_time - last_event_time > COOLDOWN_TIME:
                    if blink_counter == 1:
                        simulate_keypress(VK_DOWN)  # Simulate down arrow key press
                        last_event_time = current_time
                    elif blink_counter == 2:
                        simulate_keypress(VK_UP)  # Simulate up arrow key press
                        last_event_time = current_time
                        blink_counter = 0

            # Reset blink counter if too much time has passed
            if current_time - last_blink_time > BLINK_TIME_THRESHOLD:
                blink_counter = 0

            # Draw filled contours for eyelids with transparency
            frame = fillPolyTrans(frame, left_upper.tolist(), (255, 192, 203), 0.5)  # Pink color
            frame = fillPolyTrans(frame, left_lower.tolist(), (255, 192, 203), 0.5)
            frame = fillPolyTrans(frame, right_upper.tolist(), (255, 192, 203), 0.5)
            frame = fillPolyTrans(frame, right_lower.tolist(), (255, 192, 203), 0.5)

            # Draw contour lines for better visibility
            cv2.polylines(frame, [left_upper], True, (0, 0, 0), 1)
            cv2.polylines(frame, [left_lower], True, (0, 0, 0), 1)
            cv2.polylines(frame, [right_upper], True, (0, 0, 0), 1)
            cv2.polylines(frame, [right_lower], True, (0, 0, 0), 1)

            # Display blink counter and EAR
            cv2.putText(frame, f'Blinks: {blink_counter}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Left Closure: {left_closure:.2%}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Right Closure: {right_closure:.2%}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Blink Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
