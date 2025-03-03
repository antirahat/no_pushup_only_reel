#No pushup only reel - change your next reel with a simple eye gaze
#opencv tracking eye movement to generate a up and down key stroke 


#import libraries
import cv2 
import numpy as np
import time as Time 
from win32api import keybd_event
import mediapipe as mp

#keycode for up and down arrow keys
vk_down = 0x28
vk_up = 0x26

#mediapipe facemash
mp_face_mesh = mp.solutions.face_mesh

# Left eye indices list
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# Right eye indices list
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
#iris values
LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

#caputre video
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points=np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark]
            )
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)
            
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key ==ord('q'):
            break
#getting width and height or frame
cap.release()
cv2.destroyAllWindows()
