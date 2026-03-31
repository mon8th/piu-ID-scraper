import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

PHI = 1.618

def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_pt(landmarks, idx, w, h):
    lm = landmarks[idx]
    return (lm.x * w, lm.y * h)

def calc_ratios(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)

    left_eye      = get_pt(landmarks, 33,  w, h)
    right_eye     = get_pt(landmarks, 263, w, h)
    mouth_left    = get_pt(landmarks, 61,  w, h)
    mouth_right   = get_pt(landmarks, 291, w, h)
    nostril_left  = get_pt(landmarks, 129, w, h)
    nostril_right = get_pt(landmarks, 358, w, h)

    eye_width   = euclidean(left_eye, right_eye)
    mouth_width = euclidean(mouth_left, mouth_right)
    nose_width  = euclidean(nostril_left, nostril_right)

    ratios = {
        "face_h_w":   bbox_h / bbox_w         if bbox_w       else 0,
        "eye_mouth":  eye_width / mouth_width  if mouth_width  else 0,
        "nose_mouth": mouth_width / nose_width if nose_width   else 0,
    }
    scores = {k: max(0, 1 - abs(v - PHI) / PHI) * 100 for k, v in ratios.items()}
    overall = sum(scores.values()) / len(scores)
    return ratios, scores, overall


webcam = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh() as face_mesh:
    while True:
        ret, frame = webcam.read()
        if not ret:
            break 
        
        clean = frame.copy() 
        
        h, w, _ = frame.shape
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(color)
        
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0,255,0), -1)

            ratios, scores, overall = calc_ratios(face_landmarks.landmark, w, h)
            y = 30
            for name, score in scores.items():
                ratio_val = ratios[name]
                cv2.putText(frame, f"{name}: {ratio_val:.3f} ({score:.1f}%)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                y += 22
                
            cv2.putText(frame, f"Overall: {overall:.1f}%", (10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        cv2.imshow("output", frame)  
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
        
webcam.release()
cv2.destroyAllWindows()
