import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
PHI = 1.618
VALID_EXT = (".jpg", ".jpeg", ".png")

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

def draw_lines(frame, landmarks, w, h, x_off=0, y_off=0):
    fh, fw = frame.shape[:2]

    def pt(idx):
        x, y = get_pt(landmarks, idx, w, h)
        return (int(x) - x_off, int(y) - y_off)

    def clamp_x(x): return max(0, min(x, fw))
    def clamp_y(y): return max(0, min(y, fh))

    p_fh, p_ch = pt(10), pt(152)
    mid_x = (p_fh[0] + p_ch[0]) // 2
    cv2.line(frame, (clamp_x(mid_x), clamp_y(p_fh[1])), (clamp_x(mid_x), clamp_y(p_ch[1])), (0, 200, 255), 1)

    p_fl, p_fr = pt(234), pt(454)
    mid_y = (p_fl[1] + p_fr[1]) // 2
    cv2.line(frame, (clamp_x(p_fl[0]), clamp_y(mid_y)), (clamp_x(p_fr[0]), clamp_y(mid_y)), (0, 200, 255), 1)

    p_le, p_re = pt(33), pt(263)
    mid_y = (p_le[1] + p_re[1]) // 2
    cv2.line(frame, (clamp_x(p_le[0]), clamp_y(mid_y)), (clamp_x(p_re[0]), clamp_y(mid_y)), (255, 100, 0), 1)

    p_ml, p_mr = pt(61), pt(291)
    mid_y = (p_ml[1] + p_mr[1]) // 2
    cv2.line(frame, (clamp_x(p_ml[0]), clamp_y(mid_y)), (clamp_x(p_mr[0]), clamp_y(mid_y)), (0, 100, 255), 1)

    p_nl, p_nr = pt(129), pt(358)
    mid_y = (p_nl[1] + p_nr[1]) // 2
    cv2.line(frame, (clamp_x(p_nl[0]), clamp_y(mid_y)), (clamp_x(p_nr[0]), clamp_y(mid_y)), (0, 255, 100), 1)
            
def normalize(img, target_w=600):
    h, w = img.shape[:2]
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)))

def crop_face(img, landmarks, w, h, pad=30):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1 = max(0, int(min(xs)) - pad)
    y1 = max(0, int(min(ys)) - pad)
    x2 = min(w, int(max(xs)) + pad)
    y2 = min(h, int(max(ys)) + pad)
    return img[y1:y2, x1:x2].copy(), x1, y1

folder_dir = "images"
best_img   = None
best_score = -1
best_name  = ''
worst_img  = None
worst_score = float('inf')
worst_name  = ''

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    for filename in os.listdir(folder_dir):
        if not filename.lower().endswith(VALID_EXT):
            continue

        filepath = os.path.join(folder_dir, filename)
        if not os.path.isfile(filepath):
            continue

        img = cv2.imread(filepath)
        if img is None:
            continue

        img = normalize(img)
        h, w, _ = img.shape

        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            continue

        for face_landmarks in result.multi_face_landmarks:
            ratios, scores, overall = calc_ratios(face_landmarks.landmark, w, h)

            face, x_off, y_off = crop_face(img, face_landmarks.landmark, w, h)
            fh, fw, _ = face.shape
            draw_lines(face, face_landmarks.landmark, w, h, x_off, y_off)

            print(f"{filename} — {w}x{h} — score: {overall:.1f}%")

            if overall > best_score:
                best_score = overall
                best_img   = face.copy()
                best_name  = filename

            if overall < worst_score:
                worst_score = overall
                worst_img   = face.copy()
                worst_name  = filename

if best_img is not None:
    cv2.putText(best_img, f"Best: {best_name} ({best_score:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Best Golden Ratio", best_img)
    print("Best:", best_name)

if worst_img is not None:
    cv2.putText(worst_img, f"Worst: {worst_name} ({worst_score:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Worst Golden Ratio", worst_img)
    print("Worst:", worst_name)



# filepath = "images/220201019.jpg"

# img = cv2.imread(filepath)
# img = normalize(img)
# h, w, _ = img.shape

# with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
#     result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     if not result.multi_face_landmarks:
#         print("No face detected")
#     else:
#         for face_landmarks in result.multi_face_landmarks:
#             ratios, scores, overall = calc_ratios(face_landmarks.landmark, w, h)
#             face, x_off, y_off = crop_face(img, face_landmarks.landmark, w, h)
#             draw_lines(face, face_landmarks.landmark, w, h, x_off, y_off)

#             print(f"Score: {overall:.1f}%")
#             for name, score in scores.items():
#                 print(f"  {name}: {ratios[name]:.3f} ({score:.1f}%)")

#             cv2.imshow("Result", face)

cv2.waitKey(0)
cv2.destroyAllWindows()