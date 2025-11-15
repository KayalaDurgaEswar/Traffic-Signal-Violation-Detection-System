import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# POLYGONS
# -------------------------------
RedLight = np.array([[998, 125],[998, 155],[972, 152],[970, 127]])
GreenLight = np.array([[971, 200],[996, 200],[1001, 228],[971, 230]])
ROI = np.array([[910, 372],[388, 365],[338, 428],[917, 441]])

# -------------------------------
# MODEL
# -------------------------------
model = YOLO("yolov8m.pt")
coco = model.names
TargetLabels = ["bicycle", "car", "motorcycle", "bus", "truck", "traffic light"]


# -----------------------------------------
# ROUNDED BOX (Modern UI Look)
# -----------------------------------------
def round_rect(img, pt1, pt2, radius=6, color=(0, 0, 0), thickness=-1):
    x1, y1 = pt1
    x2, y2 = pt2

    # Draw straight fills
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)

    # Four circles on corners
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)


# -----------------------------------------
# MODERN TEXT (Rounded background + shadow)
# -----------------------------------------
def draw_modern_text(frame, text, pos, font=cv2.FONT_HERSHEY_DUPLEX, scale=0.7,
                     text_color=(255,255,255), bg_color=(0,0,0), padding=7):

    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x, y = pos

    # Background box
    round_rect(frame,
               (x - padding, y - th - padding),
               (x + tw + padding, y + padding),
               radius=8,
               color=bg_color,
               thickness=-1)

    # Shadow text
    cv2.putText(frame, text, (x+1, y+1), font, scale, (0,0,0), 2, cv2.LINE_AA)

    # Main text
    cv2.putText(frame, text, (x, y), font, scale, text_color, 2, cv2.LINE_AA)


# -----------------------------------------
# ROI BRIGHTNESS CHECK
# -----------------------------------------
def is_region_light(image, polygon, threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, [polygon], 255)
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    return cv2.mean(roi, mask=mask)[0] > threshold


# -----------------------------------------
# VIDEO LOOP
# -----------------------------------------
cap = cv2.VideoCapture("sample1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1100, 700))

    # Draw polygons thin + clean
    cv2.polylines(frame, [RedLight], True, (0, 60, 255), 2)
    cv2.polylines(frame, [GreenLight], True, (0, 255, 60), 2)
    cv2.polylines(frame, [ROI], True, (60, 130, 255), 3)

    # Light states
    red_on = is_region_light(frame, RedLight)
    green_on = is_region_light(frame, GreenLight)

    signal_text = "RED" if red_on else ("GREEN" if green_on else "UNKNOWN")
    draw_modern_text(frame, f"Signal: {signal_text}", (20, 40),
                     bg_color=(20,20,20))

    # YOLO detection
    results = model(frame, conf=0.75)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = coco[cls]

            if label not in TargetLabels:
                continue

            # Stylish bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70,255,150), 2)

            draw_modern_text(frame,
                             f"{label.upper()} {conf*100:.1f}%",
                             (x1, y1 - 10),
                             bg_color=(20,20,20))

            # Check violation
            inside = (
                cv2.pointPolygonTest(ROI, (x1,y1), False) >= 0 or
                cv2.pointPolygonTest(ROI, (x2,y2), False) >= 0
            )

            if red_on and inside:
                draw_modern_text(
                    frame,
                    f"{label.upper()} VIOLATED SIGNAL!",
                    (20, 80),
                    text_color=(255,255,255),
                    bg_color=(200,0,0)
                )

                # Highlight in red
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 3)
                cv2.polylines(frame, [ROI], True, (0,0,255), 3)

    cv2.imshow("Traffic Violation Detection - Modern UI", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()  