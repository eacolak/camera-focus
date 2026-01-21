import cv2
import numpy as np


def compute_tenengrad(gray_image):
    """
    Computes the sharpness map of the image using the Tenengrad algorithm.
    Uses the sum of squares of Sobel operators.
    """
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    focus_measure = cv2.add(np.square(gx), np.square(gy))
    return focus_measure


def draw_zebra(img, gray, under_thresh, over_thresh):
    """
    Draws a Zebra pattern based on brightness thresholds.
    under_thresh and over_thresh: float values between 0.0 and 1.0.
    """
    height, width = gray.shape

    y, x = np.ogrid[:height, :width]
    zebra_mask = ((x + y) // 8) % 2 == 0

    u_th = int(under_thresh * 255)
    o_th = int(over_thresh * 255)

    underexposed = (gray < u_th) & zebra_mask
    overexposed = (gray > o_th) & zebra_mask

    img[underexposed] = [255, 0, 0]
    img[overexposed] = [0, 0, 255]

    return img


def draw_peaking(img, focus_map, threshold_percent=0.3):
    """
    Highlights the sharpest areas (Focus Peaking) in green.
    """
    max_val = np.max(focus_map)

    if max_val > 0:
        threshold = max_val * threshold_percent
        mask = focus_map > threshold

        green_layer = np.zeros_like(img)
        green_layer[mask] = [0, 255, 0]
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)

    return img


def draw_hud(img, gray, focus_score):
    """
    Draws an information panel (HUD) and histogram in the top-left corner.
    """
    h, w = img.shape[:2]
    scale = max(0.5, min(w, h) / 1000.0)

    box_w, box_h = int(250 * scale), int(120 * scale)
    margin = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_w, margin + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)  # 0.6 Opacity

    cv2.rectangle(img, (margin, margin), (margin + box_w, margin + box_h), (255, 255, 255), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(img, "FOCUS SCORE",
                (int(margin + 10 * scale), int(margin + 30 * scale)),
                font, 0.6 * scale, (200, 200, 200), 1)

    cv2.putText(img, f"{focus_score:.2f}",
                (int(margin + 10 * scale), int(margin + 70 * scale)),
                font, 1.2 * scale, (0, 255, 0), 2)

    hist_h = int(40 * scale)
    hist_w = int(100 * scale)

    hist_x = margin + box_w - hist_w - 10
    hist_y = margin + box_h - 10

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)

    bin_w = hist_w / 256
    for i in range(1, 256):
        pt1 = (int(hist_x + (i - 1) * bin_w), int(hist_y - hist[i - 1]))
        pt2 = (int(hist_x + (i) * bin_w), int(hist_y - hist[i]))
        cv2.line(img, pt1, pt2, (150, 150, 150), 1)

    return img


def draw_center_marker(img):
    """
    Places a yellow cross (+) marker at the center.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    size = int(min(w, h) * 0.03)
    color = (0, 255, 255)  # Yellow
    thickness = 2

    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


def process_image(image, mode="General", detections=None, show_center=True, show_hud=True, show_peaking=True,
                  show_zebra=True, thresh_over=0.97, thresh_under=0.03):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()
    else:
        gray = image
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    effects_layer = vis_img.copy()
    focus_map = compute_tenengrad(gray)
    final_score = 0.0
    mask = None

    if mode == "General":
        final_score = np.mean(focus_map)

    elif mode == "Detection":
        detection_list = detections
        if hasattr(detections, 'value'):
            detection_list = detections.value

        mask = np.zeros(gray.shape, dtype=np.uint8)

        if detection_list and isinstance(detection_list, list) and len(detection_list) > 0:
            scores = []
            h, w = gray.shape

            for det in detection_list:
                try:
                    coords = None
                    if hasattr(det, "absolute_bounding_box"):
                        coords = det.absolute_bounding_box
                    elif isinstance(det, dict) and "absolute_bounding_box" in det:
                        coords = det["absolute_bounding_box"]
                    elif isinstance(det, (list, tuple)) and len(det) >= 4:
                        coords = det[:4]

                    if coords:
                        v1, v2, v3, v4 = map(int, coords)
                        x1, y1, x2, y2 = v1, v2, v3, v4
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        if x2 > x1 and y2 > y1:
                            roi_score = np.mean(focus_map[y1:y2, x1:x2])
                            scores.append(roi_score)

                            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                            if show_hud:
                                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"F:{roi_score:.1f}"
                                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(vis_img, (x1, y1 - lh - 4), (x1 + lw, y1), (0, 0, 0), -1)
                                cv2.putText(vis_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except:
                    pass

            if scores:
                final_score = sum(scores) / len(scores)
        else:
            cv2.putText(vis_img, "NO DETECTIONS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if show_zebra:
        effects_layer = draw_zebra(effects_layer, gray, thresh_under, thresh_over)

    if show_peaking:
        effects_layer = draw_peaking(effects_layer, focus_map)

    if mode == "General":
        vis_img = effects_layer
    elif mode == "Detection" and mask is not None:
        mask_bool = mask > 0
        vis_img[mask_bool] = effects_layer[mask_bool]

    if show_center:
        vis_img = draw_center_marker(vis_img)

    if show_hud:
        vis_img = draw_hud(vis_img, gray, final_score)

    return vis_img