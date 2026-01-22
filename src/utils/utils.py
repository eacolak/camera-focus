import cv2
import numpy as np


def compute_tenengrad(gray_image):
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    focus_measure = cv2.add(np.square(gx), np.square(gy))
    return focus_measure


def draw_zebra(img, gray, under_thresh, over_thresh):
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
    max_val = np.max(focus_map)
    if max_val > 0:
        threshold = max_val * threshold_percent
        mask = focus_map > threshold
        green_layer = np.zeros_like(img)
        green_layer[mask] = [0, 255, 0]
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)
    return img


def draw_hud(img, gray, focus_score):
    h, w = img.shape[:2]
    scale = max(0.5, min(w, h) / 1000.0)
    box_w, box_h = int(250 * scale), int(120 * scale)
    margin = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_w, margin + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

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
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(min(w, h) * 0.03)
    color = (0, 255, 255)
    thickness = 2
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


def get_bbox_coords(det, w, h):
    coords = None

    if isinstance(det, dict) and "boundingBox" in det:
        bbox = det["boundingBox"]
        if isinstance(bbox, dict):
            l = float(bbox.get("left", 0))
            t = float(bbox.get("top", 0))
            bw = float(bbox.get("width", 0))
            bh = float(bbox.get("height", 0))
            coords = [l, t, l + bw, t + bh]

    elif hasattr(det, "absolute_bounding_box"):
        coords = det.absolute_bounding_box
    elif hasattr(det, "bbox"):
        coords = det.bbox
    elif isinstance(det, dict) and "absolute_bounding_box" in det:
        coords = det["absolute_bounding_box"]
    elif isinstance(det, (list, tuple)) and len(det) >= 4:
        coords = det[:4]

    if coords is None:
        return None

    try:
        v1, v2, v3, v4 = map(float, coords)

        if v1 <= 1.0 and v3 <= 1.0 and v1 >= 0 and v3 >= 0:
            x1 = int(v1 * w)
            y1 = int(v2 * h)
            x2 = int(v3 * w)
            y2 = int(v4 * h)
        else:
            x1, y1, x2, y2 = int(v1), int(v2), int(v3), int(v4)

        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    except:
        return None


def process_image(
        image,
        mode="General",
        detections=None,
        show_center=True,
        show_hud=True,
        show_peaking=True,
        show_zebra=True,
        thresh_over=0.97,
        thresh_under=0.03
):
    if show_hud is None: show_hud = True
    if show_center is None: show_center = True

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()
    else:
        gray = image
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    focus_map = compute_tenengrad(gray)
    effects_layer = vis_img.copy()

    if show_zebra:
        effects_layer = draw_zebra(effects_layer, gray, thresh_under, thresh_over)
    if show_peaking:
        effects_layer = draw_peaking(effects_layer, focus_map)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    final_score = 0.0

    detection_list = []
    is_detection_mode = (detections is not None)

    if is_detection_mode:
        if hasattr(detections, 'value'):
            detection_list = detections.value
        elif isinstance(detections, list):
            detection_list = detections

    if not is_detection_mode:
        mask[:] = 255
        final_score = np.mean(focus_map)

    elif detection_list:
        scores = []
        h, w = gray.shape[:2]

        for det in detection_list:
            coords = get_bbox_coords(det, w, h)

            if coords:
                x1, y1, x2, y2 = coords

                if x2 > x1 and y2 > y1:
                    roi_score = np.mean(focus_map[y1:y2, x1:x2])
                    scores.append(roi_score)

                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    if show_hud:
                        label = f"F:{roi_score:.1f}"
                        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(vis_img, (x1, y1 - lh - 4), (x1 + lw, y1), (0, 0, 0), -1)
                        cv2.putText(vis_img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if scores:
            final_score = sum(scores) / len(scores)
        else:
            cv2.putText(vis_img, "INVALID BBOX DATA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    mask_bool = mask > 0
    vis_img[mask_bool] = effects_layer[mask_bool]

    if show_center:
        vis_img = draw_center_marker(vis_img)

    if show_hud and not is_detection_mode:
        vis_img = draw_hud(vis_img, gray, final_score)

    return vis_img