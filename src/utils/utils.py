import cv2
import numpy as np


def _draw_text_with_outline(img, text, pos, font, scale, color, thickness):
    x, y = pos
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


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


def draw_peaking(img, focus_map, peaking_threshold):
    max_val = np.max(focus_map)
    if max_val > 0:
        threshold = max_val * peaking_threshold
        mask = focus_map > threshold
        green_layer = np.zeros_like(img)
        green_layer[mask] = [0, 255, 0]
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)
    return img


def draw_hud(img, gray, original_image, focus_score):
    h, w = img.shape[:2]
    reference_size = 720
    scale = min(h, w) / reference_size
    scale = max(0.4, min(scale, 2.5))

    padding = int(14 * scale)
    hist_width = int(180 * scale)
    hist_height = int(50 * scale)
    margin = int(12 * scale)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    thickness = max(1, int(1.5 * scale))
    line_spacing = int(6 * scale)

    display_score = focus_score / 1000.0
    label_text = "GLOBAL FOCUS"
    score_text = f"{display_score:.1f}"

    label_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]

    box_width = max(hist_width, label_size[0]) + (padding * 2)
    box_height = (padding * 3) + label_size[1] + line_spacing + hist_height + int(20 * scale)

    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_width, margin + box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    cv2.rectangle(img, (margin, margin), (margin + box_width, margin + box_height), (80, 80, 80), 1)

    text_x = margin + padding
    text_y = margin + padding + label_size[1]

    _draw_text_with_outline(img, label_text, (text_x, text_y), font, font_scale, (200, 200, 200), thickness)

    text_y += int(25 * scale)
    _draw_text_with_outline(img, score_text, (text_x, text_y), font, font_scale * 1.5, (0, 255, 0), thickness + 1)

    hist_x = text_x
    hist_y = text_y + line_spacing + int(10 * scale)
    hist_bottom = hist_y + hist_height

    cv2.rectangle(img, (hist_x, hist_y), (hist_x + hist_width, hist_bottom), (20, 20, 20), -1)

    x_coords = np.linspace(hist_x, hist_x + hist_width - 1, 256).astype(np.int32)

    if len(original_image.shape) == 3:
        channel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for ch, color in enumerate(channel_colors):
            hist = cv2.calcHist([original_image], [ch], None, [256], [0, 256])
            hist_max = hist.max()
            if hist_max > 0:
                hist_normalized = (hist / hist_max * hist_height).astype(np.int32).flatten()
                pts = np.column_stack([x_coords, hist_bottom - hist_normalized]).astype(np.int32)
                cv2.polylines(img, [pts], False, color, 1)

    gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    gray_max = gray_hist.max()
    if gray_max > 0:
        gray_norm = (gray_hist / gray_max * hist_height).astype(np.int32).flatten()
        pts = np.column_stack([x_coords, hist_bottom - gray_norm]).astype(np.int32)
        cv2.polylines(img, [pts], False, (200, 200, 200), 1)

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

    if coords is None: return None

    try:
        v1, v2, v3, v4 = map(float, coords)
        if v1 <= 1.0 and v3 <= 1.0 and v1 >= 0 and v3 >= 0:
            x1, y1, x2, y2 = int(v1 * w), int(v2 * h), int(v3 * w), int(v4 * h)
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
        thresh_under=0.03,
        peaking_threshold=0.05,
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
    global_mean = np.mean(focus_map)
    if global_mean == 0: global_mean = 1e-6

    effects_layer = vis_img.copy()

    if show_zebra:
        effects_layer = draw_zebra(effects_layer, gray, thresh_under, thresh_over)
    if show_peaking:
        effects_layer = draw_peaking(effects_layer, focus_map, peaking_threshold)

    mask = np.zeros(gray.shape, dtype=np.uint8)

    detection_list = []
    is_detection_mode = (detections is not None)

    if is_detection_mode:
        if hasattr(detections, 'value'):
            detection_list = detections.value
        elif isinstance(detections, list):
            detection_list = detections

    if not is_detection_mode:
        mask[:] = 255
    elif detection_list:
        h, w = gray.shape[:2]

        for det in detection_list:
            coords = get_bbox_coords(det, w, h)

            if coords:
                x1, y1, x2, y2 = coords
                if x2 > x1 and y2 > y1:
                    roi_score = np.mean(focus_map[y1:y2, x1:x2])
                    focus_confidence = roi_score / global_mean

                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                    if show_hud:
                        label = f"x{focus_confidence:.1f}"

                        text_color = (0, 255, 0)
                        if focus_confidence < 1.0:
                            text_color = (0, 0, 255)
                        elif focus_confidence < 1.5:
                            text_color = (0, 255, 255)

                        _draw_text_with_outline(
                            vis_img,
                            label,
                            (x1 + 5, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            text_color,
                            2
                        )

    mask_bool = mask > 0
    vis_img[mask_bool] = effects_layer[mask_bool]

    if show_center:
        vis_img = draw_center_marker(vis_img)

    if show_hud and not is_detection_mode:
        vis_img = draw_hud(vis_img, gray, image, global_mean)

    return vis_img