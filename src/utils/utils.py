import cv2
import numpy as np


def compute_tenengrad(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
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


def draw_center_marker(img):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(min(w, h) * 0.03)
    color = (255, 255, 255)
    thickness = 2
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


def get_bbox_coords(det, w, h):
    try:
        bbox = det["boundingBox"]
        x1 = int(bbox["left"])
        y1 = int(bbox["top"])
        x2 = int(bbox["left"] + bbox["width"])
        y2 = int(bbox["top"] + bbox["height"])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2
    except:
        return None


def process_image(
        image,
        mode="General",
        detections=None,
        show_center=False,
        show_peaking=False,
        show_zebra=False,
        thresh_over=0.97,
        thresh_under=0.03,
        peaking_threshold=0.05,
):
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
    output_data = None

    detection_list = []

    if detections:
        if isinstance(detections, list):
            detection_list = detections
        elif isinstance(detections, dict):
            detection_list = detections.get("value", [])

    if mode == "Detection":

        if detection_list:
            h, w = gray.shape[:2]

            for det in detection_list:
                coords = get_bbox_coords(det, w, h)

                score = 0.0

                if coords:
                    x1, y1, x2, y2 = coords
                    if x2 > x1 and y2 > y1:
                        roi_score = np.mean(focus_map[y1:y2, x1:x2])
                        score = float(roi_score / global_mean)

                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                det['focus_confidence'] = score

            output_data = detection_list
        else:
            output_data = []

    else:
        output_data = float(global_mean)
        mask[:] = 255

    mask_bool = mask > 0
    vis_img[mask_bool] = effects_layer[mask_bool]

    if show_center:
        vis_img = draw_center_marker(vis_img)

    return vis_img, output_data