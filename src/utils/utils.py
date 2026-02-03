import cv2
import numpy as np


def compute_tenengrad(gray_image):
    """
    Görüntünün netlik haritasını (Tenengrad) çıkarır.
    """
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    focus_measure = cv2.add(np.square(gx), np.square(gy))
    return focus_measure


def draw_zebra(img, gray, under_thresh, over_thresh):
    """
    Aşırı parlak (Kırmızı) ve aşırı karanlık (Mavi) alanları taralı gösterir.
    """
    height, width = gray.shape
    y, x = np.ogrid[:height, :width]
    # 8 piksellik diyagonal çizgiler
    zebra_mask = ((x + y) // 8) % 2 == 0

    u_th = int(under_thresh * 255)
    o_th = int(over_thresh * 255)

    underexposed = (gray < u_th) & zebra_mask
    overexposed = (gray > o_th) & zebra_mask

    img[underexposed] = [255, 0, 0]
    img[overexposed] = [0, 0, 255]
    return img


def draw_peaking(img, focus_map, peaking_threshold):
    """
    Net olan kenarları (yüksek kontrast) yeşil ile parlatır.
    """
    max_val = np.max(focus_map)
    if max_val > 0:
        threshold = max_val * peaking_threshold
        mask = focus_map > threshold
        green_layer = np.zeros_like(img)
        green_layer[mask] = [0, 255, 0]
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)
    return img


def draw_center_marker(img):
    """
    Ekranın tam ortasına beyaz bir artı (+) işareti koyar.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(min(w, h) * 0.03)
    color = (255, 255, 255)
    thickness = 2
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


def get_bbox_coords(det, w, h):
    """
    Standart Detection yapısından koordinatları çeker.
    Yapı: det['boundingBox']['left'/'top'/'width'/'height']
    """
    try:
        bbox = det["boundingBox"]
        x1 = int(bbox["left"])
        y1 = int(bbox["top"])
        x2 = int(bbox["left"] + bbox["width"])
        y2 = int(bbox["top"] + bbox["height"])

        # Görüntü sınırları dışına taşmayı önle (Clip)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        return x1, y1, x2, y2
    except:
        return None


def process_image(
        image,
        mode="General",  # "General" veya "Detection"
        detections=None,
        show_center=False,
        show_peaking=False,
        show_zebra=False,
        thresh_over=0.97,
        thresh_under=0.03,
        peaking_threshold=0.05,
):
    """
    Görüntü işleme fonksiyonu.

    Returns:
        vis_img (numpy.ndarray): İşlenmiş görüntü.
        output_data (float | list): General modda tek float, Detection modda liste.
    """

    # 1. Görüntü Hazırlığı
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()
    else:
        gray = image
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Netlik Hesaplama
    focus_map = compute_tenengrad(gray)
    global_mean = np.mean(focus_map)
    if global_mean == 0: global_mean = 1e-6

    # 3. Efekt Katmanını Hazırla
    effects_layer = vis_img.copy()
    if show_zebra:
        effects_layer = draw_zebra(effects_layer, gray, thresh_under, thresh_over)
    if show_peaking:
        effects_layer = draw_peaking(effects_layer, focus_map, peaking_threshold)

    # 4. Maskeleme ve Veri Hesaplama
    mask = np.zeros(gray.shape, dtype=np.uint8)
    output_data = None

    # Detection listesini ayıkla (Doğrudan 'value' key'inden)
    detection_list = []
    if detections and isinstance(detections, dict):
        detection_list = detections.get("value", [])

    # --- MOD KONTROLÜ ---

    if mode == "Detection":
        # >>> DETECTION MODU <<<
        output_data = []  # Çıktı her zaman liste

        if detection_list:
            h, w = gray.shape[:2]
            for det in detection_list:
                coords = get_bbox_coords(det, w, h)

                if coords:
                    x1, y1, x2, y2 = coords
                    if x2 > x1 and y2 > y1:
                        # ROI Skor
                        roi_score = np.mean(focus_map[y1:y2, x1:x2])
                        # Confidence
                        output_data.append(roi_score / global_mean)
                        # Maskeleme (Efektler için beyaz alan)
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    else:
                        output_data.append(0.0)
                else:
                    output_data.append(0.0)

        # Detection yoksa output_data [] kalır, maske siyah kalır.

    else:
        # >>> GENERAL MOD <<<
        output_data = float(global_mean)  # Çıktı tek float
        mask[:] = 255  # Tüm ekranı maskele

    # 5. Efektleri Uygula
    mask_bool = mask > 0
    vis_img[mask_bool] = effects_layer[mask_bool]

    # 6. Center Marker
    if show_center:
        vis_img = draw_center_marker(vis_img)

    return vis_img, output_data