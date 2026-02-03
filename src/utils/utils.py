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

    # Mavi ve Kırmızı boyama
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
        green_layer[mask] = [0, 255, 0]  # Yeşil
        # Additive blending (Parlatma etkisi)
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)
    return img


def draw_center_marker(img):
    """
    Ekranın tam ortasına beyaz bir artı (+) işareti koyar.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    size = int(min(w, h) * 0.03)  # Ekran boyutuna göre ölçekle
    color = (255, 255, 255)
    thickness = 2
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


def get_bbox_coords(det, w, h):
    """
    Farklı formatlardaki bounding box verilerini (left/top/width/height)
    OpenCV formatına (x1, y1, x2, y2) çevirir.
    """
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
        # Normalized (0-1) kontrolü
        if v1 <= 1.0 and v3 <= 1.0 and v1 >= 0 and v3 >= 0:
            x1, y1, x2, y2 = int(v1 * w), int(v2 * h), int(v3 * w), int(v4 * h)
        else:
            x1, y1, x2, y2 = int(v1), int(v2), int(v3), int(v4)
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    except:
        return None


def process_image(
        image,
        detections=None,
        show_center=False,  # Center Marker'ı açar/kapatır
        show_peaking=False,  # Focus Peaking'i açar/kapatır
        show_zebra=False,  # Zebra Warnings'i açar/kapatır
        thresh_over=0.97,  # Zebra için aşırı parlaklık eşiği
        thresh_under=0.03,  # Zebra için karanlık eşiği
        peaking_threshold=0.05,  # Peaking hassasiyeti
):
    """
    Görüntü işleme ve analiz fonksiyonu.

    Returns:
        vis_img (numpy.ndarray): Efekt uygulanmış görüntü (Yazısız).
        global_score (float): Tüm görüntünün netlik skoru.
        detection_scores (list): Algılanan nesnelerin netlik skorları (Focus Confidence).
    """

    # 1. Griye Çevirme ve Kopya Oluşturma
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()
    else:
        gray = image
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Netlik Haritası ve Global Skor Hesaplama
    focus_map = compute_tenengrad(gray)
    global_mean = np.mean(focus_map)
    if global_mean == 0: global_mean = 1e-6  # Sıfıra bölünme hatası önlemi

    # 3. Efekt Katmanını Hazırla
    effects_layer = vis_img.copy()

    if show_zebra:
        effects_layer = draw_zebra(effects_layer, gray, thresh_under, thresh_over)
    if show_peaking:
        effects_layer = draw_peaking(effects_layer, focus_map, peaking_threshold)

    # 4. Maskeleme ve Detection Skorlama
    mask = np.zeros(gray.shape, dtype=np.uint8)

    detection_list = []
    detection_scores = []  # Sayısal çıktı listesi

    is_detection_mode = (detections is not None)

    # Detection verisini ayıkla
    if is_detection_mode:
        if hasattr(detections, 'value'):
            detection_list = detections.value
        elif isinstance(detections, list):
            detection_list = detections

    # Mod Kontrolü: General vs Detection
    if not is_detection_mode or not detection_list:
        # General Mod: Maske tüm ekranı kaplar
        mask[:] = 255
    else:
        # Detection Mod: Sadece kutuların içi maskelenir
        h, w = gray.shape[:2]

        for det in detection_list:
            coords = get_bbox_coords(det, w, h)

            if coords:
                x1, y1, x2, y2 = coords
                # Geçerli bir kutu mu?
                if x2 > x1 and y2 > y1:
                    # Bölgesel skor hesapla
                    roi_score = np.mean(focus_map[y1:y2, x1:x2])

                    # Focus Confidence (ROI / Global)
                    focus_confidence = roi_score / global_mean
                    detection_scores.append(focus_confidence)

                    # Maskeyi sadece bu kutu için beyaza boya (Efektler burada görünsün)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                    # DİKKAT: Buraya bilerek putText (yazı) koymadık.
                    # Kullanıcı sadece temiz görüntü ve veri istedi.
            else:
                # Koordinat hatalıysa listeye None veya 0 eklenebilir
                # İndeks kaymaması için 0.0 ekliyoruz
                detection_scores.append(0.0)

    # 5. Efektleri Uygula (Maskelenmiş alana)
    mask_bool = mask > 0
    vis_img[mask_bool] = effects_layer[mask_bool]

    # 6. Center Marker (Maskeden bağımsız, en üste çizilir)
    if show_center:
        vis_img = draw_center_marker(vis_img)

    # 7. Çıktıları Döndür (Resim, Global Veri, Liste Verisi)
    return vis_img, float(global_mean), detection_scores