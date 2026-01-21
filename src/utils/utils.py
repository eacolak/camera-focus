import cv2
import numpy as np


def compute_tenengrad(gray_image):
    """
    Tenengrad algoritması ile görüntünün netlik haritasını çıkarır.
    Sobel operatörlerinin karelerinin toplamını kullanır.
    """
    # CV_32F veya CV_64F kullanarak hassasiyeti koruyoruz
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

    # Magnitude (Gx^2 + Gy^2)
    focus_measure = cv2.add(np.square(gx), np.square(gy))
    return focus_measure


def draw_zebra(img, gray, under_thresh, over_thresh):
    """
    Parlaklık eşiklerine göre Zebra deseni çizer.
    under_thresh ve over_thresh: 0.0 - 1.0 arasında float değerler.
    """
    height, width = gray.shape

    # Zebra deseni maskesi oluştur (Diyagonal çizgiler)
    # Numpy grid işlemleri ile hızlıca oluşturuyoruz
    y, x = np.ogrid[:height, :width]
    zebra_mask = ((x + y) // 8) % 2 == 0

    # Float eşikleri 0-255 arasına çevir
    u_th = int(under_thresh * 255)
    o_th = int(over_thresh * 255)

    # Maskeleme
    underexposed = (gray < u_th) & zebra_mask
    overexposed = (gray > o_th) & zebra_mask

    # Boyama (BGR Formatı)
    # Mavi (Karanlık bölgeler)
    img[underexposed] = [255, 0, 0]
    # Kırmızı (Patlayan bölgeler)
    img[overexposed] = [0, 0, 255]

    return img


def draw_peaking(img, focus_map, threshold_percent=0.3):
    """
    En net bölgeleri (Focus Peaking) yeşil ile boyar.
    """
    max_val = np.max(focus_map)

    # Eğer görüntü tamamen siyahsa veya netlik yoksa işlem yapma
    if max_val > 0:
        # En yüksek skorun %30'undan fazlasına sahip alanları seç
        threshold = max_val * threshold_percent
        mask = focus_map > threshold

        # Yeşil katman oluştur
        green_layer = np.zeros_like(img)
        green_layer[mask] = [0, 255, 0]

        # Orijinal resimle yarı saydam karıştır (Alpha blending)
        cv2.addWeighted(img, 1.0, green_layer, 0.5, 0, img)

    return img


def draw_hud(img, gray, focus_score):
    """
    Sol üst köşeye bilgi paneli (HUD) ve histogram çizer.
    """
    h, w = img.shape[:2]
    # Dinamik ölçekleme: Resim büyüdükçe yazılar da büyüsün
    scale = max(0.5, min(w, h) / 1000.0)

    # Arkaplan kutusu boyutları
    box_w, box_h = int(250 * scale), int(120 * scale)
    margin = 10

    # Yarı saydam siyah arkaplan
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_w, margin + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)  # 0.6 Opaklık

    # Beyaz çerçeve
    cv2.rectangle(img, (margin, margin), (margin + box_w, margin + box_h), (255, 255, 255), 1)

    # Yazı Ayarları
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Başlık
    cv2.putText(img, "FOCUS SCORE",
                (int(margin + 10 * scale), int(margin + 30 * scale)),
                font, 0.6 * scale, (200, 200, 200), 1)

    # Skor Değeri
    cv2.putText(img, f"{focus_score:.2f}",
                (int(margin + 10 * scale), int(margin + 70 * scale)),
                font, 1.2 * scale, (0, 255, 0), 2)

    # --- Histogram Çizimi ---
    hist_h = int(40 * scale)
    hist_w = int(100 * scale)
    # Histogramı kutunun sağ altına yasla
    hist_x = margin + box_w - hist_w - 10
    hist_y = margin + box_h - 10

    # Histogram hesapla (Sadece parlaklık/gri kanalı)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_h, cv2.NORM_MINMAX)

    # Çizgileri çiz
    bin_w = hist_w / 256
    for i in range(1, 256):
        pt1 = (int(hist_x + (i - 1) * bin_w), int(hist_y - hist[i - 1]))
        pt2 = (int(hist_x + (i) * bin_w), int(hist_y - hist[i]))
        cv2.line(img, pt1, pt2, (150, 150, 150), 1)

    return img


def draw_center_marker(img):
    """
    Merkeze sarı bir artı (+) işareti koyar.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Artının boyutu resmin %3'ü kadar olsun
    size = int(min(w, h) * 0.03)
    color = (0, 255, 255)  # Sarı
    thickness = 2

    cv2.line(img, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, thickness)
    return img


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
    """
    Ana işlem fonksiyonu.
    Görüntüyü alır, analiz eder ve seçilen görselleştirmeleri üzerine çizer.
    Tüm parametreler argüman olarak gelmelidir.
    """
    # 1. Görüntüyü Hazırla (Renkli ve Gri Kopya)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        vis_img = image.copy()
    else:
        gray = image
        vis_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # 2. Tenengrad (Netlik) Haritasını Hesapla
    focus_map = compute_tenengrad(gray)
    final_score = 0.0

    # 3. Mod Mantığı (General vs Detection)
    if mode == "General":
        # Tüm resmin ortalaması
        final_score = np.mean(focus_map)

    elif mode == "Detection":
        # Detection Listesini İşle
        # Detections: List[DetectionObj] veya List[List] veya None olabilir
        detection_list = detections

        # Eğer parametre bir nesne içinde geldiyse (Pydantic value)
        if hasattr(detections, 'value'):
            detection_list = detections.value

        if detection_list and isinstance(detection_list, list) and len(detection_list) > 0:
            scores = []
            h, w = gray.shape

            for det in detection_list:
                try:
                    # Koordinatları Çıkarma (Farklı formatlara karşı dirençli)
                    coords = None

                    # 1. Novavision/SDK Nesnesi ise
                    if hasattr(det, "absolute_bounding_box"):
                        coords = det.absolute_bounding_box
                    # 2. Sözlük (Dictionary) ise
                    elif isinstance(det, dict) and "absolute_bounding_box" in det:
                        coords = det["absolute_bounding_box"]
                    # 3. Liste/Tuple ise [x, y, x2, y2]
                    elif isinstance(det, (list, tuple)) and len(det) >= 4:
                        coords = det[:4]

                    if coords:
                        # Koordinatları integer'a çevir
                        # Varsayım: [x1, y1, x2, y2] formatı
                        v1, v2, v3, v4 = map(int, coords)
                        x1, y1, x2, y2 = v1, v2, v3, v4

                        # Clipping (Resim sınırlarına zorla)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        # Geçerli bir kutu mu?
                        if x2 > x1 and y2 > y1:
                            # Sadece kutu içindeki odak skorunu al
                            roi_score = np.mean(focus_map[y1:y2, x1:x2])
                            scores.append(roi_score)

                            # Kutu Görselleştirmesi (Eğer HUD açıksa skorları kutuya yaz)
                            if show_hud:
                                # Yeşil Kutu
                                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                # Skor Etiketi
                                label = f"F:{roi_score:.1f}"
                                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                cv2.rectangle(vis_img, (x1, y1 - lh - 4), (x1 + lw, y1), (0, 0, 0),
                                              -1)  # Yazı arkaplanı
                                cv2.putText(vis_img, label, (x1, y1 - 2),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"DEBUG: Detection işlenirken hata oluştu: {e}")

            # Ortalama skoru hesapla (HUD için)
            if scores:
                final_score = sum(scores) / len(scores)
        else:
            # Detection modunda ama veri yoksa uyarı bas
            cv2.putText(vis_img, "NO DETECTIONS FOUND", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 4. Görselleştirmeleri Uygula (Sırayla)

    # Zebra (Pozlama Uyarısı)
    if show_zebra:
        vis_img = draw_zebra(vis_img, gray, thresh_under, thresh_over)

    # Focus Peaking (Yeşil Odak)
    if show_peaking:
        vis_img = draw_peaking(vis_img, focus_map)

    # Center Marker (Artı İşareti)
    if show_center:
        vis_img = draw_center_marker(vis_img)

    # HUD (Bilgi Paneli - En üstte olması iyidir)
    if show_hud:
        vis_img = draw_hud(vis_img, gray, final_score)

    return vis_img