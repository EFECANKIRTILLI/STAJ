from PIL import Image, ImageDraw

from feature_detection import (
    calculate_harris_response,
    find_feature_points
)


def main():

    image_path = "cukur_deneme_data/cukur_01.jpg"

    # Orijinal görüntüyü renkli olarak aç
    image = Image.open(image_path).convert("RGB")

    # Harris için grayscale görüntü
    gray = image.convert("L")

    # Harris hesabı için numpy array
    import numpy as np

    gray_array = np.array(
        gray,
        dtype=np.float32
    )

    # Harris response
    harris = calculate_harris_response( # Fotoğraftaki belirgin noktaları Harris ile değerlendir.
        gray_array
    )

    # Feature noktaları
    points = find_feature_points(
        harris,
        max_points=300  # Bunların arasından en iyi 300 noktayı seç.
    )

    # Çizim yapabilmek için
    draw = ImageDraw.Draw(image)

    # Her feature noktasını çiz
    for x, y, response in points:  # Her bulunan noktayı tek tek dolaşıyor.

        radius = 3

        draw.ellipse(  #o noktaya kırmızı yuvarlak çiziyor.
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ),
            fill=(255, 0, 0)
        )

    # Sonucu kaydet
    output_path = "cukur_deneme_data/features_01.jpg"

    image.save(
        output_path,
        quality=95
    )

    print("Feature görselleştirmesi oluşturuldu.")
    print("Nokta sayısı:", len(points))
    print("Dosya:", output_path)


if __name__ == "__main__":

    main()