from PIL import Image, ImageDraw
import numpy as np

from feature_detection import (
    calculate_harris_response,
    find_feature_points
)

from feature_matching import (
    create_patch_descriptors,
    find_ratio_matches
)


def load_gray(path):  # fotoğrafın adresi

    image = Image.open(path).convert("L")  # fotoğrafı siyah beyaz yapıyor

    return np.array(
        image,
        dtype=np.float32  # fotoğrafı sayılardan oluşan bir matrise çeviriyor.
                          #Çünkü Harris gibi matematiksel işlemleri resim dosyasının kendisi üzerinde değil, piksel değerleri üzerinde yapıyoruz.

    )


def extract_features(path):  # Bir fotoğraf ver → bana önemli noktaları ve bu noktaların descriptorlarını ver.

    image = load_gray(path)  # fotoğrafı yükle.

    harris = calculate_harris_response(   # Harris algoritmasını çalıştır.
        image                             # Harris'in görevi: Fotoğrafta takip edilmesi kolay, belirgin noktalar nerede?
    )

    points = find_feature_points(
        harris,
        max_points=500   # Harris'in bulduğu noktalardan en fazla 500 tane seçiyoruz.
    )

    valid_points, descriptors = (
        create_patch_descriptors(
            image,
            points,
            patch_size=24   # Burada her feature noktasının etrafındaki 24×24 piksellik bölgeye bakıyoruz.
        )
    )

    return (
        valid_points,
        descriptors
    )


def main():

    image1_path = (
        "sfm_data/"
        "1494491536630704618.png"
    )

    image2_path = (
        "sfm_data/"
        "1494491536704432618.png"
    )

    # --------------------------------------------------
    # Feature çıkar
    # --------------------------------------------------

    points1, descriptors1 = (
        extract_features(
            image1_path
        )
    )

    points2, descriptors2 = (
        extract_features(
            image2_path
        )
    )

    # --------------------------------------------------
    # Matching
    # --------------------------------------------------

    matches = find_ratio_matches(
        descriptors1,
        descriptors2,
        ratio_threshold=0.85    # Fotoğraf 1'deki bu noktanın Fotoğraf 2'de karşılığı hangisi? onu eşleştiriyor.
    )

    print(
        "Toplam eşleşme:",
        len(matches)
    )

    # --------------------------------------------------
    # Görüntüleri renkli yükle
    # --------------------------------------------------

    image1 = Image.open(
        image1_path
    ).convert("RGB")

    image2 = Image.open(
        image2_path
    ).convert("RGB")

    width1, height1 = (
        image1.size
    )

    width2, height2 = (
        image2.size
    )

    # --------------------------------------------------
    # Yan yana görüntü oluştur
    # --------------------------------------------------

    output_width = (
        width1 +
        width2
    )

    output_height = max(
        height1,
        height2
    )

    output = Image.new(
        "RGB",
        (
            output_width,
            output_height
        )
    )

    output.paste(
        image1,
        (0, 0)
    )

    output.paste(
        image2,
        (width1, 0)
    )

    draw = ImageDraw.Draw(
        output
    )

    # --------------------------------------------------
    # Sadece en iyi 100 eşleşmeyi çiz. NEDEN 100 ? Hepsini çizersek görüntü çok karışır. Ondan dolayı en iyi 100'ü seçtim.
    # --------------------------------------------------

    best_matches = (
        matches[:100]
    )

    for match in best_matches:

        index1 = match[0]
        index2 = match[1]

        point1 = points1[
            index1
        ]

        point2 = points2[
            index2
        ]

        x1 = int(
            point1[0]
        )

        y1 = int(
            point1[1]
        )

        x2 = (
            int(point2[0])
            +
            width1
        )

        y2 = int(
            point2[1]
        )

        # Noktalar
        radius = 3

        draw.ellipse(
            (
                x1 - radius,
                y1 - radius,
                x1 + radius,
                y1 + radius
            ),
            fill=(255, 0, 0)
        )

        draw.ellipse(
            (
                x2 - radius,
                y2 - radius,
                x2 + radius,
                y2 + radius
            ),
            fill=(255, 0, 0)
        )

        # Eşleşme çizgisi
        draw.line(
            (
                x1,
                y1,
                x2,
                y2
            ),
            fill=(255, 0, 0),
            width=1
        )

    # --------------------------------------------------
    # Kaydet
    # --------------------------------------------------

    output_path = (
        "sfm_data/"
        "matches_01_02.jpg"
    )

    output.save(
        output_path,
        quality=95
    )

    print(
        "Matching görseli oluşturuldu:"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()