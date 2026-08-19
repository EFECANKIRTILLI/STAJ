import numpy as np


def calculate_gradients(image):  # Burada görüntüyle aynı boyutta iki boş matris oluşturuyoruz.

    gx = np.zeros_like(image)  # gx = görüntünün yatay yöndeki değişimi
    gy = np.zeros_like(image)  # gy = görüntünün dikey yöndeki değişimi
                               # ortada ciddi bir yatay değişim var. gx bunu yakalar.
    gx[:, 1:-1] = (
        image[:, 2:] - image[:, :-2]
    ) / 2.0            # Burada bir pikselin sağındaki ve solundaki pikseli karşılaştırıyoruz.
                       # Sağa doğru giderken görüntü ne kadar değişiyor?
    gy[1:-1, :] = (
        image[2:, :] - image[:-2, :]
    ) / 2.0            # Aşağı/yukarı giderken görüntü ne kadar değişiyor?

    return gx, gy


def calculate_harris_response(image):

    gx, gy = calculate_gradients(image)

    ix2 = gx * gx
    iy2 = gy * gy
    ixy = gx * gy                # Bunlar bize bir bölgenin hangi yönlerde değiştiğini anlatıyor.

    kernel = np.ones(   # Şöyle bir filtre oluşturuyoruz:
        (3, 3),   #Yani tek bir piksele bakmıyoruz.
        dtype=np.float32 # Pikselin çevresindeki 3×3 bölgeye bakıyoruz:
    ) / 9.0              # Çünkü tek piksel gürültü olabilir. Çevresiyle beraber değerlendirmek daha güvenilir.

    sxx = np.zeros_like(image)
    syy = np.zeros_like(image)
    sxy = np.zeros_like(image)   # Bunlar Harris'in kullanacağı yerel gradient bilgilerini saklıyor.

    for y in range(1, image.shape[0] - 1):   # görüntünün bütün piksellerini geziyoruz.

        for x in range(1, image.shape[1] - 1):

            window_xx = ix2[    # Her pikselin etrafından: 3×3 alan alıyoruz.
                y - 1:y + 2,
                x - 1:x + 2
            ]

            window_yy = iy2[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            window_xy = ixy[
                y - 1:y + 2,
                x - 1:x + 2
            ]

            sxx[y, x] = np.sum(
                window_xx * kernel
            )

            syy[y, x] = np.sum(
                window_yy * kernel
            )

            sxy[y, x] = np.sum(
                window_xy * kernel
            )

    k = 0.04

    determinant = (   # sonra bunun determinantını hesaplıyoruz  determinant iki yönlü değişimin gücünü yakalamamıza yardım ediyor.
        sxx * syy -
        sxy * sxy
    )

    trace = sxx + syy  # Bu bölgedeki toplam değişim ne kadar?

    response = (
        determinant -
        k * (trace ** 2)
    )     #burada baktıgımızz şey şu  değişim yüksek mi değil sadece
          # Düz alan mı? Kenar mı?  Köşe mi?

    return response   # Artık görüntüdeki her pikselin bir Harris skoru var.


def find_feature_points(
        response,
        max_points=300,
        grid_rows=8,
        grid_cols=12
):

    height, width = response.shape

    candidates = []

    margin = 10

    threshold = np.max(response) * 0.01   # Harris skoru 1000'in altındaysa bu noktayla uğraşma örneğin.
                                          # Böylece zayıf noktaları eliyoruz.
    # --------------------------------------------------
    # 1. Önce bütün güçlü Harris noktalarını buluyoruz
    # --------------------------------------------------

    for y in range(
        margin,
        height - margin
    ):

        for x in range(
            margin,
            width - margin
        ):

            current = response[y, x]

            if current < threshold:
                continue

            neighborhood = response[
                y - 1:y + 2,
                x - 1:x + 2     # 3×3 çevrede sadece en güçlü olanı tut.
                                # Buna Non-Maximum Suppression (NMS) deniyor.
            ]

            if current == np.max(neighborhood):

                candidates.append(
                    (
                        x,
                        y,
                        current
                    )
                )

    # Güçlü noktalar önce gelsin
    candidates.sort(
        key=lambda point: point[2],
        reverse=True
    )

    # --------------------------------------------------
    # 2. Görüntüyü grid bölgelerine ayırıyoruz
    # Diyelim fotoğrafın sol altında çok fazla taş var
    # Harris en güçlü 500 noktayı seçerse hepsi burada toplanabilir.
    # --------------------------------------------------

    cell_height = height / grid_rows
    cell_width = width / grid_cols

    points_per_cell = max(
        1,
        max_points // (grid_rows * grid_cols)
    )

    selected_points = []

    # --------------------------------------------------
    # 3. Her bölgeden sınırlı sayıda nokta alıyoruz
    # --------------------------------------------------

    for row in range(grid_rows):

        for col in range(grid_cols):

            x_min = col * cell_width
            x_max = (col + 1) * cell_width

            y_min = row * cell_height
            y_max = (row + 1) * cell_height

            cell_points = []

            for point in candidates:

                x, y, score = point

                if (
                    x_min <= x < x_max
                    and
                    y_min <= y < y_max
                ):

                    cell_points.append(point)

                    if len(cell_points) >= points_per_cell:
                        break

            selected_points.extend(
                cell_points
            )

    # --------------------------------------------------
    # 4. Eğer hedef sayıya ulaşmadıysak
    #    kalan güçlü noktaları ekle
    # --------------------------------------------------

    if len(selected_points) < max_points:

        selected_set = {
            (p[0], p[1])
            for p in selected_points
        }

        for point in candidates:

            if (
                point[0],
                point[1]
            ) in selected_set:
                continue

            selected_points.append(point)

            if len(selected_points) >= max_points:
                break

    # Güç sırasına göre tekrar sırala
    selected_points.sort(
        key=lambda point: point[2],
        reverse=True
    )

    return selected_points[:max_points]