import numpy as np


def calculate_local_gradients(patch):   # Harris'in bulduğu bir feature noktasının çevresindeki küçük görüntü parçası= patch

    gx = np.zeros_like(
        patch,
        dtype=np.float32
    )

    gy = np.zeros_like(
        patch,
        dtype=np.float32
    )

    gx[:, 1:-1] = (
        patch[:, 2:] -
        patch[:, :-2]
    ) / 2.0

    gy[1:-1, :] = (
        patch[2:, :] -
        patch[:-2, :]
    ) / 2.0 # sağ sol yukarı aşağı değişimleri hesaplıyor
            # Ama bu sefer bütün fotoğraf yerine feature'ın küçük çevresinde yapıyoruz.

    magnitude = np.sqrt(
        gx * gx +
        gy * gy
    )      # Buradaki görüntü değişimi ne kadar güçlü?
           # düz alasna düşük ama keskin kenarsa yüksek mesela

    angle = np.arctan2(
        gy,
        gx    # Değişim hangi yönde?
    )
    # -pi ... +pi
    # değerlerini
    # 0 ... 2*pi aralığına getir
    angle = (
        angle +
        2 * np.pi
    ) % (2 * np.pi)   # 0° → 360°  aralığına getiriyoruz.
                      # Böylece yönleri histogramlara koymak daha kolay oluyor.
    return (
        magnitude,
        angle
    )   # hangi yönde ve ne kadar güçlü ?
        # Çünkü bir feature'ın çevresindeki şekli bu yön bilgileriyle tarif edeceğiz.


def create_gradient_descriptor(  # amaç: 24×24 piksellik görüntü parçasını sayılardan oluşan bir kimliğe dönüştürmek.
        patch,
        cells=4,    # 4 hücreye
        bins=8      #Her hücrede gradient yönlerini 8 gruba ayırıyoruz.
        # Patch'i 4×4'e bölüyoruz
):

    magnitude, angle = (
        calculate_local_gradients(
            patch
        )
    )

    height, width = patch.shape

    cell_height = (
        height // cells
    )

    cell_width = (
        width // cells
    )

    descriptor = []

    # ----------------------------------------------
    # Patch'i 4x4 bölgeye ayır
    # ----------------------------------------------

    for row in range(cells):

        for col in range(cells):

            y_start = (
                row *
                cell_height
            )

            x_start = (
                col *
                cell_width
            )

            # Son hücrelerde kalan
            # pikselleri de dahil et
            if row == cells - 1:
                y_end = height
            else:
                y_end = (
                    y_start +
                    cell_height
                )

            if col == cells - 1:
                x_end = width
            else:
                x_end = (
                    x_start +
                    cell_width
                )

            cell_magnitude = magnitude[
                y_start:y_end,
                x_start:x_end
            ]

            cell_angle = angle[
                y_start:y_end,
                x_start:x_end
            ]

            histogram = np.zeros(
                bins,
                dtype=np.float32   #  başlangıçta [0, 0, 0, 0, 0, 0, 0, 0] oluşturuyor
            )

            # --------------------------------------
            # Orientation histogram
            # --------------------------------------

            for y in range(
                cell_angle.shape[0]
            ):

                for x in range(
                    cell_angle.shape[1]
                ):

                    current_angle = (   # Sonra hücredeki her piksel için:
                        cell_angle[y, x]  # açıyı ve gücü alıyoruz
                    )

                    current_magnitude = (
                        cell_magnitude[y, x]
                    )

                    bin_index = int(
                        (
                            current_angle /
                            (2 * np.pi)
                        ) *
                        bins
                    )

                    if bin_index >= bins:
                        bin_index = bins - 1

                    histogram[
                        bin_index
                    ] += current_magnitude

            descriptor.extend(
                histogram
            )

    descriptor = np.array(
        descriptor,
        dtype=np.float32
    )

    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------

    norm = np.sqrt(
        np.sum(
            descriptor *
            descriptor
        )
    )

    descriptor = (
        descriptor /
        (norm + 1e-8)   # ışık değişimlerinin etkisini azaltmaya çalışıyoruz.
    )

    # Büyük değerleri bastır
    descriptor = np.clip(   # Çok büyük descriptor değerlerini maksimum 0.2 yapıyoruz.
        descriptor,         # Amaç tek bir çok güçlü gradientin bütün descriptorı domine etmesini azaltmak.
        0,
        0.2
    )

    # Tekrar normalize et
    norm = np.sqrt(
        np.sum(
            descriptor *
            descriptor
        )
    )

    descriptor = (
        descriptor /
        (norm + 1e-8)   # Sonra tekrar normalize ediyoruz.
    )

    return descriptor


def create_patch_descriptors(
        image,
        points,
        patch_size=24   # points Harris'in bize verdiği aday noktalar.
):

    radius = (
        patch_size // 2
    )

    descriptors = []
    valid_points = []

    height, width = (
        image.shape
    )

    for point in points:   #Her birini tek tek dolaşıyoruz

        x = int(
            point[0]
        )

        y = int(
            point[1]
        )   # Her noktanın x-y'sini alıyoruz

        # Patch görüntü sınırını
        # aşmasın
        if (
            x - radius < 0
            or
            x + radius >= width
            or
            y - radius < 0
            or
            y + radius >= height
        ):    # Kenardaki featureları eliyoruz. Biz bunun etrafından 24×24 kare almak istiyoruz.
              # Ama karenin yarısı fotoğrafın dışında kalacak. Patch'i tam alamıyorsam bu feature'ı kullanma

            continue

        patch = image[
            y - radius:y + radius,
            x - radius:x + radius
        ]    # Feature'ın etrafından  24 × 24  görüntü alıyoruz.

        descriptor = (
            create_gradient_descriptor(   # ile bunun 128 sayılık descriptorını oluşturuyoruz.
                patch,
                cells=4,
                bins=8
            )
        )

        descriptors.append(
            descriptor
        )

        valid_points.append(
            point
        )

    return (
        np.array(
            valid_points
        ),
        np.array(
            descriptors
        )
    )


def descriptor_distance(
        descriptor1,
        descriptor2
):

    difference = (
        descriptor1 -
        descriptor2   # İki descriptor'ın birbirine ne kadar benzediğini ölçüyor.
    )

    distance = np.sqrt(
        np.sum(
            difference *   # 128 sayıyı birbirinden çıkarıyoruz.
            difference    #Öklid mesafesi: Distance küçükse → descriptorlar benzer.
        )                 #  Distance büyükse → descriptorlar farklı.
    )

    return distance


def find_ratio_matches(
        descriptors1,                 # Fotoğraf 1'deki her feature'ı tek tek alıyoruz.
        descriptors2,                 # Fotoğraf 2'deki bütün featurelarla karşılaştırıyoruz.
        ratio_threshold=0.85           
):

    matches = []

    for i in range(
        len(descriptors1)
    ):

        distances = []

        for j in range(
            len(descriptors2)
        ):

            distance = (
                descriptor_distance(
                    descriptors1[i],
                    descriptors2[j]
                )
            )

            distances.append(
                (
                    distance,
                    j
                )
            )

        distances.sort(
            key=lambda item:
            item[0]
        )

        if len(distances) < 2:
            continue

        best_distance = (
            distances[0][0]
        )

        second_distance = (
            distances[1][0]
        )

        best_index = (
            distances[0][1]
        )

        ratio = (
            best_distance /
            (
                second_distance +
                1e-8
            )
        )

        if (
            ratio <
            ratio_threshold
        ):

            matches.append(
                (
                    i,
                    best_index,
                    best_distance,
                    ratio
                )
            )

    matches.sort(
        key=lambda match:
        match[3]
    )

    return matches