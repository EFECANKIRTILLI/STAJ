import numpy as np


def normalize_points(points):
    """
    2D noktaları normalize eder.

    1. Noktaların merkezini orijine taşır.
    2. Ortalama uzaklığı sqrt(2) yapar.

    Bu işlem 8-point algoritmasının
    sayısal kararlılığını artırır.
    """

    points = np.asarray(
        points,
        dtype=np.float64    # Bunları NumPy dizisine çeviriyoruz ki matematiksel işlemleri rahat yapabilelim.
    )                       # float64 da daha hassas ondalıklı hesap yapmamızı sağlıyor.

    # Noktaların merkezi
    mean_x = np.mean(points[:, 0])   #Burada bütün noktaların merkezini buluyoruz.
    mean_y = np.mean(points[:, 1])

    centered = points.copy() # Orijinal noktaları değiştirmemek için bir kopya oluştur

    centered[:, 0] -= mean_x  # noktaların merkezini (0,0)'a taşı.
    centered[:, 1] -= mean_y

    # Merkeze uzaklıklar
    distances = np.sqrt(     # Her noktanın yeni merkezimiz olan (0,0)'a uzaklığını hesaplıyor.
        centered[:, 0] ** 2 +
        centered[:, 1] ** 2
    )

    mean_distance = np.mean(
        distances               # Bütün bu uzaklıkların ortalamasını alıyoruz.
    )                           # Noktaları Fundamental Matrix hesabının daha kararlı  yapılabilmesi için düzenliyoruz.

    # Ortalama uzaklığı sqrt(2) yap
    scale = (
        np.sqrt(2)
        /
        (mean_distance + 1e-12)   # Burada amaç noktaları uygun bir büyüklüğe ölçeklemek.
    )                             # Noktaların ölçeğini standartlaştırmak için
                                  # gerekli scale değerini hesaplıyoruz.

    # Normalizasyon matrisi
    T = np.array(
        [
            [
                scale,
                0,
                -scale * mean_x
            ],

            [
                0,
                scale,
                -scale * mean_y
            ],

            [
                0,
                0,
                1
            ]
        ],
        dtype=np.float64
    )                            # Az önce yaptığımız merkezleme ve ölçekleme işlemlerini tek bir matriste topluyoruz.

    # Homojen koordinatlar
    homogeneous = np.column_stack(
        (
            points,
            np.ones(len(points))         # 3x3 dönüşüm matrisiyle işlem yapabilmek için
        )                                # (x,y) noktalarını (x,y,1) homojen koordinatına çeviriyoruz.
    )

    normalized_h = (
        T @ homogeneous.T
    ).T

    normalized_points = (
        normalized_h[:, :2]         # Homojen koordinattaki son değeri çıkarıp
                                    # tekrar sadece (x,y) koordinatlarını alıyoruz.
    )

    return (
        normalized_points,
        T
    )


def compute_fundamental_matrix(
        points1,     # 1 ve 2 . fotoğraftaki eşleşmiş noktalar.
        points2
):
    """
    Normalize edilmiş 8-point algoritması
    kullanarak Fundamental Matrix hesaplar.
    """

    points1 = np.asarray(
        points1,
        dtype=np.float64
    )

    points2 = np.asarray(
        points2,
        dtype=np.float64
    )          # İki fotoğraftaki eşleşmiş noktaları
             # matematiksel işlemler için NumPy dizisine çeviriyoruz.

    if len(points1) < 8:

        raise ValueError(
            "Fundamental Matrix için "
            "en az 8 eşleşme gerekir."
        )

    # ==================================================
    # 1. NORMALIZATION
    # ==================================================

    norm1, T1 = normalize_points(
        points1
    )

    norm2, T2 = normalize_points(
        points2
    )   # İki fotoğraftaki noktaları Fundamental Matrix hesabının daha kararlı olması için normalize ediyoruz.
            # aslında buyuk koordinattaki sayıları daha basit küçük hale getiriyor hesap daha iyi yapılıyor. 
    # ==================================================
    # 2. A MATRİSİ
    # ==================================================

    A = []

    for i in range(
        len(norm1)
    ):

        x1 = norm1[i, 0]
        y1 = norm1[i, 1]

        x2 = norm2[i, 0]
        y2 = norm2[i, 1]

        row = [
            x2 * x1,
            x2 * y1,
            x2,

            y2 * x1,
            y2 * y1,
            y2,

            x1,
            y1,

            1
        ]

        A.append(
            row
        )

    A = np.array(
        A,
        dtype=np.float64
    )

    # ==================================================
    # 3. Af = 0 SİSTEMİ
    # ==================================================

    U, S, Vt = np.linalg.svd(
        A
    )

    # En küçük singular value'a
    # karşılık gelen vektör
    f = Vt[-1]

    F = f.reshape(
        3,
        3
    )

    # ==================================================
    # 4. RANK 2 ZORLAMASI
    # ==================================================

    U_f, S_f, Vt_f = np.linalg.svd(
        F
    )

    # Fundamental Matrix rank 2 olmalı
    S_f[-1] = 0

    F_rank2 = (
        U_f
        @ np.diag(S_f)
        @ Vt_f
    )

    # ==================================================
    # 5. DENORMALIZATION
    # ==================================================

    F_final = (
        T2.T
        @ F_rank2
        @ T1
    )

    # Ölçeği normalize et
    norm = np.linalg.norm(
        F_final
    )

    if norm > 0:

        F_final = (
            F_final /
            norm
        )

    return F_final


def calculate_sampson_errors(
        F,
        points1,
        points2
):
    """
    Eşleşmelerin Fundamental Matrix'e
    ne kadar uyduğunu Sampson error
    ile hesaplar.

    Küçük değer = iyi geometrik eşleşme.
    Büyük değer = muhtemel outlier.
    """

    points1 = np.asarray(
        points1,
        dtype=np.float64
    )

    points2 = np.asarray(
        points2,
        dtype=np.float64
    )

    # Homojen koordinatlar
    x1 = np.column_stack(
        (
            points1,
            np.ones(len(points1))
        )
    )

    x2 = np.column_stack(
        (
            points2,
            np.ones(len(points2))
        )
    )

    # F * x1
    Fx1 = (
        F @ x1.T
    ).T

    # F^T * x2
    Ftx2 = (
        F.T @ x2.T
    ).T

    # x2^T * F * x1
    numerator = np.sum(
        x2 * Fx1,
        axis=1
    )

    numerator = (
        numerator ** 2
    )

    denominator = (
        Fx1[:, 0] ** 2
        +
        Fx1[:, 1] ** 2
        +
        Ftx2[:, 0] ** 2
        +
        Ftx2[:, 1] ** 2
    )

    errors = (
        numerator /
        (denominator + 1e-12)
    )

    return errors


def ransac_fundamental_matrix(
        points1,
        points2,
        iterations=2000,
        threshold=1.5,
        random_seed=42
):
    """
    RANSAC kullanarak yanlış eşleşmeleri
    temizler ve en iyi Fundamental Matrix'i
    hesaplar.

    Her iterasyonda:
    1. Rastgele 8 eşleşme seçilir.
    2. Fundamental Matrix hesaplanır.
    3. Tüm eşleşmeler test edilir.
    4. Inlier sayısı hesaplanır.
    """

    points1 = np.asarray(
        points1,
        dtype=np.float64
    )

    points2 = np.asarray(
        points2,
        dtype=np.float64
    )

    number_of_points = len(
        points1
    )

    if number_of_points < 8:

        raise ValueError(
            "RANSAC için en az "
            "8 eşleşme gerekir."
        )

    rng = np.random.default_rng(
        random_seed
    )

    best_F = None

    best_inlier_mask = None

    best_inlier_count = 0

    best_mean_error = float(
        "inf"
    )

    # Threshold karesi
    threshold_squared = (
        threshold ** 2
    )

    # ==================================================
    # RANSAC ITERATIONS
    # ==================================================

    for iteration in range(
        iterations
    ):

        # Rastgele 8 farklı nokta
        sample_indices = rng.choice(
            number_of_points,
            size=8,
            replace=False
        )

        sample1 = points1[
            sample_indices
        ]

        sample2 = points2[
            sample_indices
        ]

        try:

            F_candidate = (
                compute_fundamental_matrix(
                    sample1,
                    sample2
                )
            )

        except np.linalg.LinAlgError:

            continue

        # Bütün noktaların hatasını hesapla
        errors = calculate_sampson_errors(
            F_candidate,
            points1,
            points2
        )

        # Threshold altındakiler inlier
        inlier_mask = (
            errors <
            threshold_squared
        )

        inlier_count = int(
            np.sum(inlier_mask)
        )

        if inlier_count > 0:

            mean_error = np.mean(
                errors[inlier_mask]
            )

        else:

            mean_error = float(
                "inf"
            )

        # ==================================================
        # EN İYİ MODELİ SEÇ
        # ==================================================

        if (
            inlier_count >
            best_inlier_count
        ):

            best_inlier_count = (
                inlier_count
            )

            best_inlier_mask = (
                inlier_mask
            )

            best_F = (
                F_candidate
            )

            best_mean_error = (
                mean_error
            )

        elif (
            inlier_count ==
            best_inlier_count
            and
            mean_error <
            best_mean_error
        ):

            best_inlier_mask = (
                inlier_mask
            )

            best_F = (
                F_candidate
            )

            best_mean_error = (
                mean_error
            )

    # ==================================================
    # RANSAC SONUCU KONTROL
    # ==================================================

    if (
        best_inlier_mask is None
        or
        best_inlier_count < 8
    ):

        raise RuntimeError(
            "RANSAC yeterli inlier bulamadı."
        )

    # ==================================================
    # FINAL F
    #
    # En iyi RANSAC modelindeki bütün
    # inlier noktaları kullanılarak F tekrar
    # hesaplanır.
    # ==================================================

    inlier_points1 = points1[
        best_inlier_mask
    ]

    inlier_points2 = points2[
        best_inlier_mask
    ]

    final_F = (
        compute_fundamental_matrix(
            inlier_points1,
            inlier_points2
        )
    )

    # Final model ile tekrar hata hesapla
    final_errors = (
        calculate_sampson_errors(
            final_F,
            points1,
            points2
        )
    )

    final_inlier_mask = (
        final_errors <
        threshold_squared
    )

    # Eğer final model sonrası hala yeterli
    # nokta varsa bir kez daha refine et
    if np.sum(final_inlier_mask) >= 8:

        final_F = (
            compute_fundamental_matrix(
                points1[
                    final_inlier_mask
                ],
                points2[
                    final_inlier_mask
                ]
            )
        )

        final_errors = (
            calculate_sampson_errors(
                final_F,
                points1,
                points2
            )
        )

        final_inlier_mask = (
            final_errors <
            threshold_squared
        )

    return (
        final_F,
        final_inlier_mask,
        final_errors
    )