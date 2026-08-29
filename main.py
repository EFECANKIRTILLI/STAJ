from PIL import Image
import numpy as np

from feature_detection import (
    calculate_harris_response,
    find_feature_points
)

from feature_matching import (
    create_patch_descriptors,
    find_ratio_matches
)

from geometry import (
    ransac_fundamental_matrix
)


def load_image(path):

    image = Image.open(
        path
    ).convert("L")

    image = np.array(
        image,
        dtype=np.float32
    )

    return image


def extract_features(
        image_path
):

    print(
        "\nGörüntü işleniyor:"
    )

    print(
        image_path
    )

    image = load_image(
        image_path
    )

    print(
        "Görüntü boyutu:",
        image.shape[1],
        "x",
        image.shape[0]
    )

    # ==================================================
    # HARRIS
    # ==================================================

    harris = (
        calculate_harris_response(
            image
        )
    )

    points = (
        find_feature_points(
            harris,
            max_points=500
        )
    )

    print(
        "Bulunan Harris noktası:",
        len(points)
    )

    # ==================================================
    # DESCRIPTOR
    # ==================================================

    valid_points, descriptors = (
        create_patch_descriptors(
            image,
            points,
            patch_size=24
        )
    )

    print(
        "Geçerli feature:",
        len(valid_points)
    )

    print(
        "Descriptor boyutu:",
        descriptors.shape
    )

    return (
        valid_points,
        descriptors
    )


def main():

    # ==================================================
    # ETH3D DATASET
    # ==================================================

    image1_path = (
        "sfm_data/"
        "1494491536630704618.png"
    )

    image2_path = (
        "sfm_data/"
        "1494491536704432618.png"
    )

    print(
        "=================================="
    )

    print(
        "CUSTOM SFM"
    )

    print(
        "=================================="
    )

    # ==================================================
    # FEATURE EXTRACTION
    # ==================================================

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

    # ==================================================
    # FEATURE MATCHING
    # ==================================================

    print(
        "\nFeature matching yapılıyor..."
    )

    matches = (
        find_ratio_matches(
            descriptors1,
            descriptors2,
            ratio_threshold=0.85
        )
    )

    print(
        "\nDescriptor eşleşmesi:",
        len(matches)
    )

    if len(matches) < 8:

        print(
            "Yeterli eşleşme yok."
        )

        return

    # ==================================================
    # MATCHED POINT ARRAYS
    # ==================================================

    matched_points1 = []

    matched_points2 = []

    for match in matches:

        index1 = match[0]
        index2 = match[1]

        point1 = points1[
            index1
        ]

        point2 = points2[
            index2
        ]

        matched_points1.append(
            [
                point1[0],
                point1[1]
            ]
        )

        matched_points2.append(
            [
                point2[0],
                point2[1]
            ]
        )

    matched_points1 = np.array(
        matched_points1,
        dtype=np.float64
    )

    matched_points2 = np.array(
        matched_points2,
        dtype=np.float64
    )

    # ==================================================
    # RANSAC
    # ==================================================

    print(
        "\n=================================="
    )

    print(
        "RANSAC BAŞLIYOR"
    )

    print(
        "=================================="
    )

    F, inlier_mask, errors = (
        ransac_fundamental_matrix(
            matched_points1,
            matched_points2,
            iterations=2000,
            threshold=1.5
        )
    )

    # ==================================================
    # RANSAC SONUÇLARI
    # ==================================================

    inlier_count = int(
        np.sum(
            inlier_mask
        )
    )

    outlier_count = (
        len(matches)
        -
        inlier_count
    )

    inlier_ratio = (
        inlier_count
        /
        len(matches)
    )

    print(
        "\n=================================="
    )

    print(
        "RANSAC TAMAMLANDI"
    )

    print(
        "=================================="
    )

    print(
        "Toplam eşleşme:",
        len(matches)
    )

    print(
        "Inlier:",
        inlier_count
    )

    print(
        "Outlier:",
        outlier_count
    )

    print(
        "Inlier oranı:",
        round(
            inlier_ratio * 100,
            2
        ),
        "%"
    )

    # ==================================================
    # FINAL FUNDAMENTAL MATRIX
    # ==================================================

    print(
        "\n=================================="
    )

    print(
        "FINAL FUNDAMENTAL MATRIX"
    )

    print(
        "=================================="
    )

    print(
        F
    )

    # ==================================================
    # ERROR STATISTICS
    # ==================================================

    inlier_errors = errors[
        inlier_mask
    ]

    print(
        "\nSampson Error:"
    )

    print(
        "Ortalama:",
        np.mean(
            inlier_errors
        )
    )

    print(
        "Median:",
        np.median(
            inlier_errors
        )
    )

    print(
        "Minimum:",
        np.min(
            inlier_errors
        )
    )

    print(
        "Maximum:",
        np.max(
            inlier_errors
        )
    )

    print(
        "\nBir sonraki aşama:"
    )

    print(
        "F -> E -> R,t -> Triangulation"
    )


if __name__ == "__main__":

    main()