import pathlib
import shutil
import torch
import pycolmap
import open3d as o3d
import numpy as np

from hloc import extract_features, match_features, reconstruction


def generate_clean_pairs(output_path, image_dir):
    """
    HLOC parser hatasını engellemek için sadece
    2 sütunlu (resim1 resim2) temiz pairs dosyası oluşturur.
    """
    images = sorted([p.name for p in image_dir.iterdir() if p.is_file()])
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((images[i], images[j]))

    with open(output_path, "w", encoding="utf-8") as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")

    print(f"✅ BİZİM FONKSİYON ÇALIŞTI: {len(pairs)} adet temiz çift yazıldı.")


def main():
    # Türkçe karaktersiz güvenli kök dizin
    guvenli_alan = pathlib.Path("C:/Users/canem/cukur_islem")

    # Resimlerinin bulunduğu ham klasör
    raw_image_dir = guvenli_alan / "denemckr"

    # OpenCV ve HLOC için temiz geçici okuma klasörü
    safe_image_dir = guvenli_alan / "temp_img"

    if safe_image_dir.exists():
        shutil.rmtree(safe_image_dir)
    safe_image_dir.mkdir(parents=True, exist_ok=True)

    uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    gorseller = []
    for ext in uzantilar:
        gorseller.extend(list(raw_image_dir.glob(ext)))

    if not gorseller:
        print(f"⚠️ HATA: '{raw_image_dir}' klasöründe fotoğraf bulunamadı! Lütfen klasör yolunu kontrol et.")
        return

    print(f"📁 '{raw_image_dir}' dizinindeki {len(gorseller)} adet fotoğraf işleme hazırlanıyor...")

    # WhatsApp isimlerindeki boşluk/parantez sorununu engellemek için img_001.jpg gibi standart isimlerle kopyala
    for idx, img in enumerate(gorseller, start=1):
        yeni_ad = f"img_{idx:03d}{img.suffix.lower()}"
        shutil.copy(img, safe_image_dir / yeni_ad)

    # Çıktı Klasörü (C:/Users/canem/cukur_islem/sfm_cikti)
    output_dir = guvenli_alan / "sfm_cikti"
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_dir = output_dir / "sfm"
    features_path = output_dir / "features.h5"
    pairs_path = output_dir / "pairs.txt"
    matches_path = output_dir / "matches.h5"

    # Eski kalıntıları temizle
    if pairs_path.exists():
        pairs_path.unlink()
    if features_path.exists():
        features_path.unlink()

    print(f"🔥 [AI Fotogrametri] Çukur rekonstrüksiyonu başlatılıyor...")

    # 1. SuperPoint
    print("\n[1/4] SuperPoint (AI) ile kilit noktalar taranıyor...")
    feature_conf = extract_features.confs['superpoint_max']
    extract_features.main(feature_conf, safe_image_dir, feature_path=features_path)

    # 1.5 Temiz Çift Oluşturma
    print("\n[Eşleştirme Listesi] Fotoğraf çiftleri listeleniyor...")
    generate_clean_pairs(pairs_path, safe_image_dir)

    # 2. LightGlue
    print("\n[2/4] LightGlue (AI) ile fotoğraflar eşleştiriliyor...")
    matcher_conf = match_features.confs['superpoint+lightglue']
    match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path, overwrite=True)

    # 3. COLMAP
    print("\n[3/4] 3B Nokta Bulutu Hesaplaması Yapılıyor...")
    model = reconstruction.main(
        sfm_dir,
        safe_image_dir,
        pairs_path,
        features_path,
        matches_path
    )

    if model and model.num_points3D() > 0:
        print(f"\n✅ AI Rekonstrüksiyon Başarılı! Toplam {model.num_points3D()} nokta yakalandı.")

        ply_yolu = output_dir / "cukur_noktalari.ply"
        model.export_PLY(str(ply_yolu))

        # 4. Open3D Katı Yüzey Örme
        print("\n[4/4] Noktalar birleştirilip PÜRÜZSÜZ KATI YÜZEY örülüyor...")
        pcd = o3d.io.read_point_cloud(str(ply_yolu))

        if not pcd.is_empty():
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=50))
            pcd.orient_normals_consistent_tangent_plane(k=20)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9, scale=1.1)

            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, 0.12)
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            mesh_yolu = output_dir / "cukur_kati_model.ply"
            o3d.io.write_triangle_mesh(str(mesh_yolu), mesh)
            print(f"📁 Kaydedildi: {mesh_yolu}")

            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="AI Destekli Çukur 3B Modeli",
                width=1024,
                height=768,
                mesh_show_back_face=True
            )
    else:
        print("\n⚠️ Model oluşturulamadı.")


if __name__ == "__main__":
    main()