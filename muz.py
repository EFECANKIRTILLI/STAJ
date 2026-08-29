import pathlib
import shutil
import torch
import pycolmap
import open3d as o3d
import numpy as np

from hloc import extract_features, match_features, reconstruction


def generate_all_pairs(output_path, image_dir):
    """16 resim için 120 adet %100 kapsayıcı çift oluşturur."""
    images = sorted([p.name for p in image_dir.iterdir() if p.is_file()])
    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            pairs.append((images[i], images[j]))

    with open(output_path, "w", encoding="utf-8") as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")


def main():
    guvenli_alan = pathlib.Path("C:/Users/canem/cukur_islem")
    raw_image_dir = guvenli_alan / "muz"
    safe_image_dir = guvenli_alan / "temp_img_muz"

    if safe_image_dir.exists():
        shutil.rmtree(safe_image_dir)
    safe_image_dir.mkdir(parents=True, exist_ok=True)

    uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    gorseller = []
    for ext in uzantilar:
        gorseller.extend(list(raw_image_dir.glob(ext)))

    if not gorseller:
        print(f"⚠️ HATA: '{raw_image_dir}' klasöründe fotoğraf bulunamadı!")
        return

    print(f"📁 Muz setinden {len(gorseller)} adet fotoğraf işleniyor...")

    for idx, img in enumerate(gorseller, start=1):
        yeni_ad = f"img_{idx:03d}{img.suffix.lower()}"
        shutil.copy(img, safe_image_dir / yeni_ad)

    output_dir = guvenli_alan / "muz_sfm_cikti"
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_dir = output_dir / "sfm"
    features_path = output_dir / "features.h5"
    pairs_path = output_dir / "pairs.txt"
    matches_path = output_dir / "matches.h5"

    if pairs_path.exists():
        pairs_path.unlink()
    if features_path.exists():
        features_path.unlink()

    print(f"🔥 [AI Fotogrametri] Renkli Muz 3B Rekonstrüksiyonu Başlatılıyor...")

    # 1. SuperPoint
    feature_conf = extract_features.confs["superpoint_max"]
    extract_features.main(
        feature_conf, safe_image_dir, feature_path=features_path
    )

    # 1.5 Çiftler (120 adet)
    generate_all_pairs(pairs_path, safe_image_dir)

    # 2. LightGlue
    matcher_conf = match_features.confs["superpoint+lightglue"]
    match_features.main(
        matcher_conf,
        pairs_path,
        features=features_path,
        matches=matches_path,
        overwrite=True,
    )

    # 3. COLMAP
    model = reconstruction.main(
        sfm_dir, safe_image_dir, pairs_path, features_path, matches_path
    )

    if model and model.num_points3D() > 0:
        print(
            f"\n✅ AI Rekonstrüksiyon Başarılı! Toplam {model.num_points3D()} renkli nokta yakalandı."
        )

        ply_yolu = output_dir / "muz_renkli_noktalar.ply"
        model.export_PLY(str(ply_yolu))

        # 4. Renkli Yüzey Örme (Texture Giydirme)
        print("\n[4/4] Gerçek Fotoğraf Renkleriyle Katı Yüzey Örülüyor...")
        pcd = o3d.io.read_point_cloud(str(ply_yolu))

        if not pcd.is_empty():
            # Temizlik
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=30, std_ratio=2.2
            )
            pcd = pcd.select_by_index(ind)

            # Normalleri hesapla
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=0.15, max_nn=50
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=20)

            # Poisson Katı Yüzey Oluşturma
            mesh, densities = (
                o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=11, scale=1.1, linear_fit=True
                )
            )

            # Uçlardaki zayıf noktaları temizle (Sapın kopmasını engeller)
            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, 0.03)
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            # *** RENK GİYDİRME (TEXTURE MAPPING): Nokta Bulutundaki Gerçek Renkleri Yüzeye Aktar ***
            mesh_kdtree = o3d.geometry.KDTreeFlann(pcd)
            mesh_colors = []
            pcd_colors = np.asarray(pcd.colors)
            mesh_vertices = np.asarray(mesh.vertices)

            for v in mesh_vertices:
                [_, idx, _] = mesh_kdtree.search_knn_vector_3d(v, 1)
                mesh_colors.append(pcd_colors[idx[0]])

            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.array(mesh_colors)
            )

            # Modeli Kaydet
            mesh_yolu = output_dir / "muz_renkli_kati_model.ply"
            o3d.io.write_triangle_mesh(str(mesh_yolu), mesh)
            print(f"📁 Renkli model kaydedildi: {mesh_yolu}")

            mesh.compute_vertex_normals()

            # Ekrana Renkli Olarak Getir
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="Attığın Görseldeki Gibi Renkli 3B Muz Modeli",
                width=1024,
                height=768,
                mesh_show_back_face=True,
            )
    else:
        print("\n⚠️ Model oluşturulamadı.")


if __name__ == "__main__":
    main()