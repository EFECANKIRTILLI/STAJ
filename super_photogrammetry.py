import pathlib
import shutil
import torch
import pycolmap
import open3d as o3d
import numpy as np

# HLOC Importları
from hloc import extract_features, match_features, pairs_from_exhaustive, reconstruction


def main():
    base_dir = pathlib.Path(__file__).parent

    # Türkçe karakter sorununu aşmak için resimleri güvenli dizinden okuyoruz
    raw_image_dir = base_dir / "denemckr"
    safe_image_dir = pathlib.Path("C:/cukur_temp_img")

    if safe_image_dir.exists():
        shutil.rmtree(safe_image_dir)
    safe_image_dir.mkdir(parents=True, exist_ok=True)

    # Resimleri kopyala
    uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    gorseller = []
    for ext in uzantilar:
        gorseller.extend(list(raw_image_dir.glob(ext)))

    if not gorseller:
        print(f"⚠️ HATA: '{raw_image_dir}' klasöründe fotoğraf bulunamadı!")
        return

    print(f"📁 {len(gorseller)} adet fotoğraf güvenli okuma dizinine aktarılıyor...")
    for img in gorseller:
        shutil.copy(img, safe_image_dir / img.name)

    output_dir = pathlib.Path("C:/cukur_sfm_cikti")
    output_dir.mkdir(parents=True, exist_ok=True)

    sfm_dir = output_dir / "sfm"
    features_path = output_dir / "features.h5"
    pairs_path = output_dir / "pairs.txt"
    matches_path = output_dir / "matches.h5"

    print(f"🔥 [AI Fotogrametri] Yapay zeka ile 3B rekonstrüksiyon başlatılıyor...")

    # 1. SuperPoint
    print("\n[1/4] SuperPoint (AI) ile kilit noktalar taranıyor...")
    feature_conf = extract_features.confs['superpoint_max']
    extract_features.main(feature_conf, safe_image_dir, feature_path=features_path)

    # 1.5 Fotoğraf Çiftlerinin Oluşturulması
    print("\n[Eşleştirme Listesi] Fotoğraf çiftleri listeleniyor...")
    pairs_from_exhaustive.main(pairs_path, features=features_path)

    # 2. LightGlue (Parametre isimleri HLOC standartlarına göre düzeltildi)
    print("\n[2/4] LightGlue (AI) ile fotoğraflar eşleştiriliyor...")
    matcher_conf = match_features.confs['superpoint+lightglue']
    match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path)

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

        ply_yolu = output_dir / "ai_model_noktalari.ply"
        model.export_PLY(str(ply_yolu))

        # 4. Open3D Katı Model
        print("\n[4/4] Noktalar birleştirilip PÜRÜZSÜZ KATI YÜZEY örülüyor...")
        pcd = o3d.io.read_point_cloud(str(ply_yolu))

        if not pcd.is_empty():
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)

            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, 0.06)
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            mesh_yolu = output_dir / "en_iyi_kati_model.ply"
            o3d.io.write_triangle_mesh(str(mesh_yolu), mesh)
            print(f"📁 Kaydedildi: {mesh_yolu}")

            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="AI Destekli En İyi 3B Model",
                width=1024,
                height=768
            )
    else:
        print("\n⚠️ Model oluşturulamadı.")


if __name__ == "__main__":
    main()