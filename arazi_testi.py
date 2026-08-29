import pathlib
import pycolmap
import open3d as o3d
import numpy as np


def main():
    # 1. Klasör Yapılandırması (Yeni arazi_img Klasörü)
    base_dir = pathlib.Path(__file__).parent
    image_dir = base_dir / "arazi_img"  # Fotoğrafları buraya koyduğun için güncellendi
    output_dir = base_dir / "arazi_cikti"
    output_dir.mkdir(exist_ok=True)

    database_path = output_dir / "arazi_database.db"

    # Fotoğraf Kontrolü
    uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    gorseller = []
    for ext in uzantilar:
        gorseller.extend(list(image_dir.glob(ext)))

    if not gorseller:
        print(f"⚠️ HATA: '{image_dir}' klasöründe drone fotoğrafı bulunamadı!")
        print("💡 Lütfen arazi fotoğraflarını 'arazi_img' klasörünün içine attığınızdan emin olun.")
        return

    print(f"🚁 {len(gorseller)} adet arazi/drone fotoğrafı işleniyor...")

    # Eski veritabanı kilitlenmelerini önlemek için silme denemesi
    if database_path.exists():
        try:
            database_path.unlink()
        except PermissionError:
            pass

    # 2. Öznitelik Çıkarımı ve Eşleştirme
    print("\n[1/4] Arazi üzerindeki kilit noktalar taranıyor...")
    try:
        pycolmap.extract_features(database_path=database_path, image_path=image_dir, max_num_threads=1)
    except TypeError:
        pycolmap.extract_features(database_path=database_path, image_path=image_dir)

    try:
        pycolmap.match_exhaustive(database_path=database_path, max_num_threads=1)
    except TypeError:
        pycolmap.match_exhaustive(database_path=database_path)

    # 3. SfM Rekonstrüksiyonu (Structure from Motion)
    print("\n[2/4] 3B Arazi Nokta Bulutu Hesaplaşıyor...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

    if reconstructions:
        if isinstance(reconstructions, dict):
            rec = max(reconstructions.values(), key=lambda r: r.num_points3D())
        else:
            rec = reconstructions[0]

        if rec.num_points3D() == 0:
            print(
                "\n⚠️ Fotoğraflar eşleştirilemedi. Lütfen veri setindeki tüm fotoğrafların klasörde olduğundan emin olun.")
            return

        print(f"\n✅ Arazi Nokta Bulutu Başarıyla Oluştu! ({rec.num_points3D()} Nokta Yakalandı)")

        ply_yolu = output_dir / "arazi_noktalari.ply"
        rec.export_PLY(str(ply_yolu))

        # 4. Arazi Yüzeyi (Mesh) Örme
        print("\n[3/4] Arazi Çukuru ve Topoğrafik Yüzey (Mesh) Örülüyor...")
        pcd = o3d.io.read_point_cloud(str(ply_yolu))

        if not pcd.is_empty():
            # Yüzey yönlerini hesapla
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)

            # Poisson algoritması ile arazi modelini kapla
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            # Kenarlardaki uçuk gürültü noktalarını temizle
            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, 0.08)
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            mesh_yolu = output_dir / "arazi_3d_cukur_modeli.ply"
            o3d.io.write_triangle_mesh(str(mesh_yolu), mesh)
            print(f"📁 3B Arazi Yüzey Dosyası Kaydedildi: {mesh_yolu}")

            # 5. Görselleştirme
            print("\n[4/4] 🖥️ 3B Arazi Modeli Açılıyor...")
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="3B Arazi ve Çukur Modeli",
                width=1024,
                height=768
            )
    else:
        print("\n⚠️ COLMAP arazi modelini oluşturamadı.")


if __name__ == "__main__":
    main()