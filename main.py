import pathlib
import pycolmap
import open3d as o3d
import numpy as np


def main():
    base_dir = pathlib.Path(__file__).parent
    image_dir = base_dir / "images"
    output_dir = base_dir / "sfm_cikti"
    output_dir.mkdir(exist_ok=True)

    database_path = output_dir / "database.db"

    # 1. Fotoğraf kontrolü
    uzantilar = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    gorseller = []
    for ext in uzantilar:
        gorseller.extend(list(image_dir.glob(ext)))

    if not gorseller:
        print(f"⚠️ HATA: '{image_dir}' klasörü boş!")
        return

    print(f"📸 {len(gorseller)} adet 2D fotoğraf işleniyor...")

    if database_path.exists():
        try:
            database_path.unlink()
        except PermissionError:
            pass

    # 2. Özellik Çıkarımı ve Eşleştirme
    print("\n[1/4] Fotoğraflardaki ortak noktalar taranıyor...")
    try:
        pycolmap.extract_features(database_path=database_path, image_path=image_dir, max_num_threads=1)
    except TypeError:
        pycolmap.extract_features(database_path=database_path, image_path=image_dir)

    try:
        pycolmap.match_exhaustive(database_path=database_path, max_num_threads=1)
    except TypeError:
        pycolmap.match_exhaustive(database_path=database_path)

    # 3. SfM Rekonstrüksiyonu
    print("\n[2/4] 3B nokta bulutu hesaplanıyor (Sparse Cloud)...")
    reconstructions = pycolmap.incremental_mapping(database_path, image_dir, output_dir)

    # 4. Model Kontrolü
    if reconstructions:
        if isinstance(reconstructions, dict):
            rec = max(reconstructions.values(), key=lambda r: r.num_points3D())
        else:
            rec = reconstructions[0]

        # Nokta sayısı kontrolü (COLMAP başlangıç çifti bulamadıysa nokta üretemez)
        if rec.num_points3D() == 0:
            print("\n⚠️ HATA: COLMAP fotoğraflar arasında yeterli ortak nokta bulamadı!")
            print("💡 Çözüm: Fotoğraf sayısını artırın ve çekimler arası %60-70 örtüşme olmasına dikkat edin.")
            return

        print(f"\n✅ COLMAP Model Oluşturdu! Toplam {rec.num_points3D()} nokta yakalandı.")

        ply_yolu = output_dir / "obje_noktalari.ply"
        rec.export_PLY(str(ply_yolu))

        # 5. Open3D ile Yüzey (Mesh) Örme Adımı
        print("\n[3/4] Noktalar birleştirilip KATI YÜZEY (Mesh) örülüyor...")
        pcd = o3d.io.read_point_cloud(str(ply_yolu))

        if not pcd.is_empty() and len(pcd.points) > 10:
            # Yüzey yönlerini (normalleri) hesapla
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(k=15)

            # Poisson algoritması ile mesh üret
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

            # Open3D NumPy filtreleme (Vector1dVector hatasını çözen kısım)
            densities_np = np.asarray(densities)
            density_threshold = np.quantile(densities_np, 0.05)  # En düşük %5 yoğunluktaki gürültüleri temizle
            vertices_to_remove = densities_np < density_threshold
            mesh.remove_vertices_by_mask(vertices_to_remove)

            mesh_yolu = output_dir / "obje_kati_model.ply"
            o3d.io.write_triangle_mesh(str(mesh_yolu), mesh)
            print(f"📁 Katı 3B Yüzey Kaydedildi: {mesh_yolu}")

            print("\n[4/4] 🖥️ Katı Yüzey Modeli Ekrana Basılıyor...")
            mesh.compute_vertex_normals()
            o3d.visualization.draw_geometries(
                [mesh],
                window_name="Noktaların Yüzeye Dönüştürülmüş Hali (Mesh)",
                width=1024,
                height=768,
                mesh_show_wireframe=True
            )
        else:
            print("⚠️ Oluşturulan PLY dosyasında mesh örmek için yeterli nokta bulunamadı.")
    else:
        print("\n⚠️ COLMAP model oluşturamadı. Fotoğraflarınız birbirini yeterince kapsamıyor olabilir.")


if __name__ == "__main__":
    main()