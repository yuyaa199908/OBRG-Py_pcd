import os
import numpy as np
import open3d as o3d

# ディレクトリ内のファイル一覧を取得
directory = "/home/aichi2204/Documents/bkl2go/20240404-ib-w-out/aichi-20240404-ib-w-out_mini/voxel/OBRG"
file_list = os.listdir(directory)
file_list = [f for f in file_list if f.startswith("aichi-20240404-ib-w-out_mini.pcdplane_") and f.endswith(".txt")]

# 各ファイルを処理
for file_name in file_list:
    # ファイルパスを構築
    file_path = os.path.join(directory, file_name)
    
    # ファイルからデータを読み込む
    data = np.loadtxt(file_path)
    print(data.shape)
    # 2次元配列から点群データを作成する
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data[:,0:3])
    
    # PCDファイルに保存する
    output_file_path = os.path.splitext(file_path)[0] + ".pcd"
    o3d.io.write_point_cloud(output_file_path, point_cloud)
    print(f"Converted {file_name} to {output_file_path}")