import os
from typing import List, Set
import numpy as np
import open3d as o3d
from .octree import Octree


def get_points(path):
    points = np.loadtxt(path, dtype=float, usecols=(0, 1, 2)).tolist()
    return points


def save_planes(complete_segments: List[Set[Octree]], folder: str, extra: str = ""):
    results_folder = os.path.join(folder, 'OBRG')
    if 'OBRG' not in os.listdir(folder):
        os.mkdir(results_folder)
    for i, segment in enumerate(complete_segments):
        filename = os.path.join(results_folder, f'{extra}plane_{i}.txt')
        with open(filename, 'w') as ofile:
            for leaf in segment:
                for inlier in leaf.indices:
                    ofile.write(
                        f'{leaf.cloud[inlier][0]} {leaf.cloud[inlier][1]} {leaf.cloud[inlier][2]}\n')

def save_planes_pcd(complete_segments: List[Set[Octree]], folder: str, extra: str = ""):
    results_folder = os.path.join(folder, 'OBRG')
    if 'OBRG' not in os.listdir(folder):
        os.mkdir(results_folder)
    for i, segment in enumerate(complete_segments):
        filename = os.path.join(results_folder, f'{extra}plane_{i}.pcd')
        segment_color = np.random.rand(3)
        segment_cloud_list = []
        for leaf in segment:
            leaf_indices = np.ravel(leaf.indices)
            leaf_cloud = np.asarray(leaf.cloud)[leaf_indices]
            segment_cloud_list.append(leaf_cloud)
        segment_cloud = np.concatenate(segment_cloud_list, axis=0)

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(segment_cloud)
        cloud.colors = o3d.utility.Vector3dVector(np.tile(segment_color, (len(segment_cloud),1)))
        o3d.io.write_point_cloud(filename, cloud)

def save_time(time: float, pre: float, post: float,  folder: str, dataset_name: str, extra: str = ""):
    results_folder = os.path.join(folder, 'OBRG')
    if 'OBRG' not in os.listdir(folder):
        os.mkdir(results_folder)
    filename = os.path.join(results_folder, f'{extra}times-{dataset_name}.txt')
    print(f'saving time to {filename}')
    with open(filename, 'w') as ofile:
        ofile.write('pre pc post\n')
        ofile.write(f'{pre} {time} {post}')
