# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma

import os
import numpy as np
import vnoise
import pyrender
from binocmesher import BinocMesher
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import subprocess

noise = vnoise.Noise()

def f(XYZ):
    scale = 10
    height = 5
    h = noise.noise2(XYZ[:, 0] / scale, XYZ[:, 1] / scale, grid_mode=False, octaves=4) * height
    return XYZ[:, 2] - h

cam_poses, Ks, Hs, Ws, Ts = [], [], [], [], []

fx = 2000
fy = 2000
W = 1280
H = 720

fov_x = 2 * np.arctan(W / (2 * fx))
fov_y = 2 * np.arctan(H / (2 * fy))

for i in range(480):
    cam_poses.append(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, i * 0.3],
        [0, -1, 0, 3],
        [0, 0, 0, 1],
    ]))
    Ks.append(np.array([
        [fx, 0, W/2],
        [0, fy, H/2],
        [0, 0, 1]
    ]))
    Hs.append(H)
    Ws.append(W)
    Ts.append((0.5+i) / 24)

bounds = [-1e3, 1e3, -1e3, 1e3, -10, 10]


for i in tqdm(range(480)):
    # Only the first call compute the 4D mesh, the following calls are slicing the 4D mesh at different time
    mesher = BinocMesher((cam_poses, Ks, Hs, Ws, Ts), bounds=bounds, slicing_time=(0.5+i) / 24, pixels_per_cube=30, path=Path("logs"))
    meshes, in_view_tags = mesher([f])
    mesh = meshes[0]
    scene = pyrender.Scene(bg_color=[200, 200, 200, 255], ambient_light=[0.6, 0.6, 0.6])
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],  # gray clay
        metallicFactor=0.0,
        roughnessFactor=1.0
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
    scene.add(mesh_pr)
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=W/H)
    cam_pose = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, i * 0.3],
        [0, 1, 0, 3],
        [0, 0, 0, 1],
    ])
    scene.add(camera, pose=cam_pose)
    light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light1, pose=cam_pose)
    r = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
    color, depth = r.render(scene)
    r.delete()
    os.makedirs("results", exist_ok=True)
    Image.fromarray(color).save(f"results/demo_{i:03d}.png")

cmd = f"ffmpeg -framerate 24 -i results/demo_%03d.png -c:v libx264 -pix_fmt yuv420p results/demo.mp4"
subprocess.call(cmd, shell=True)