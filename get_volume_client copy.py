#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import argparse
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser

import matplotlib.pyplot as pl
import open3d as o3d

def find_alpha(pcd):
    left = 1.0
    right = 0.1
    mesh = pcd
    while 1:
        if left-right < 0.2: break
        a = (left+right)/2
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha=a)
        if mesh.is_watertight():
            left = (left+right)/2
        else:
            right = (left+right)/2
    return mesh

def pick_points(pcd):
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points(),vis.get_cropped_geometry()

def get_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))


def pct_process():
    pct = o3d.io.read_point_cloud('scene.ply')
    pts,pct = pick_points(pct)
    if len(pts)%2 == 1:
        pts = pts[:-1]
    result = [pct]
    for i in range(0,6):
        if len(pts) <= 2*i:
            result.append(-1)
        else:
            pct_distance = get_distance(pct.points[pts[2*i]],pct.points[pts[(2*i)+1]])
            result.append(pct_distance)
    return result

def get_mesh_from_pct(pct,method):
    if method.method == 'auto alpha':
        mesh = find_alpha(pct)
    elif method.method == 'alpha':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pct, alpha=method.alpha)
    elif method.method == 'convex hull':
        mesh,_ = pct.compute_convex_hull()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    if not mesh.is_watertight():
        return mesh, "Mesh is not watertight"
    return mesh, "OK"

def get_volume_with_mesh(d1,d2,d3,d4,d5,d6,r1,r2,r3,r4,r5,r6,mesh):
    ratio = 0
    pts_distance = [d1,d2,d3,d4,d5,d6]
    real_distance = [r1,r2,r3,r4,r5,r6]
    k = 0
    for i in range(0,len(pts_distance)):
        if pts_distance[i] == -1 or real_distance[i] == -1 or real_distance[i] == 0:
            break
        ratio += real_distance[i]/pts_distance[i]
        k += 1
    if k == 0:
        return 1, 0, 0
    ratio/=k
    if not mesh.is_watertight():
        print("Mesh is not watertight")
        return ratio, 0, 0
    mesh_volume = mesh.get_volume()
    volume = mesh_volume*ratio
    return ratio, mesh_volume, volume

def get_volume_from_pct(d1,d2,d3,d4,d5,d6,r1,r2,r3,r4,r5,r6, pct,method):
    mesh, error = get_mesh_from_pct(pct,method)
    if error:
        return 1, 0, 0, error
    ratio, mesh_volume, volume = get_volume_with_mesh(d1,d2,d3,d4,d5,d6,r1,r2,r3,r4,r5,r6,mesh)
    return ratio, mesh_volume, volume, "OK"

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser_weights = parser.add_mutually_exclusive_group()
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                                default = "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
                                choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')
    parser.prog = 'mast3r demo'
    return parser

with gradio.Blocks(title="MASt3R Demo") as demo:
    # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
    scene = gradio.State(None)
    pct = gradio.State(None)
    mesh = gradio.State(None)
    gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
    with gradio.Column():
        inputfiles = gradio.File(file_count="multiple")
        with gradio.Row():
            # adjust the confidence threshold
            min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
            # adjust the camera size in the output pointcloud
            cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
            TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
        with gradio.Row():
            as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
            # two post process implemented
            mask_sky = gradio.Checkbox(value=False, label="Mask sky")
            clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
            transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

        outmodel = gradio.Model3D(label="preview")
        

        pct_process_btn = gradio.Button("Start Picking Points")

        help = gradio.Markdown(
                                    """ - 1)Cop the point cloud: [K] drag a box, [C] accept the change, leave only points inside the box
                                        - 2)Pick at least three correspondences using [shift + left click]
                                        - 3)Press [shift + right click] to undo point picking
                                        - 4)After picking points, press 'Q' to close the window
                                        """
        )

        with gradio.Row():
            d1 = gradio.Number(label="Distance 1", interactive=False, value = -1)
            d2 = gradio.Number(label="Distance 2", interactive=False, value = -1)
            d3 = gradio.Number(label="Distance 3", interactive=False, value = -1)
            d4 = gradio.Number(label="Distance 4", interactive=False, value = -1)
            d5 = gradio.Number(label="Distance 5", interactive=False, value = -1)
            d6 = gradio.Number(label="Distance 6", interactive=False, value = -1)

        with gradio.Row():
            rd1 = gradio.Number(label="Real Dist 1", interactive = True, value = -1)
            rd2 = gradio.Number(label="Real Dist 2", interactive = True, value = -1)
            rd3 = gradio.Number(label="Real Dist 3", interactive = True, value = -1)
            rd4 = gradio.Number(label="Real Dist 4", interactive = True, value = -1)
            rd5 = gradio.Number(label="Real Dist 5", interactive = True, value = -1)
            rd6 = gradio.Number(label="Real Dist 6", interactive = True, value = -1)

        volume_btn = gradio.Button("Calculate Volume")

        with gradio.Row():
            method_name = gradio.Dropdown(["auto alpha", "alpha", "convex hull"],
                            value='convex hull', label="Mesh Method",
                            info="Method to create the mesh from the point cloud")
            alpha = gradio.Slider(label="Alpha", value=0.5, minimum=0.0, maximum=2.0, step=0.01)

        with gradio.Row():
            ratio = gradio.Number(label="Ratio",info="the ratio of the real volume to the estimated volume", value = 1)
            pct_volume = gradio.Number(label="Relative Volume",info="the relative volume of the mesh", interactive=False, value = 0)
            volume = gradio.Number(label="Volume",info="the real volume", interactive=False, value = 0)

        with gradio.Row():
            result = gradio.Textbox(label="Result", value="")
            exit = gradio.Button("Exit Mast3r")
        # events
        
        pct_process_btn.click(fn=pct_process,
                        inputs=[],
                        outputs=[pct, d1,d2,d3,d4,d5,d6])
        
        method_name.change(fn=get_method, inputs=[method_name, alpha], outputs=[method])
        alpha.change(fn=get_method, inputs=[method_name, alpha], outputs=[method])

        volume_btn.click(fn=get_volume_from_pct, inputs=[d1,d2,d3,d4,d5,d6, rd1,rd2,rd3,rd4,rd5,rd6, pct, method], outputs=[ratio,pct_volume,volume,result])
demo.launch(share=share, server_name=server_name, server_port=server_port)
