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

class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None

class ReconstructSurfaceMethod():
    def __init__(self, method, alpha):
        self.method = method
        self.alpha = alpha

def get_method(method, alpha):
    return ReconstructSurfaceMethod(method, alpha)

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


def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask,
                                  silent=False):
    assert len(pts3d) == len(mask) <= len(imgs)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)

    scene = trimesh.Scene()

    pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
    valid_msk = np.isfinite(pts.sum(axis=1))
    pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])

    scene.add_geometry(pct)
    scene.export(file_obj=outfile)
    pct.export('scene.ply')
    return outfile


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2,
                            clean_depth=False, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile,rgbimg, pts3d, msk,
                                         silent=silent)


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, 
                                      clean_depth, TSDF_thresh)
    return scene_state, outfile


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                            minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
    win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
    win_col = gradio.Column(visible=show_win_controls)
    refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=scenegraph_type == 'oneref')
    return win_col, winsize, win_cyclic, refid

def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model, device,
                                  silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        pct = gradio.State(None)
        mesh = gradio.State(None)
        method = gradio.State(get_method('convex hull', 0.5))
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
                                               label="num_iterations", info="For coarse alignment!")
                        lr2 = gradio.Slider(label="Fine LR", value=0.014, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Number(value=200, precision=0, minimum=0, maximum=100_000,
                                               label="num_iterations", info="For refinement!")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=5.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                           ("swin: sliding window", "swin"),
                                                           ("logwin: sliding window with long range", "logwin"),
                                                           ("oneref: match one image with all", "oneref")],
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as win_col:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                        refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                              minimum=0, maximum=0, step=1, visible=False)
            run_btn = gradio.Button("Run")


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
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                          outputs=[scene, outmodel])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                    outputs=outmodel)
            
            pct_process_btn.click(fn=pct_process,
                          inputs=[],
                          outputs=[pct, d1,d2,d3,d4,d5,d6])
            
            method_name.change(fn=get_method, inputs=[method_name, alpha], outputs=[method])
            alpha.change(fn=get_method, inputs=[method_name, alpha], outputs=[method])

            volume_btn.click(fn=get_volume_from_pct, inputs=[d1,d2,d3,d4,d5,d6, rd1,rd2,rd3,rd4,rd5,rd6, pct, method], outputs=[ratio,pct_volume,volume,result])

            pct_process_btn.click(fn=demo.close,
                          inputs=[],
                          outputs=[])
            demo.unload(demo.close)
    demo.launch(share=share, server_name=server_name, server_port=server_port)
