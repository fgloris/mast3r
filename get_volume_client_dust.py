from gradio_client import Client, handle_file
import shutil,os

if not os.path.exists("client"):
    os.mkdir("client")

if os.path.exists("client/scene.ply"): 
	print("get 3d model with images in dir client ?(Y/N)")
	key = input()
else: key = 'Y'
if (key=='y' or key=='Y'):
	client = Client("http://47.120.49.17:8860/",httpx_kwargs = {"timeout":60})

	filelist = []
	for file in os.listdir("client"):
		file = os.path.join("client",file)    
		filelist.append(handle_file(file))
		
	if len(filelist)==0:
		print("no images input in client/, leaving...")
		exit()
	result = client.predict(
			filelist=filelist,
			schedule="linear",
			niter=300,
			min_conf_thr=3,
			as_pointcloud=False,
			mask_sky=False,
			clean_depth=True,
			transparent_cams=False,
			cam_size=0.05,
			scenegraph_type="complete",
			winsize=1,
			refid=0,
			api_name="/partial"
	)

	file_path = result[0]

	shutil.copy(file_path,"./client")

import open3d as o3d
import math
import numpy as np

def find_alpha(pcd):
    #二分法算alpha
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
    print("alpha:",a)
    return mesh

def get_volume(mesh):
    if mesh.is_watertight():
        print("The mesh is watertight.")
        volume = mesh.get_volume()
        return volume
    else:
        print("The mesh is not watertight. Trying convex hull for approximate volume.")
        hull, _ = mesh.compute_convex_hull()
        hull_volume = hull.get_volume()
        return hull_volume

def pick_points(pcd):
    print("Please pick at least three correspondences using [shift + left click]")
    print("Press [shift + right click] to undo point picking")
    print("After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    print(vis.get_picked_points())
    return vis.get_picked_points(),vis.get_cropped_geometry()

def get_distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]))

# 读取并预处理点云
if os.path.exists("client/output.ply"):
    key = input("use previously saved edited pointcloud output.ply?")
else: key = 'N'
if key == 'Y' or key == 'y':
    mesh = o3d.io.read_point_cloud("client/output.ply")
else :
    mesh = o3d.io.read_point_cloud("client/scene.ply")
pts,mesh = pick_points(mesh)
o3d.io.write_point_cloud(filename = "client/output.ply",pointcloud = mesh,write_ascii=True)
ratio = 1
if len(pts)<6: print("less than 6 pair of points selected! it is suggested to add more pair to improve accuracy.")
for i in range(0,int(len(pts)/2)):
    pcd_distance = get_distance(mesh.points[pts[2*i]],mesh.points[pts[(2*i)+1]])
    print(mesh.points[pts[2*i]],mesh.points[pts[(2*i)+1]])
    print("the relative distance between the %d point pair is"%(i+1),pcd_distance)
    real_distance = float(input("please input the real distance between them:(cm)"))
    print("the ratio (real_distance/pcd_distance) = ",real_distance/pcd_distance)
    ratio *= real_distance/pcd_distance
ratio = pow(ratio, 3/int(len(pts)/2))
mesh = mesh.voxel_down_sample(voxel_size=0.02)
mesh.estimate_normals()
mesh = find_alpha(mesh)
mesh.compute_vertex_normals()
mesh_volume = get_volume(mesh)
print("the real volume is:",mesh_volume*ratio,"cm^3")
o3d.visualization.draw_geometries([mesh])