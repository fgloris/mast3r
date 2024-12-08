from gradio_client import Client, handle_file

client = Client("http://0.0.0.1:7860/")
result = client.predict(
        filelist=[handle_file("/home/ginger/Downloads/IMG_20241114_141345.jpg"),handle_file("/home/ginger/Downloads/IMG_20241114_141350.jpg")],
		min_conf_thr=1.5,
		as_pointcloud=True,
		mask_sky=False,
		clean_depth=True,
		transparent_cams=False,
		cam_size=0.2,
		TSDF_thresh=0,
		api_name="/partial"
)
print(result)