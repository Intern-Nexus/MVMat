import os
import os.path as osp
from PIL import Image
from glob import glob
import numpy as np
import imageio.v2 as imageio
import torch
from renderer.project import UVProjection as UVP
from utils import *
import argparse

if torch.cuda.is_available():
	device = torch.device("cuda:0")
	torch.cuda.set_device(device)
else:
	device = torch.device("cpu")

class MatMerge(object):
	def __init__(self, uv_size, img_size, mesh_path, auto_center=True, scale_factor=1, autouv=False):
		self.uv_size = uv_size
		self.img_size = img_size
		self.uvp = UVP(texture_size=uv_size, render_size=img_size, sampling_mode="nearest", channels=3, device=torch.device('cuda'))
		if mesh_path[-3:] == 'obj':
			self.uvp.load_mesh(mesh_path, auto_center=auto_center, scale_factor=scale_factor, autouv=autouv)
		else:
			raise ValueError("3D object format error!")
		self.mesh_id = osp.basename(mesh_path)[:-4]

		self.azims = [0, 90, 180, -90]
		self.camera_poses = [(0, azim) for azim in self.azims]
		self.camera_poses.extend([(90, 0), (-90, 0)])
		self.camera_poses_info = [f"{x[0]}_{x[1]}" for x in self.camera_poses]
		self.camera_poses_names = ['front', 'right', 'back', 'left', 'top', 'bottom']
		self.uvp.set_cameras_and_render_settings(camera_poses=self.camera_poses, camera_distance=2.6)
	
	@staticmethod
	def save_image(images, basedir, img_names):
		assert len(images) == len(img_names)
		for img, img_name in zip(images, img_names):
			if len(img.shape) == 3:
				assert img.shape[-1] == 3 or img.shape[-1] == 4
			img = (img * 255).astype(np.uint8)
			imageio.imwrite(f"{basedir}/{img_name}.png", img)
	
	def get_vis_map(self):
		os.makedirs(f"results/uv_mask/{self.mesh_id}", exist_ok=True)
		mv_vismap = sum(self.uvp.visible_triangles).clamp(min=0, max=1)
		imageio.imwrite(f"results/uv_mask/{self.mesh_id}/mask_mv.png", (mv_vismap[..., 0].cpu().numpy() * 255).astype(np.uint8))

		all_uvmap = self.uvp.all_visible_triangles
		imageio.imwrite(f"results/uv_mask/{self.mesh_id}/mask_alluv.png", (all_uvmap[..., 0].cpu().numpy() * 255).astype(np.uint8))
		
		cos_vismap = self.uvp.cos_visible_triangles
		imageio.imwrite(f"results/uv_mask/{self.mesh_id}/mask_cosvis.png", (cos_vismap[..., 0].cpu().numpy() * 255).astype(np.uint8))

		zero_pad = torch.zeros_like(mv_vismap)
		vismap_com_cosvis = torch.cat([zero_pad, all_uvmap, cos_vismap], dim=-1)
		imageio.imwrite(f"results/uv_mask/{self.mesh_id}/mask_compare_cosvis.png", (vismap_com_cosvis.cpu().numpy() * 255).astype(np.uint8))
		vismap_com_mv = torch.cat([zero_pad, all_uvmap, mv_vismap], dim=-1)
		imageio.imwrite(f"results/uv_mask/{self.mesh_id}/mask_compare_mv.png", (vismap_com_mv.cpu().numpy() * 255).astype(np.uint8))

	def get_normal(self, out_dir):
		verts, normals, depths, cos_angles, texels, fragments = self.uvp.render_geometry()
		# normals: [Num_cam, H, W, 4], the last channel is the mask
		# world_normals = [normal[..., :3].cpu().numpy() / 2 + 0.5 for normal in normals]
		normal_masks = [normal[..., 3].cpu().numpy() for normal in normals]
		view_normals = self.uvp.decode_view_normal(normals) # convert to view space
		view_normals = [view_normal.cpu().numpy() for view_normal in view_normals]
		view_normals_masked = [vn * nm[..., np.newaxis] for (vn, nm) in zip(view_normals, normal_masks)]
		
		os.makedirs(out_dir, exist_ok=True)
		# self.save_image(world_normals, out_dir, img_names=self.camera_poses_names)
		self.save_image(normal_masks, out_dir, img_names=[f"mask_{x}" for x in self.camera_poses_names])
		self.save_image(view_normals, out_dir, img_names=[f"view_{x}" for x in self.camera_poses_names])
		self.save_image(view_normals_masked, out_dir, img_names=self.camera_poses_names)
	
	def loat_mv_mat(self, mat_path, mat_id, img_suffix=''):
		mat_types = {'albedo', 'metalness', 'roughness'}
		views = ['front', 'right', 'back', 'left', 'top', 'bottom']
		all_mats = {}
		for mat_type in mat_types:
			all_mats[mat_type] = []
			for view in views:
				img_file = f"{mat_path}/{mat_id}_{mat_type[0]}_{view}{img_suffix}.png"
				assert osp.exists(img_file)
				cur_mat = imageio.imread(img_file) / 255. # normalize to [0, 1]
				cur_mat = torch.from_numpy(cur_mat).to('cuda').permute(2, 0, 1) # shape to [C, H, W]
				all_mats[mat_type].append(cur_mat)

		return all_mats
	
	def merge_view(self, mat_dir, mat_id, out_dir, img_suffix=''):
		mats = self.loat_mv_mat(f"{mat_dir}/{mat_id}/{mat_id}_upscale", mat_id=mat_id, img_suffix=img_suffix)
		# combine orm channels
		rmo_views = []
		for idx in range(len(self.camera_poses)):
			rmo_view = torch.stack([mats['roughness'][idx][1], mats['metalness'][idx][1], mats['roughness'][idx][1]])
			rmo_views.append(rmo_view)

		_, albedo_uv, _ = self.uvp.bake_texture(views=mats['albedo'], main_views=[], exp=6, noisy=False)
		_, rmo_uv, _ = self.uvp.bake_texture(views=rmo_views, main_views=[], exp=6, noisy=False)
		out_dir = osp.join(out_dir, mat_id)
		os.makedirs(out_dir, exist_ok=True)
		self.uvp.save_mesh(f"{out_dir}/textured.obj", albedo_uv.permute(1,2,0))
		imageio.imwrite(f"{out_dir}/roughness.png", (rmo_uv[0].cpu().numpy() * 255).astype(np.uint8))
		imageio.imwrite(f"{out_dir}/metalness.png", (rmo_uv[1].cpu().numpy() * 255).astype(np.uint8))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Render the mesh for 6-view normal and merge 6-view PBR images.")
	parser.add_argument("--mesh_dir", type=str, default="data/input_mesh")
	parser.add_argument("--mesh_id", type=str, default=None, required=True)
	parser.add_argument("--run_mode", type=str, default='get_normal', choices=['get_normal', 'merge_PBR'])
	parser.add_argument("--normal_size", type=int, default=512)
	parser.add_argument("--normal_dir", type=str, default="data/normal")
	parser.add_argument("--mat_dir", type=str, default="data/PBR_6views")
	parser.add_argument("--out_dir", type=str, default="data/out_mesh")
	args = parser.parse_args()
    
	mesh_path = f"{args.mesh_dir}/{args.mesh_id}.obj"
	assert osp.exists(mesh_path)

	if args.run_mode == 'get_normal':
		#### get normal ####
		normal_path = osp.join(args.normal_dir, args.mesh_id)
		os.makedirs(normal_path, exist_ok=True)
		matmerger = MatMerge(
			uv_size=args.normal_size*2,
			img_size=args.normal_size,
			mesh_path=mesh_path,
			autouv=False    # set True for the mesh w/o vt
		)
		matmerger.get_normal(out_dir=normal_path)
		# matmerger.get_vis_map()  # visualize the area not covered by 6 views
	elif args.run_mode == 'merge_PBR':
		#### combine 6 views ####
		render_size = args.normal_size * 4  # 2K-res PBR, 4K-res UV map
		mat_id = args.mesh_id  # used for multiple attempts where different mat folders are generated
		img_suffix = '_upscale_8x'
		matmerger = MatMerge(uv_size=render_size*2, img_size=render_size, mesh_path=mesh_path, autouv=False)
		matmerger.merge_view(mat_dir=args.mat_dir, mat_id=mat_id, out_dir=args.out_dir, img_suffix=img_suffix)
	else:
		pass
	