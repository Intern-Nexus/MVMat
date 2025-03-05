import os
import os.path as osp
import shutil
import subprocess

src_path = "data/PBR_6views"
mesh_ids = [
    'f_pants',
]
mat_names = ['a', 'r', 'm']
views = ['top', 'bottom']

for mesh_id in mesh_ids:
    if mesh_id.split('_')[-1] in ['tmp', 'upscale']: # ignore 'xxx_tmp' and 'xxx_upscale'
        continue
    cur_path = osp.join(src_path, mesh_id)
    tmp_dir = f"{cur_path}/{mesh_id}_tmp"
    out_dir = f"{cur_path}/{mesh_id}_upscale"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    for mat_name in mat_names:
        for view in views:
            input_img = f"{cur_path}/{mesh_id}/{mesh_id}_{mat_name}_{view}.png"
            assert osp.exists(input_img)
            shutil.copy(input_img, tmp_dir)

    cmd = f"python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i {tmp_dir} -o {out_dir} -s 2 --suffix upscale"
    subprocess.call(cmd, shell=True)

    cmd2 = f"mv {cur_path}/{mesh_id}_upscale/* {cur_path}/{mesh_id}"
    subprocess.call(cmd2, shell=True)

    cmd3 = f"cp {cur_path}/{mesh_id}/*_upscale.png {out_dir}"
    subprocess.call(cmd3, shell=True)

    cmd4 = f"python Real-ESRGAN/inference_realesrgan.py -n RealESRGAN_x4plus -i {out_dir} -o {out_dir} -s 4 --suffix 8x"
    subprocess.call(cmd4, shell=True)
