## Installation
- Download [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN), and set it up.
- Download the ckpt (controlnet-tile/) from [**Unique3D**](https://github.com/AiuniAI/Unique3D), and the ckpt folder in `mvsr`.
- Download **Blender** and leverage it for the conversion in the last step.

## Usage
- Put the untextured mesh in the data folder **data/input_mesh** and the reference image in **data/ctrlnet_img**.
- Follow the `run_all.sh` to finish the whole pipeline. In step 2, users can also set `controlnet_cond_mode='text'` to switch to the text-control setting
- Set the `mesh_id` for different input meshes.