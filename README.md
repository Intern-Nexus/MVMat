# Multi-view PBR Material Diffusion Model (MVMat)
## Installation
- Install the required dependencies via **requirements.txt**. For the installation of [**CuPy**](https://docs.cupy.dev/en/stable/install.html) and [**PyTorch3D**](https://github.com/facebookresearch/pytorch3d), refer to their guidance for more details.
```bash
# remember to remove packages related to cupy and pytorch3d if they cannot be successfully installed directly via pip install -r requirements.txt
pip install -r requirements.txt
```

- (Optional) Install PyTorch at first via [**PyTorch Official**](https://pytorch.org/get-started/previous-versions/), then install other dependencies via **requirements.txt**. For the installation of [**CuPy**](https://docs.cupy.dev/en/stable/install.html) and [**PyTorch3D**](https://github.com/facebookresearch/pytorch3d), refer to their guidance for more details.
```bash
# CUDA 11.6
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# remember to remove packages related to torch
# remember to remove packages related to cupy and pytorch3d if they cannot be successfully installed directly via pip install -r requirements.txt
pip install -r requirements.txt
```

## Download checkpoints
### Pre-trained weights
- MVDream & MVControlNet

For MVDream base model and Multi-view ControlNet, we use a third-party [**diffusers implementation**](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion) inherited from the [**HuggingFace repo**](https://huggingface.co/lzq49/mvdream-sd21-diffusers) of [**Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting**](https://lizhiqi49.github.io/MVControl/), instead of the official implementation.

Download [**MVDream**](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion) and [**MVControlNet**](https://huggingface.co/lzq49/mvcontrol-4v-normal).

- CLIP

Download [**CLIP-ViT-H-14-laion2B-s32B-b79K**](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K).

- MVSuperResolution

For multi-view super-resolution, we use a ControlNet-Tile model provided by [**Unique3D**](https://wukailu.github.io/Unique3D/).

Download [**controlnet-tile**](https://huggingface.co/spaces/Wuvin/Unique3D/tree/main/ckpt/controlnet-tile).

- SuperResolution

Download [**Real-ESRGAN**](https://github.com/xinntao/Real-ESRGAN), and set it up.

- Blender

Download [**Blender**](https://download.blender.org/release/Blender3.4/blender-3.4.1-linux-x64.tar.xz) and leverage it for the conversion in the last step.

### (Optional) Fine-tuned checkpoints
- All modules are unified in a single checkpoint.

Download [**pytorch_model.bin**]().

### Final saved fine-tuned checkpoints
- IP-Adapter & Image Projection Model

Download [**ip_adapter.pt**](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/ip_adapter.pt?download=true) and [**image_proj_model.pt**](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/image_proj_model.pt?download=true).

- UNet LoRA & Multi-branch

Download [**unet.pt**](https://huggingface.co/SnowflakeWang/MV-PBRMat-Diffusion/resolve/main/unet.pt?download=true), LoRA and Multi-branch have been unified in it.

## Usage
### Prepare input
- Put the untextured mesh in the data folder **data/input_mesh**.

- In step 2, if use 'image' as the controlnet_cond_mode, put the reference image in **data/ctrlnet_img**.
```bash
MVMat
|-- data
    |-- ctrlnet_img
        |-- fc4_crop.png
        |-- ironman.png
        |-- ...
    |-- input_mesh
        |-- batman.obj
        |-- f_pants.obj
        |-- ...
```

### Run code
- **run_all.sh** demonstrates the full pipeline.
```bash
# in run_all.sh

mesh_id=f_pants # set the `mesh_id` for different input meshes.

#### step 1: get 6 views normal: mesh → 6 views normal
python render_combine.py --mesh_id $mesh_id \
    --mesh_dir data/input_mesh \
    --run_mode get_normal \
    --normal_dir data/normal

#### step 2: get sample image: front normal + img/txt prompt → sample image
python ctrlnet_gen.py --ctrlnet_seed 12 \
    --controlnet_cond_mode="image" \ # set controlnet_cond_mode='text' to switch to the text-control setting
    --prompt="data/ctrlnet_img/fc4_crop.png" \ # if controlnet_cond_mode=='text', directly input a text prompt
    --controlnet_normal_path="data/normal/$mesh_id/front.png"

#### step 3: get 6 views PBR: sample image + 6 views normal → 6 views PBR
python inference.py --mvdiff_seed=1234 \
    --input_path="data/normal/$mesh_id" \
    --out_path="data/PBR_6views/$mesh_id" \
    --text_prompt="" \ # MVMat Diffusion can also take an additional text prompt, but it's optional. We recommend injecting text prompts into step 2 to generate normal-aligned images first to achieve text-guided multi-view PBR generation
    --do_mv_super_res

#### step 4: PBR SR: 6 views PBR → 6 views PBR upsample
python run_PBR_SR.py --mesh_id $mesh_id

#### step 5: PBR combine: 6 views PBR upsample → PBR UV map + obj
python render_combine.py --mesh_id $mesh_id \
    --run_mode merge_PBR \
    --mat_dir data/PBR_6views \
    --normal_dir data/out_mesh

#### step 6: Convert obj to glb: PBR UV map + obj → glb
blender -b -P bpy_obj_to_glb.py -- --mesh_id $mesh_id
```
- Run **run_all.sh** for inference.
```bash
bash run_all.sh
```

## Acknowledgments
- [CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets](https://sites.google.com/view/clay-3dlm)
- [MVDream: Multi-view Diffusion for 3D Generation](https://mv-dream.github.io/)
- [Controllable Text-to-3D Generation via Surface-Aligned Gaussian Splatting](https://lizhiqi49.github.io/MVControl/)
- [IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models](https://ip-adapter.github.io/)
- [HyperHuman: Hyper-Realistic Human Generation with Latent Structural Diffusion](https://snap-research.github.io/HyperHuman/)
- [Objaverse: A Universe of Annotated 3D Objects](https://objaverse.allenai.org/)
- [G-buffer Objaverse: High-Quality Rendering Dataset of Objaverse](https://aigc3d.github.io/gobjaverse/)
- [Material Anything: Generating Materials for Any 3D Object via Diffusion](https://xhuangcv.github.io/MaterialAnything/)
- [Unique3D: High-Quality and Efficient 3D Mesh Generation from a Single Image](https://wukailu.github.io/Unique3D/)