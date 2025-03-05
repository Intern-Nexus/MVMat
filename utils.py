import numpy as np
from PIL import Image
import gc
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
from torchvision.transforms import Resize, InterpolationMode

GRADIO_CACHE = "/tmp/gradio/"

def clean_up():
    torch.cuda.empty_cache()
    gc.collect()

def remove_color(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    # calc diffs
    base = arr[0, 0]
    diffs = np.abs(arr.astype(np.int32) - base.astype(np.int32)).sum(axis=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    arr = np.concatenate([arr, alpha[..., None].astype(np.int32) * 255], axis=-1)
    return arr

def rgba_to_rgb(rgba: Image.Image, bkgd="WHITE"):
    new_image = Image.new("RGBA", rgba.size, bkgd)
    new_image.paste(rgba, (0, 0), rgba)
    new_image = new_image.convert('RGB')
    return new_image

def change_rgba_bg(rgba: Image.Image, bkgd="WHITE"):
    rgb_white = rgba_to_rgb(rgba, bkgd)
    new_rgba = Image.fromarray(np.concatenate([np.array(rgb_white), np.array(rgba)[:, :, 3:4]], axis=-1))
    return new_rgba

def split_image(image, rows=None, cols=None):
    """
        inverse function of make_image_grid
    """
    # image is in square
    if rows is None and cols is None:
        # image.size [W, H]
        rows = 1
        cols = image.size[0] // image.size[1]
        assert cols * image.size[1] == image.size[0]
        subimg_size = image.size[1]
    elif rows is None:
        subimg_size = image.size[0] // cols
        rows = image.size[1] // subimg_size
        assert rows * subimg_size == image.size[1]
    elif cols is None:
        subimg_size = image.size[1] // rows
        cols = image.size[0] // subimg_size
        assert cols * subimg_size == image.size[0]
    else:
        subimg_size = image.size[1] // rows
        assert cols * subimg_size == image.size[0]
    subimgs = []
    for i in range(rows):
        for j in range(cols):
            subimg = image.crop((j*subimg_size, i*subimg_size, (j+1)*subimg_size, (i+1)*subimg_size))
            subimgs.append(subimg)
    return subimgs

def make_image_grid(images, rows=None, cols=None, resize=None):
    if rows is None and cols is None:
        rows = 1
        cols = len(images)
    if rows is None:
        rows = len(images) // cols
        if len(images) % cols != 0:
            rows += 1
    if cols is None:
        cols = len(images) // rows
        if len(images) % rows != 0:
            cols += 1
    total_imgs = rows * cols
    if total_imgs > len(images):
        images += [Image.new(images[0].mode, images[0].size) for _ in range(total_imgs - len(images))]
    
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new(images[0].mode, size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

# copied from https://github.com/huggingface/diffusers/blob/5d476f57c58c3cf7f39e764236c93c267fe83ca1/src/diffusers/training_utils.py#L196
def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)

'''
	Encoding and decoding functions similar to diffusers library implementation
'''
@torch.no_grad()
def encode_latents(vae, imgs):
	imgs = (imgs-0.5)*2
	latents = vae.encode(imgs).latent_dist.sample()
	latents = vae.config.scaling_factor * latents
	return latents


@torch.no_grad()
def decode_latents(vae, latents):

	latents = 1 / vae.config.scaling_factor * latents

	image = vae.decode(latents, return_dict=False)[0]
	torch.cuda.current_stream().synchronize()

	image = (image / 2 + 0.5).clamp(0, 1)
	# we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
	image = image.permute(0, 2, 3, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	
	return image


# A fast decoding method based on linear projection of latents to rgb
@torch.no_grad()
def latent_preview(x):
	# adapted from https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/7
	v1_4_latent_rgb_factors = torch.tensor([
		#   R        G        B
		[0.298, 0.207, 0.208],  # L1
		[0.187, 0.286, 0.173],  # L2
		[-0.158, 0.189, 0.264],  # L3
		[-0.184, -0.271, -0.473],  # L4
	], dtype=x.dtype, device=x.device)
	image = x.permute(0, 2, 3, 1) @ v1_4_latent_rgb_factors
	image = (image / 2 + 0.5).clamp(0, 1)
	image = image.float()
	image = image.cpu()
	image = image.numpy()
	return image


# Decode each view and bake them into a rgb texture
def get_rgb_texture(vae, uvp_rgb, latents):
	result_views = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
	resize = Resize((uvp_rgb.render_size,)*2, interpolation=InterpolationMode.NEAREST_EXACT, antialias=True)
	result_views = resize(result_views / 2 + 0.5).clamp(0, 1).unbind(0)
	textured_views_rgb, result_tex_rgb, visibility_weights = uvp_rgb.bake_texture(views=result_views, main_views=[], exp=6, noisy=False)
	result_tex_rgb_output = result_tex_rgb.permute(1,2,0).cpu().numpy()[None,...]
	return result_tex_rgb, result_tex_rgb_output