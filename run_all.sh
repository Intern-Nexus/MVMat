#!/bin/bash

mesh_id=f_pants

#### step 1: get 6 views normal: mesh → 6 views normal
python render_combine.py --mesh_id $mesh_id \
    --mesh_dir data/input_mesh \
    --run_mode get_normal \
    --normal_dir data/normal

#### step 2: get sample image: front normal + img/txt prompt → sample image
python ctrlnet_gen.py --ctrlnet_seed 12 \
    --controlnet_cond_mode="image" \
    --prompt="data/ctrlnet_img/fc4_crop.png" \
    --controlnet_normal_path="data/normal/$mesh_id/front.png"

#### step 3: get 6 views PBR: sample image + 6 views normal → 6 views PBR
python inference.py --mvdiff_seed=1234 \
    --input_path="data/normal/$mesh_id" \
    --out_path="data/PBR_6views/$mesh_id" \
    --text_prompt="" \
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
