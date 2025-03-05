import bpy
import os
import sys
import argparse

def convert_obj_to_glb(
    obj_path: str,
    texture_path: str,
    roughness_path: str,
    metallic_path: str,
    output_glb_path: str
):
    # 清除初始场景
    bpy.ops.wm.read_factory_settings(use_empty=True)

    try:
        # 导入OBJ文件
        bpy.ops.import_scene.obj(filepath=obj_path)

        # 创建新材质
        mat = bpy.data.materials.new(name="PBR_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # 清除默认节点
        for node in nodes:
            nodes.remove(node)

        # 创建原理化BSDF节点
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.location = (0, 0)
        
        # 创建输出节点
        output = nodes.new(type='ShaderNodeOutputMaterial')
        output.location = (400, 0)
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

        # 添加纹理贴图
        def create_texture_node(path, color_space='sRGB', name="Texture"):
            tex_node = nodes.new('ShaderNodeTexImage')
            tex_node.image = bpy.data.images.load(path)
            tex_node.image.colorspace_settings.name = color_space
            tex_node.label = name
            return tex_node

        # 基础色贴图
        base_color = create_texture_node(texture_path, 'sRGB', "Base Color")
        base_color.location = (-400, 300)
        links.new(base_color.outputs['Color'], bsdf.inputs['Base Color'])

        # 粗糙度贴图
        roughness = create_texture_node(roughness_path, 'Non-Color', "Roughness")
        roughness.location = (-400, 0)
        links.new(roughness.outputs['Color'], bsdf.inputs['Roughness'])

        # 金属度贴图
        metallic = create_texture_node(metallic_path, 'Non-Color', "Metallic")
        metallic.location = (-400, -300)
        links.new(metallic.outputs['Color'], bsdf.inputs['Metallic'])

        # 应用材质到所有网格对象
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH':
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)

        # 设置GLB导出参数
        bpy.ops.export_scene.gltf(
            filepath=output_glb_path,
            export_format='GLB',
            export_image_format='AUTO',
            export_texcoords=True,
            export_normals=True,
            export_materials='EXPORT',
            export_apply=True
        )

        print(f"Export Sucessed: {output_glb_path}")

    except Exception as e:
        print(f"Export failed: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert obj to glb.")
    parser.add_argument("--data_dir", type=str, default="data/out_mesh")
    parser.add_argument("--mesh_id", type=str, default=None, required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    convert_obj_to_glb(
        obj_path=f"{args.data_dir}/{args.mesh_id}/textured.obj",
        texture_path=f"{args.data_dir}/{args.mesh_id}/textured.png",
        roughness_path=f"{args.data_dir}/{args.mesh_id}/roughness.png",
        metallic_path=f"{args.data_dir}/{args.mesh_id}/metalness.png",
        output_glb_path=f"{args.data_dir}/{args.mesh_id}.glb"
    )
