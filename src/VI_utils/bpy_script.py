import bpy
import os
import sys

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)

def get_armature_and_mesh():
    armature = None
    mesh = None
    for obj in bpy.context.selected_objects:
        if obj.type == 'ARMATURE':
            armature = obj
        elif obj.type == 'MESH':
            mesh = obj
    return armature, mesh

def center_and_scale_obj(obj, height=1.8):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    
    max_dim = max(obj.dimensions)
    scale_factor = height / max_dim
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()
    
    obj.location.z = height / 2

def apply_transforms(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

def parent_obj_to_turn(obj):
    empty = bpy.data.objects.get('turn')
    if not empty:
        print("Turntable object 'turn' not found in the scene")
        return
    obj.parent = empty

def assign_material(obj, mat_name, texture_bool):
    print(f"Assigning material: {mat_name} to object: {obj.name}")
    
    # Get the specified material from the scene
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        raise ValueError(f"Material {mat_name} not found in the scene")
    
    # Create a copy of the material
    new_mat = mat.copy()
    new_mat.name = f"{obj.name}_procedural_MAT"
    
    # Dynamically pick up the GLB imported material and textures
    imported_material = bpy.data.materials.get("Material_0") or bpy.data.materials.get("Material_1")
    diffuse_texture = bpy.data.images.get("Image_0") or bpy.data.images.get("texture_0") or bpy.data.images.get("tex_base_color_0")
    normal_texture = bpy.data.images.get("Image_1") or bpy.data.images.get("texture_1")
    
    print(f"Diffuse texture: {diffuse_texture.name if diffuse_texture else 'Not found'}")
    print(f"Normal texture: {normal_texture.name if normal_texture else 'Not found'}")
    
    if imported_material and texture_bool == 'True':
        print(f"Found imported material: {imported_material.name}")
        
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes
        links = new_mat.node_tree.links
        
        # Find the ARV_SHAD node group
        arv_shad = nodes.get("ARV_SHAD")
        if arv_shad:
            # Create and connect diffuse texture if it exists
            if diffuse_texture:
                tex_diffuse = nodes.new(type='ShaderNodeTexImage')
                tex_diffuse.image = diffuse_texture
                links.new(tex_diffuse.outputs['Color'], arv_shad.inputs['COLOR'])
                
                # # For the ROUGH input, we'll use the alpha channel of the diffuse texture if it exists
                # if tex_diffuse.image.channels == 4:  # Check if the image has an alpha channel
                #     links.new(tex_diffuse.outputs['Alpha'], arv_shad.inputs['ROUGH'])
                # else:
                #     print("No alpha channel found for roughness, using default value")
            
            # Create and connect normal texture if it exists
            if normal_texture:
                tex_normal = nodes.new(type='ShaderNodeTexImage')
                tex_normal.image = normal_texture
                
                # Create normal map node
                normal_map = nodes.new(type='ShaderNodeNormalMap')
                normal_map.inputs['Strength'].default_value = 1.0
                
                links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
                links.new(normal_map.outputs['Normal'], nodes['Principled BSDF'].inputs['Normal'])
            
    obj.data.materials.clear()
    obj.data.materials.append(new_mat)

def frame_camera_to_obj(obj, camera_name='Camera', zoom_out_factor=1.2):
    camera = bpy.data.objects.get(camera_name)
    if not camera:
        print(f"Camera '{camera_name}' not found in the scene")
        return

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.view3d.camera_to_view_selected()

    if camera.data.type == 'ORTHO':
        camera.data.ortho_scale *= zoom_out_factor
    else:
        direction = camera.location - obj.location
        camera.location = obj.location + direction * zoom_out_factor

    bpy.context.view_layer.update()

def save_scene(output_path, base_name):
    version = 1
    while True:
        versioned_name = f"{base_name}_blendFile_v{version:03d}.blend"
        full_path = os.path.join(output_path, versioned_name)
        if not os.path.exists(full_path):
            break
        version += 1
    
    bpy.ops.wm.save_as_mainfile(filepath=full_path, copy=True)
    print(f"Scene saved as {full_path}")

def main():
    if len(sys.argv) < 5:
        print("Usage: blender -b <blend_file> -P <script> -- <glb_file> <output_path>")
        sys.exit(1)

    glb_file = sys.argv[-2]
    output_path = sys.argv[-1]

    if not os.path.exists(glb_file):
        print(f"Error: File not found - {glb_file}")
        sys.exit(1)

    # Import GLB
    import_glb(glb_file)

    # Get the imported armature and mesh objects
    armature, mesh = get_armature_and_mesh()
    if not armature and not mesh:
        print("Error: No armature or mesh object found in the imported GLB file")
        sys.exit(1)

    # Use armature for transformations if available, otherwise use mesh
    main_obj = armature if armature else mesh

    # Center and scale the imported object
    center_and_scale_obj(main_obj)

    # Apply transforms
    apply_transforms(main_obj)

    # Parent to turn object
    parent_obj_to_turn(main_obj)

    # Assign material to the mesh
    if mesh:
        assign_material(mesh, "grey_procedural_MAT", texture_bool='True')
    else:
        print("Warning: No mesh found to assign material")

    # Frame camera to object
    frame_camera_to_obj(main_obj)

    # Set viewport shading to rendered
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'RENDERED'

    # Save the scene
    base_name = os.path.splitext(os.path.basename(glb_file))[0]
    save_scene(output_path, base_name)

    # Play the timeline automatically
    bpy.ops.screen.animation_play()

    # Start viewport rendering
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

if __name__ == "__main__":
    main()