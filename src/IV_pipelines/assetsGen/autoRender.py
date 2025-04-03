import bpy
import os
import sys

def enable_addons():
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_obj")
        print("Enabled io_scene_obj add-on")
    except Exception as e:
        print(f"Warning: io_scene_obj add-on not available: {e}")
    
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
        print("Enabled io_scene_gltf2 add-on")
    except Exception as e:
        print(f"Error enabling io_scene_gltf2 add-on: {e}")

def set_gpu_rendering():
    print("Setting up GPU rendering")
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences

    # Check for available GPU types
    if bpy.context.preferences.addons['cycles'].preferences.get("compute_device_type") is not None:
        if bpy.context.preferences.addons['cycles'].preferences.compute_device_type in {'CUDA', 'OPENCL', 'OPTIX'}:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
        elif bpy.context.preferences.addons['cycles'].preferences.compute_device_type == 'METAL':
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'METAL'
            bpy.context.scene.cycles.device = 'GPU'
        else:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'NONE'
            print("No compatible GPU found, defaulting to CPU rendering")
    else:
        print("No compatible GPU found, defaulting to CPU rendering")

    # Enable all available GPU devices
    for device in cycles_prefs.devices:
        if device.type in {'CUDA', 'OPTIX', 'METAL'}:
            device.use = True
            print(f"Enabled GPU device: {device.name}")
        else:
            device.use = False
    print("GPU rendering set up")

def import_scene_template(template_path):
    enable_addons()
    print(f"Importing scene template from: {template_path}")
    bpy.ops.wm.open_mainfile(filepath=template_path)
    if len(bpy.data.objects) == 0:
        raise ValueError(f"Failed to load template from {template_path}. No objects found in the scene.")
    print("Scene template imported successfully")

def import_obj(obj_path):
    if not os.path.exists(obj_path):
        raise ValueError(f"File not found: {obj_path}")
    
    file_extension = os.path.splitext(obj_path)[1][1:].lower()  # Get file extension without the dot
    
    print(f"Importing {file_extension.upper()} file from: {obj_path}")
    
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')
    
    if file_extension == 'obj':
        bpy.ops.wm.obj_import(filepath=obj_path)
    elif file_extension == 'glb' or file_extension == 'gltf':
        bpy.ops.import_scene.gltf(filepath=obj_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Get all imported objects
    imported_objs = [obj for obj in bpy.context.scene.objects if obj.select_get()]
    if not imported_objs:
        raise ValueError("No objects were imported.")
    
    # Select all imported objects and set the active object
    for obj in imported_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_objs[0]
    
    print(f"Imported {file_extension.upper()} file: {', '.join([obj.name for obj in imported_objs])}")
    
    # Apply shade smooth to all imported objects
    for obj in imported_objs:
        try:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.shade_smooth()
            obj.select_set(False)
        except Exception as e:
            print(f"Failed to apply shade smooth to {obj.name}: {str(e)}")
            print("Continuing without applying shade smooth.")
    
    return imported_objs[0] if len(imported_objs) == 1 else imported_objs

def center_and_scale_obj(obj, height=1.8):
    print(f"Centering and scaling object: {obj.name}")
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0.0, 0.0, 0.0)
    
    max_dim = max(obj.dimensions)
    if max_dim == 0:
        print(f"Object {obj.name} has zero dimensions, skipping scaling")
        return
    
    scale_factor = height / max_dim
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update()
    
    obj.location.z = height / 2
    print(f"Object centered and scaled to height: {height}")

def fix_rotations(obj, single_frame_render):
    # render single frame with current ssetup, send frame to llm vision
    # approximate with llm vision rotations needed to turn the model with face to the camera + 3/4 towards left
    obj.rotation_euler = (0, 0, 0)
    # rotate for that amount.
    
def apply_transforms(obj):
    print(f"Applying transformations to object: {obj.name}")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Transformations applied to object: {obj.name}")

def parent_obj_to_turn(obj):
    print(f"Parenting object: {obj.name} to turntable")
    empty = bpy.data.objects.get('turn')
    if not empty:
        raise ValueError("Turntable object 'turn' not found in the scene")
    obj.select_set(True)
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
    print(f"Object parented to turntable: {empty.name}")

def assign_given_mat(obj, mat_name, texture_bool):
    print(f"Assigning material: {mat_name} to object: {obj.name}")
    
    # Get the specified material from the scene
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        raise ValueError(f"Material {mat_name} not found in the scene")
    
    # Create a copy of the material
    new_mat = mat.copy()
    new_mat.name = f"{mat_name}_copy"
    
    # Dynamically pick up the GLB imported material and textures
    imported_material = bpy.data.materials.get("Material_0")
    diffuse_texture = bpy.data.images.get("Image_0")
    normal_texture = bpy.data.images.get("Image_1")
    
    if imported_material and diffuse_texture and normal_texture and texture_bool == 'True':
        print(f"Found imported material: {imported_material.name}")
        print(f"Found diffuse texture: {diffuse_texture.name}")
        print(f"Found normal texture: {normal_texture.name}")
        
        new_mat.use_nodes = True
        nodes = new_mat.node_tree.nodes
        links = new_mat.node_tree.links
        
        # Find the ARV_SHAD node group
        arv_shad = nodes.get("ARV_SHAD")
        if arv_shad:
            # Create texture nodes
            tex_diffuse = nodes.new(type='ShaderNodeTexImage')
            tex_diffuse.image = diffuse_texture
            tex_normal = nodes.new(type='ShaderNodeTexImage')
            tex_normal.image = normal_texture
            
            # Create normal map node
            normal_map = nodes.new(type='ShaderNodeNormalMap')
            normal_map.inputs['Strength'].default_value = 1.0
            
            # Connect nodes
            links.new(tex_diffuse.outputs['Color'], arv_shad.inputs['COLOR'])
            links.new(tex_normal.outputs['Color'], normal_map.inputs['Color'])
            links.new(normal_map.outputs['Normal'], arv_shad.inputs['DISP'])
            
            # For the ROUGH input, we'll use the alpha channel of the diffuse texture if it exists
            if tex_diffuse.image.channels == 4:  # Check if the image has an alpha channel
                links.new(tex_diffuse.outputs['Alpha'], arv_shad.inputs['ROUGH'])
            else:
                print("No alpha channel found for roughness, using default value")
    
    # Assign the material to the object
    if obj.data.materials:
        obj.data.materials[0] = new_mat
    else:
        obj.data.materials.append(new_mat)

    print(f"Assigned material: {new_mat.name} to object: {obj.name}")

def set_render_settings(resolution_x=1920, resolution_y=1080, samples=128, render_engine='CYCLES', resolution_percentage=100, camera_name='default_render_CAM'):
    print(f"Setting render settings: resolution ({resolution_x}x{resolution_y}), samples: {samples}, engine: {render_engine}, resolution percentage: {resolution_percentage}, camera: {camera_name}")
    
    scene = bpy.context.scene
    scene.render.resolution_x = resolution_x
    scene.render.resolution_y = resolution_y
    scene.render.resolution_percentage = resolution_percentage

    if render_engine == 'CYCLES':
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        set_gpu_rendering()
    elif render_engine == 'EEVEE':
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.eevee.taa_render_samples = samples
    elif render_engine == 'WORKBENCH':
        scene.render.engine = 'BLENDER_WORKBENCH'
    else:
        raise ValueError(f"Unsupported render engine: {render_engine}")

    # Set the active camera
    camera = bpy.data.objects.get(camera_name)
    if camera:
        scene.camera = camera
        print(f"Camera set to {camera_name}")
    else:
        print(f"Camera {camera_name} not found, using default camera")

    print("Render settings set")

def render_single_frame(output_path, frame=1, file_format='PNG'):
    print(f"Rendering single frame: {frame} to {output_path}")
    scene = bpy.context.scene
    scene.frame_set(frame)
    scene.render.image_settings.file_format = file_format
    scene.render.filepath = os.path.join(output_path, f"frame_{frame:04d}.{file_format.lower()}")
    bpy.ops.render.render(write_still=True)
    print(f"Rendered frame {frame} to {scene.render.filepath}")

def render_sequence(output_path, start_frame=1, end_frame=24, file_format='PNG'):
    print(f"Rendering sequence from frame {start_frame} to {end_frame} to {output_path}")
    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    scene.render.image_settings.file_format = file_format
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        frame_output_path = os.path.join(output_path, f"frame_{frame:04d}.{file_format.lower()}")
        scene.render.filepath = frame_output_path
        bpy.ops.render.render(write_still=True)
        print(f"Rendered frame {frame} to {frame_output_path}")
    print(f"Rendered sequence to {output_path}")
    
def frame_camera_to_obj(obj, zoom_out_factor=1.2):
    print(f"Framing camera to object: {obj.name} with zoom out factor: {zoom_out_factor}")
    camera = bpy.context.scene.camera
    if not camera:
        raise ValueError("No camera found in the scene")

    # Deselect all and select the object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)

    # Set origin to geometry center
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')

    # Frame camera to view selected
    bpy.context.view_layer.objects.active = camera
    bpy.ops.view3d.camera_to_view_selected()

    # Zoom out by adjusting the camera distance or orthographic scale
    if camera.data.type == 'ORTHO':
        camera.data.ortho_scale *= zoom_out_factor
    else:  # Perspective camera
        direction = camera.location - obj.location
        camera.location = obj.location + direction * zoom_out_factor

    bpy.context.view_layer.update()
    print(f"Camera framed to object: {obj.name} and zoomed out by {zoom_out_factor * 100 - 100}%")
    
def save_scene(output_path, base_name):
    # Get the directory one level up from output_path
    parent_directory = os.path.dirname(output_path)
    
    version = 1
    while True:
        versioned_name = f"{base_name}_sceneFile_v{version:03d}.blend"
        full_path = os.path.join(parent_directory, versioned_name)
        if not os.path.exists(full_path):
            break
        version += 1
    
    # Save all data blocks
    bpy.ops.wm.save_as_mainfile(filepath=full_path, copy=True)
    print(f"Scene saved as {full_path}")

def automate_rendering(template_path, obj_path, mtl_name, texture_bool, output_path, height=1.8):
    import_scene_template(template_path)
    imported_objs = import_obj(obj_path)
    
    if isinstance(imported_objs, list):
        valid_objs = [obj for obj in imported_objs if obj.type == 'MESH' and obj.dimensions.x != 0 and obj.dimensions.y != 0 and obj.dimensions.z != 0]
        if not valid_objs:
            raise ValueError("No valid mesh objects found in the imported file")
        for obj in valid_objs:
            center_and_scale_obj(obj, height / len(valid_objs))
            apply_transforms(obj)
            parent_obj_to_turn(obj)
            assign_given_mat(obj, mtl_name, texture_bool)
        main_obj = valid_objs[0]
    else:
        if imported_objs.type != 'MESH' or imported_objs.dimensions.x == 0 or imported_objs.dimensions.y == 0 or imported_objs.dimensions.z == 0:
            raise ValueError("Imported object is not a valid mesh object")
        main_obj = imported_objs
        center_and_scale_obj(main_obj, height)
        apply_transforms(main_obj)
        parent_obj_to_turn(main_obj)
        assign_given_mat(main_obj, mtl_name, texture_bool)
    
    frame_camera_to_obj(main_obj, zoom_out_factor=1.25)  # Adjusted zoom out factor
    
    # Set render settings for Instagram format
    # set_render_settings(
    #     resolution_x=1024, 
    #     resolution_y=1024, 
    #     render_engine='EEVEE', 
    #     samples=32, 
    #     resolution_percentage=200, 
    #     camera_name='square_render_CAM'
    # )
    # Save the scene only once at the end
    save_scene(output_path, os.path.basename(obj_path).split('.')[0])
    
    render_sequence(output_path, start_frame=1, end_frame=24)
    

def run_automation(template_path, obj_path, mtl_name, texture_bool, output_path, height):
    automate_rendering(template_path, obj_path, mtl_name, texture_bool, output_path, height)

if __name__ == "__main__":
    template_path = os.getenv('TEMPLATE_PATH')
    obj_path = os.getenv('OBJ_PATH')
    mtl_name = os.getenv('MTL_NAME')
    output_path = os.getenv('OUTPUT_PATH')
    height = float(os.getenv('HEIGHT'))
    texture_bool = (os.getenv('TEXTURE_BOOL'))

    print(f"Arguments received:\nTemplate Path: {template_path}\nOBJ Path: {obj_path}\nMaterial Name: {mtl_name}\nOutput Path: {output_path}\nHeight: {height}")

    save_blend = os.getenv('SAVE_BLEND', 'False').lower() == 'true'
    
    if save_blend:
        save_scene(output_path, os.path.basename(obj_path).split('.')[0])

    run_automation(template_path, obj_path, mtl_name, texture_bool, output_path, height)