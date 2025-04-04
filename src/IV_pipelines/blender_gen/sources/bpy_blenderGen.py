import bpy
import os
import sys
import gc
from mathutils import Vector

def enable_addons():
    """Enable the required Blender add-ons that are available in this Blender version"""
    # Core add-ons we know exist in recent Blender versions
    standard_addons = [
        "io_scene_gltf2",  # For GLB/GLTF files
        "io_scene_fbx"     # For FBX files
    ]
    
    # Optional add-ons that might be available
    optional_addons = [
        # These use the newer importers via wm operators
        "import_mesh_obj",     # For OBJ files (new method)
        "import_mesh_stl",     # For STL files (new method)
        
        # Legacy or less common add-ons
        "io_import_scene_obj", # Alternative OBJ import
        "io_import_mesh_stl",  # Alternative STL import
        "io_scene_usd",        # For USD files
        "io_scene_3ds"         # For 3DS files
    ]
    
    # First enable core add-ons - these should always work
    for addon in standard_addons:
        try:
            bpy.ops.preferences.addon_enable(module=addon)
            print(f"✓ Enabled {addon} add-on successfully")
        except Exception as e:
            print(f"! Error enabling {addon}: {e}")
    
    # Then try optional add-ons but don't make noise about failures
    enabled_optional = []
    for addon in optional_addons:
        try:
            bpy.ops.preferences.addon_enable(module=addon)
            enabled_optional.append(addon)
        except Exception:
            # Silently continue if the optional add-on isn't available
            pass
    
    if enabled_optional:
        print(f"✓ Also enabled: {', '.join(enabled_optional)}")
    else:
        print("! Note: No optional importers were found")

def cleanup_unused_data():
    """Clean up unused data blocks to free memory"""
    print("Cleaning up unused data...")
    # Force Blender to clean up orphaned data
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
            
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
            
    for block in bpy.data.textures:
        if block.users == 0:
            bpy.data.textures.remove(block)
            
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    
    # Force garbage collection
    gc.collect()
    print("✓ Cleanup complete")

def set_gpu_rendering():
    print("Setting up GPU rendering")
    prefs = bpy.context.preferences
    
    # Make sure cycles add-on exists
    if 'cycles' not in prefs.addons:
        print("! Cycles not found, skipping GPU setup")
        return
        
    cycles_prefs = prefs.addons['cycles'].preferences

    # Check for available GPU types
    if hasattr(cycles_prefs, "compute_device_type"):
        if hasattr(cycles_prefs, "compute_device_type") and cycles_prefs.compute_device_type in {'CUDA', 'OPENCL', 'OPTIX'}:
            cycles_prefs.compute_device_type = 'CUDA'
            bpy.context.scene.cycles.device = 'GPU'
            print("✓ CUDA/OPTIX GPU rendering enabled")
        elif hasattr(cycles_prefs, "compute_device_type") and cycles_prefs.compute_device_type == 'METAL':
            cycles_prefs.compute_device_type = 'METAL'
            bpy.context.scene.cycles.device = 'GPU'
            print("✓ Metal GPU rendering enabled")
        else:
            print("! No compatible GPU found, using CPU rendering")
    else:
        print("! No compute_device_type found, using CPU rendering")

    # Enable all available GPU devices
    if hasattr(cycles_prefs, "devices"):
        gpu_devices = 0
        for device in cycles_prefs.devices:
            if device.type in {'CUDA', 'OPTIX', 'METAL'}:
                device.use = True
                gpu_devices += 1
            else:
                device.use = False
                
        if gpu_devices > 0:
            print(f"✓ Enabled {gpu_devices} GPU device(s)")
            
    # Optimize memory usage for GPU rendering
    if bpy.context.scene.render.engine == 'CYCLES' and bpy.context.scene.cycles.device == 'GPU':
        # Reduce tile size for GPU for better memory usage
        bpy.context.scene.cycles.tile_size = 256
        
        # Set feature set to supported to avoid memory issues
        if hasattr(bpy.context.scene.cycles, "feature_set"):
            bpy.context.scene.cycles.feature_set = 'SUPPORTED'
            
        print("✓ Optimized memory settings for GPU rendering")

def import_scene_template(template_path):
    enable_addons()
    print(f"Importing scene template from: {template_path}")
    bpy.ops.wm.open_mainfile(filepath=template_path)
    if len(bpy.data.objects) == 0:
        raise ValueError(f"Failed to load template from {template_path}. No objects found in the scene.")
    print("✓ Scene template imported successfully")

def import_asset(asset_path):
    """Import any 3D model file based on its extension with improved fallbacks"""
    if not os.path.exists(asset_path):
        raise ValueError(f"File not found: {asset_path}")
    
    print(f"Importing 3D model from: {asset_path}")
    file_extension = os.path.splitext(asset_path)[1][1:].lower()  # Get file extension without the dot
    
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Track objects before import to identify new ones
    objects_before = set(bpy.data.objects)
    
    # Import based on file extension
    if file_extension == 'obj':
        # Try multiple OBJ import methods in order of preference
        try:
            bpy.ops.wm.obj_import(filepath=asset_path)
            print("✓ Imported OBJ file (wm method)")
        except Exception as e:
            try:
                bpy.ops.import_scene.obj(filepath=asset_path)
                print("✓ Imported OBJ file (import_scene method)")
            except Exception as e2:
                try:
                    bpy.ops.import_mesh.obj(filepath=asset_path)
                    print("✓ Imported OBJ file (import_mesh method)")
                except Exception as e3:
                    raise ValueError(f"Could not import OBJ file using any method: {asset_path}")
                
    elif file_extension in ['glb', 'gltf']:
        try:
            bpy.ops.import_scene.gltf(filepath=asset_path)
            print(f"✓ Imported {file_extension.upper()} file")
        except Exception as e:
            raise ValueError(f"Could not import GLTF file: {asset_path}")
            
    elif file_extension == 'fbx':
        try:
            bpy.ops.import_scene.fbx(filepath=asset_path)
            print("✓ Imported FBX file")
        except Exception as e:
            raise ValueError(f"Could not import FBX file: {asset_path}")
            
    elif file_extension in ['usd', 'usda', 'usdc', 'usdz']:
        # Try multiple USD import methods
        success = False
        for import_func in ["wm.usd_import", "import_scene.usd"]:
            try:
                getattr(bpy.ops, import_func)(filepath=asset_path)
                print(f"✓ Imported {file_extension.upper()} file")
                success = True
                break
            except AttributeError:
                # Function doesn't exist, try next
                continue
            except Exception as e:
                print(f"! Error with {import_func}: {e}")
                continue
                
        if not success:
            raise ValueError(f"Could not import USD file: {asset_path}")
                
    elif file_extension in ['blend']:
        try:
            # For blend files, we append objects rather than importing
            with bpy.data.libraries.load(asset_path) as (data_from, data_to):
                data_to.objects = data_from.objects
                
            for obj in data_to.objects:
                if obj is not None:
                    bpy.context.collection.objects.link(obj)
                    obj.select_set(True)
            print("✓ Appended objects from BLEND file")
        except Exception as e:
            raise ValueError(f"Could not append from BLEND file: {asset_path}")
            
    elif file_extension in ['3ds']:
        try:
            bpy.ops.import_scene.autodesk_3ds(filepath=asset_path)
            print("✓ Imported 3DS file")
        except Exception as e:
            raise ValueError(f"Could not import 3DS file: {asset_path}")
            
    elif file_extension in ['stl']:
        # Try multiple STL import methods
        try:
            bpy.ops.import_mesh.stl(filepath=asset_path)
            print("✓ Imported STL file (import_mesh method)")
        except Exception as e:
            try:
                bpy.ops.wm.stl_import(filepath=asset_path)
                print("✓ Imported STL file (wm method)")
            except Exception as e2:
                raise ValueError(f"Could not import STL file: {asset_path}")
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Identify new objects by comparing before and after sets
    new_objects = set(bpy.data.objects) - objects_before
    
    if not new_objects:
        raise ValueError("No objects were imported.")
    
    # Select all new objects
    for obj in new_objects:
        obj.select_set(True)
    
    # Make the first one active
    if new_objects:
        bpy.context.view_layer.objects.active = list(new_objects)[0]
    
    # Apply shade smooth to all imported objects
    for obj in new_objects:
        if obj.type == 'MESH':
            try:
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.shade_smooth()
                obj.select_set(False)
            except Exception:
                # Just continue if shade smooth fails
                pass
    
    # Get the most important object (armature or first mesh)
    armature = None
    mesh = None
    
    for obj in new_objects:
        if obj.type == 'ARMATURE':
            armature = obj
            break
        elif obj.type == 'MESH' and not mesh:
            mesh = obj
    
    # Return either the single object or the list
    main_obj = armature if armature else mesh if mesh else list(new_objects)[0]
    return main_obj

def center_and_scale_obj(obj, height=1.8):
    """Center and scale object, ensuring its bottom rests exactly at z=0 (floor level)"""
    print(f"Centering and scaling object: {obj.name}")
    
    # 1. Ensure we are in object mode and the object is selected/active
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    # 2. Set origin to geometric center for stable scaling
    print("Setting origin to geometry center...")
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # 3. Reset location to world origin
    print("Resetting location...")
    obj.location = (0, 0, 0)
    
    # 4. Apply location transform
    print("Applying location transform...")
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    
    # 6. Calculate scale factor based on current height and desired height
    current_height = obj.dimensions.z
    print(f"Current object height: {current_height}")
    if current_height == 0:
        print(f"Object {obj.name} has zero height, using default scale of 1.0")
        scale_factor = 1.0
    else:
        scale_factor = height / current_height
        print(f"Calculated scale factor: {scale_factor}")
        
    # 7. Apply uniform scale
    print("Applying scale...")
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.context.view_layer.update() # Ensure dimensions are updated before applying
    
    # 8. Apply scale transform
    print("Applying scale transform...")
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    # 9. Calculate the lowest point of the bounding box *after* scaling
    print("Calculating lowest Z point after scaling...")
    # Use world coordinates for bounding box corners
    world_bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_z = min([corner.z for corner in world_bbox_corners])
    print(f"Lowest Z point found at: {min_z}")
    
    # 10. Move the object vertically so its lowest point is at Z=0
    print(f"Moving object vertically by {-min_z} to place on floor...")
    obj.location.z = -min_z
    bpy.context.view_layer.update()
    
    print(f"✓ Object {obj.name} positioned with bottom at z=0 and scaled to height: {height}")
    print(f"  Final location: {obj.location}")
    print(f"  Final dimensions: {obj.dimensions}")
    
    # Deselect object
    obj.select_set(False)

def apply_transforms(obj):
    print(f"Applying transformations to object: {obj.name}")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Transformations applied to object: {obj.name}")

def parent_obj_to_turn(obj):
    """Parent the object directly to the 'turn' empty."""
    print(f"Parenting object: {obj.name} to turntable")
    empty = bpy.data.objects.get('turn')
    if not empty:
        raise ValueError("Turntable object 'turn' not found in the scene")

    # Direct parenting assignment (less context-dependent)
    try:
        obj.parent = empty
        # Optional: Keep transform if needed (usually default for direct parenting)
        # obj.matrix_parent_inverse = empty.matrix_world.inverted()
        print(f"✓ Object '{obj.name}' directly parented to '{empty.name}'")
    except Exception as e:
        # Catch potential errors during direct assignment (though less common)
        print(f"!!! ERROR during direct parenting assignment: {e}")
        active_obj_name = bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else 'None'
        selected_obj_names = [o.name for o in bpy.context.view_layer.objects.selected]
        print(f"    Active object: {active_obj_name}")
        print(f"    Selected objects: {selected_obj_names}")
        raise # Re-raise the error

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

def render_sequence(renders_save_path, start_frame=1, end_frame=24, file_format='PNG'):
    """Render the sequence to the specified renders path"""
    print(f"Rendering sequence from frame {start_frame} to {end_frame}")
    print(f"Render output directory: {renders_save_path} (absolute: {os.path.abspath(renders_save_path)})")
    
    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = end_frame
    
    # Set render format settings
    scene.render.image_settings.file_format = file_format
    
    # Ensure the directory exists
    os.makedirs(renders_save_path, exist_ok=True)
    
    # Set the base output directory for the render
    scene.render.filepath = os.path.join(renders_save_path, "")  # Trailing slash important for dir
    print(f"Global render output path set to: {scene.render.filepath}")
    
    # Set animation output format
    if hasattr(scene.render, 'use_file_extension'):
        scene.render.use_file_extension = True
    
    # Add padding for frame numbers
    scene.render.use_overwrite = True
    scene.render.use_placeholder = True
    if hasattr(scene.render, 'views_format'):
        scene.render.views_format = 'STEREO_3D'
    
    # Save render settings
    if hasattr(bpy.ops.wm, 'save_userpref'):
        bpy.ops.wm.save_userpref()
    
    # Render individual frames for better control and progress reporting
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        
        # Per-frame filepath with frame padding (####)
        frame_output_path = os.path.join(renders_save_path, f"frame_{frame:04d}.{file_format.lower()}")
        print(f"Rendering frame {frame} to: {frame_output_path}")
        
        # Set filepath per frame
        scene.render.filepath = frame_output_path
        
        try:
            # Force Blender to render with the current settings
            bpy.ops.render.render(write_still=True)
            
            # Verify file was created
            if os.path.exists(frame_output_path):
                print(f"✓ Successfully rendered frame {frame} to {frame_output_path}")
            else:
                print(f"!!! WARNING: Render completed but file not found at {frame_output_path}")
                
        except Exception as e:
            print(f"!!! ERROR rendering frame {frame}: {e}")
            # Continue with next frame rather than stopping
            continue
    
    # Reset to the base path
    scene.render.filepath = os.path.join(renders_save_path, "")
    
    # Check for rendered files
    rendered_files = [f for f in os.listdir(renders_save_path) if f.endswith(f'.{file_format.lower()}')]
    print(f"✓ Rendering complete. Found {len(rendered_files)} rendered files in {renders_save_path}")
    
    if len(rendered_files) == 0:
        print("!!! WARNING: No rendered files found in the output directory!")
    else:
        print(f"First rendered file: {rendered_files[0]}")

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
    
def save_scene(scenes_save_path, base_name):
    """Save the scene to the specified scenes path"""
    print(f"Attempting to save scene to: {scenes_save_path}")
    print(f"Absolute path: {os.path.abspath(scenes_save_path)}")
    
    # Use the provided path directly
    output_directory = scenes_save_path 
    
    # Ensure the directory exists
    os.makedirs(output_directory, exist_ok=True)
    dir_contents_before = os.listdir(output_directory) if os.path.exists(output_directory) else []
    print(f"Directory contents before saving: {len(dir_contents_before)} files")
    
    version = 1
    while True:
        versioned_name = f"{base_name}_sceneFile_v{version:03d}.blend"
        full_path = os.path.join(output_directory, versioned_name)
        if not os.path.exists(full_path):
            break
        version += 1
    
    print(f"Will save to: {full_path}")
    
    # Save all data blocks
    try:
        # Change to save as mainfile WITHOUT the copy parameter, so it updates the current file path
        bpy.ops.wm.save_as_mainfile(filepath=full_path)
        
        # Verify file was created
        if os.path.exists(full_path):
            print(f"✓ Scene saved successfully as {full_path}")
            print(f"File size: {os.path.getsize(full_path) / (1024*1024):.2f} MB")
        else:
            print(f"!!! ERROR: File not created at {full_path} despite no errors!")
            
        # List directory contents after save to verify
        dir_contents_after = os.listdir(output_directory)
        new_files = [f for f in dir_contents_after if f not in dir_contents_before]
        print(f"New files in directory after save: {new_files}")
        
        # Print the current .blend filename to verify
        print(f"Current .blend file is now: {bpy.data.filepath}")
        
        return full_path
    except Exception as e:
        print(f"!!! ERROR saving scene: {e}")
        raise

def export_glb(exports_save_path, base_name):
    """Export the scene as GLB to the specified exports path"""
    print(f"Attempting to export GLB to: {exports_save_path}")
    
    version = 1
    while True:
        versioned_name = f"{base_name}_export_v{version:03d}.glb"
        full_path = os.path.join(exports_save_path, versioned_name)
        if not os.path.exists(full_path):
            break
        version += 1
        
    # Ensure the directory exists
    os.makedirs(exports_save_path, exist_ok=True)
    
    try:
        # CRITICAL: Ensure there's an active object in the scene for the exporter context
        if not bpy.context.view_layer.objects.active and len(bpy.data.objects) > 0:
            # Find any mesh object to make active
            mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
            if mesh_objects:
                bpy.context.view_layer.objects.active = mesh_objects[0]
                print(f"Set active object to {mesh_objects[0].name} for export")
            else:
                # If no mesh objects, just pick any object
                bpy.context.view_layer.objects.active = bpy.data.objects[0]
                print(f"Set active object to {bpy.data.objects[0].name} for export")
                
        # Ensure we're in object mode for the export
        try:
            if bpy.ops.object.mode_set.poll():
                bpy.ops.object.mode_set(mode='OBJECT')
                print("Set OBJECT mode for export")
        except Exception as e:
            print(f"Note: Could not set object mode: {e}")
            
        # Make sure the glTF exporter add-on is enabled
        print("Exporting GLB using export_scene.gltf...")
        bpy.ops.export_scene.gltf(
            filepath=full_path, 
            export_format='GLB', 
            use_selection=False  # Export entire scene
        )
        print(f"✓ Scene exported successfully as {full_path}")
        return full_path
    except AttributeError as e:
        print(f"!!! ERROR: GLTF exporter issue: {e}")
        print("Attempting alternate export method...")
        
        # Fallback method: try to export selected objects if the whole scene fails
        try:
            # Select all objects in the scene
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.export_scene.gltf(
                filepath=full_path, 
                export_format='GLB', 
                use_selection=True
            )
            print(f"✓ Scene exported successfully (using selection) as {full_path}")
            return full_path
        except Exception as e2:
            print(f"!!! ERROR: Alternative export method also failed: {e2}")
            raise
    except Exception as e:
        print(f"!!! ERROR exporting GLB: {e}")
        raise

def set_shade_smooth(obj):
    """Set an object to shade smooth and adjust autosmooth settings"""
    print(f"Setting shade smooth for object: {obj.name}")
    
    # Ensure we're in object mode
    if bpy.context.view_layer.objects.active and bpy.context.view_layer.objects.active.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Select only our target object
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    
    try:
        # Apply shade smooth
        bpy.ops.object.shade_smooth()
        
        # Enable autosmooth for better results
        if obj.type == 'MESH':
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = 0.523599  # 30 degrees in radians
            
        print(f"✓ Applied shade smooth to {obj.name}")
    except Exception as e:
        print(f"! Warning: Could not set shade smooth for {obj.name}: {e}")
    finally:
        # Cleanup selection
        obj.select_set(False)

def scene_setup(template_path, asset_path, mtl_name, texture_bool, scenes_save_path, renders_save_path, exports_save_path, height=1.8, interactive=False):
    """Main setup function using specific paths for output"""
    try:
        # Report exact paths being used
        print(f"Working with the following absolute paths:")
        print(f"Template: {os.path.abspath(template_path)}")
        print(f"Asset: {os.path.abspath(asset_path)}")
        print(f"Scenes dir: {os.path.abspath(scenes_save_path)}")
        print(f"Renders dir: {os.path.abspath(renders_save_path)}")
        print(f"Exports dir: {os.path.abspath(exports_save_path)}")
        
        # Import template and model
        import_scene_template(template_path)
        imported_obj = import_asset(asset_path) # Renamed for clarity
        
        # Handle single object or list (though import_asset now returns single main obj)
        # This logic might need refinement based on how import_asset truly behaves with complex scenes
        main_obj = None
        if isinstance(imported_obj, list):
            valid_objs = [obj for obj in imported_obj if obj.type == 'MESH' and obj.dimensions.x != 0 and obj.dimensions.y != 0 and obj.dimensions.z != 0]
            if not valid_objs:
                # Fallback if no valid meshes are found but objects were imported
                valid_objs = [obj for obj in imported_obj if obj.dimensions.x != 0 and obj.dimensions.y != 0 and obj.dimensions.z != 0]
                if not valid_objs:
                   raise ValueError("No valid objects with dimensions found in the imported file")
                   
            # Process all valid objects
            for i, obj in enumerate(valid_objs):
                 # Adjust height distribution if needed
                obj_height = height / len(valid_objs) if len(valid_objs) > 1 else height
                center_and_scale_obj(obj, obj_height)
                apply_transforms(obj)
                parent_obj_to_turn(obj)
                set_shade_smooth(obj)  # Add shade smooth before material
                assign_given_mat(obj, mtl_name, texture_bool)
            main_obj = valid_objs[0] # Still frame based on the first valid one
            
        else: # Single object returned
            # Check if it's a valid object to process
            if not hasattr(imported_obj, 'type') or imported_obj.dimensions.x == 0 or imported_obj.dimensions.y == 0 or imported_obj.dimensions.z == 0:
                 # Attempt to find the first valid mesh in the scene if the returned obj isn't good
                meshes = [obj for obj in bpy.data.objects if obj.type == 'MESH']
                if not meshes:
                    raise ValueError("Imported object is not valid and no other mesh object found")
                imported_obj = meshes[0] # Use the first mesh found
                print(f"Warning: Initial imported object invalid, using first mesh found: {imported_obj.name}")

            main_obj = imported_obj
            center_and_scale_obj(main_obj, height)
            apply_transforms(main_obj)
            parent_obj_to_turn(main_obj)
            set_shade_smooth(main_obj)  # Add shade smooth before material
            assign_given_mat(main_obj, mtl_name, texture_bool)
        
        # Position camera based on the main object
        if main_obj:
            frame_camera_to_obj(main_obj, zoom_out_factor=1.25)
        else:
             print("Warning: No main object identified for camera framing.")

        # Define base name for saving/exporting
        base_name = os.path.basename(asset_path).split('.')[0]
        
        # Save the scene to the 'scenes' folder
        blend_path = save_scene(scenes_save_path, base_name)
        
        # Explicitly verify file paths after saving
        print("\nVERIFYING FILE PATHS:")
        print(f"Current .blend: {bpy.data.filepath}")
        print(f"Scenes path: {os.path.abspath(scenes_save_path)}")
        print(f"Renders path: {os.path.abspath(renders_save_path)}")
        print(f"Exports path: {os.path.abspath(exports_save_path)}")
        
        # For interactive mode, we're done after saving to keep the file open with changes
        if interactive:
            print(f"✓ INTERACTIVE MODE: Scene prepared and saved as: {blend_path}")
            print(f"  You are now working with the saved file - not the template.")
            print(f"  Export GLB manually when ready using File > Export > glTF 2.0 (.glb)")
            # Create a placeholder in exports folder to indicate manual export is needed
            placeholder_path = os.path.join(exports_save_path, f"{base_name}_export_manually.txt")
            os.makedirs(exports_save_path, exist_ok=True)
            with open(placeholder_path, 'w') as f:
                f.write("Please export the GLB file manually from Blender using:\n")
                f.write("File > Export > glTF 2.0 (.glb/.gltf)\n")
                f.write("Recommended settings: Format: glTF Binary (.glb)\n")
            return blend_path, placeholder_path
        
        # Only continue with GLB export and rendering in background mode
        try:
            # Export GLB to the 'exports' folder
            glb_path = export_glb(exports_save_path, base_name)
        except Exception as e:
            print(f"Warning: GLB export failed, but continuing: {e}")
            # Create a placeholder message file to explain
            placeholder_path = os.path.join(exports_save_path, f"{base_name}_export_note.txt")
            with open(placeholder_path, 'w') as f:
                f.write(f"GLB export was attempted but failed due to: {str(e)}\n")
                f.write("You can export the GLB manually from Blender using File > Export > glTF 2.0")
            print(f"Created note at {placeholder_path} about manual export")
            glb_path = placeholder_path
        
        # Ensure the render directory exists
        os.makedirs(renders_save_path, exist_ok=True)
        
        # Render animation to the 'renders' folder
        render_sequence(renders_save_path, start_frame=1, end_frame=30)
        
        # Final cleanup after rendering (only in background mode)
        cleanup_unused_data()
        
        # Final verification of all output files
        print("\n✓ FINAL OUTPUT VERIFICATION:")
        
        # Check blend file
        if os.path.exists(blend_path):
            blend_size = os.path.getsize(blend_path) / (1024*1024)
            print(f"✓ Blend file: {os.path.basename(blend_path)} ({blend_size:.2f} MB)")
        else:
            print(f"!!! ERROR: Blend file not found at {blend_path}")
            
        # Check GLB file
        if os.path.exists(glb_path) and glb_path.endswith('.glb'):
            glb_size = os.path.getsize(glb_path) / (1024*1024)
            print(f"✓ GLB Export: {os.path.basename(glb_path)} ({glb_size:.2f} MB)")
        else:
            print(f"! Note: GLB export not found or is not a GLB file: {glb_path}")
            
        # Check render files
        render_files = [f for f in os.listdir(renders_save_path) if f.endswith('.png')]
        if render_files:
            print(f"✓ Renders: {len(render_files)} files in {renders_save_path}")
            first_render = os.path.join(renders_save_path, render_files[0])
            size = os.path.getsize(first_render) / 1024
            print(f"  First render: {render_files[0]} ({size:.2f} KB)")
        else:
            print(f"!!! ERROR: No render files found in {renders_save_path}")
            
        return blend_path, glb_path # Return both paths
        
    except Exception as e:
        print(f"!!! ERROR in scene_setup: {str(e)}")
        # Optionally save the scene even on error for debugging
        try:
             error_save_path = os.path.join(scenes_save_path, f"{os.path.basename(asset_path).split('.')[0]}_ERROR_STATE.blend")
             bpy.ops.wm.save_as_mainfile(filepath=error_save_path)
             print(f"Saved error state to {error_save_path}")
        except Exception as save_e:
             print(f"Could not save error state: {save_e}")
        raise

if __name__ == "__main__":
    # Read specific paths from environment variables
    template_path = os.getenv('TEMPLATE_PATH')
    asset_path = os.getenv('ASSET_PATH')
    mtl_name = os.getenv('MTL_NAME')
    scenes_save_path = os.getenv('SCENES_SAVE_PATH') 
    renders_save_path = os.getenv('RENDERS_SAVE_PATH')
    exports_save_path = os.getenv('EXPORTS_SAVE_PATH')
    height = float(os.getenv('HEIGHT'))
    texture_bool = os.getenv('TEXTURE_BOOL')
    interactive_mode_str = os.getenv('INTERACTIVE_MODE', 'False')
    interactive_mode = interactive_mode_str.lower() in ('true', '1', 't', 'yes')

    # Validate mandatory paths
    if not all([template_path, asset_path, mtl_name, scenes_save_path, renders_save_path, exports_save_path]):
        print("!!! ERROR: Missing one or more required environment variables:")
        print(f"  TEMPLATE_PATH: {template_path}")
        print(f"  ASSET_PATH: {asset_path}")
        print(f"  MTL_NAME: {mtl_name}")
        print(f"  SCENES_SAVE_PATH: {scenes_save_path}")
        print(f"  RENDERS_SAVE_PATH: {renders_save_path}")
        print(f"  EXPORTS_SAVE_PATH: {exports_save_path}")
        sys.exit(1)

    # Corrected multi-line f-string
    print(f"""Arguments received:
Template Path: {template_path}
Asset Path: {asset_path}
Material Name: {mtl_name}
Scenes Save Path: {scenes_save_path}
Renders Save Path: {renders_save_path}
Exports Save Path: {exports_save_path}
Height: {height}
Texture Bool: {texture_bool}
Interactive Mode: {interactive_mode} (from string value: {interactive_mode_str})""")

    try:
        # save_blend flag is now handled by checking scenes_save_path
        # No need for separate save_blend flag as saving is part of the flow
        
        blend_path, glb_path = scene_setup(
            template_path, 
            asset_path, 
            mtl_name, 
            texture_bool, 
            scenes_save_path,
            renders_save_path,
            exports_save_path,
            height,
            interactive_mode
        )
        
        if interactive_mode:
            print(f"✓ INTERACTIVE MODE: Scene prepared and saved as: {blend_path}")
            print(f"  You are now working with the saved file - not the template.")
            print(f"  Export GLB manually when ready using File > Export > glTF 2.0 (.glb)")
        else:
            print(f"✓ Processing complete! Blend file saved to: {blend_path}. GLB exported to: {glb_path}. Renders saved to: {renders_save_path}")
            
    except Exception as e:
        # Error already printed within scene_setup
        print(f"!!! Main execution block caught error: {str(e)}")
        sys.exit(1)