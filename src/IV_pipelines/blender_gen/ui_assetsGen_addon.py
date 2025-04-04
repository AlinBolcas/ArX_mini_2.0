bl_info = {
    "name": "ArX - AssetsGen",
    "author": "ArX Tools",
    "version": (1, 0, 0),
    "blender": (3, 3, 0),  # Minimum required Blender version
    "location": "View3D > Sidebar > ArX",
    "description": "Generate 3D models from text or images, and create high-quality renders",
    "warning": "Requires internet connection and API keys",
    "doc_url": "",
    "category": "3D View",
}

import bpy
import os
import sys
import time
import threading
import tempfile
import json
import subprocess
import shutil
from pathlib import Path
from bpy.props import (
    StringProperty, 
    BoolProperty,
    EnumProperty, 
    FloatProperty, 
    IntProperty, 
    PointerProperty,
    CollectionProperty,
)
from bpy.types import (
    Panel, 
    Operator, 
    PropertyGroup, 
    UIList, 
    AddonPreferences
)

# For backward compatibility with different Blender versions
try:
    # Blender 4.0+ - previews moved to addon_utils
    from addon_utils import previews
    has_previews = True
except ImportError:
    try:
        # Blender 3.x and earlier
        from bpy.utils import previews
        has_previews = True
    except ImportError:
        # If no previews module is available
        has_previews = False
        print("Warning: No previews module available, icon previews will be disabled")
        
        # Create a fallback previews module
        class DummyPreviewsCollection:
            def __init__(self):
                self.icons = {}
                
            def load(self, name, filepath, type):
                self.icons[name] = 0
                return 0
                
            def remove(self, *args):
                pass
                
            def new(self):
                return DummyPreviewsCollection()
                
        previews = type('DummyPreviews', (), {'new': lambda: DummyPreviewsCollection(), 'remove': lambda x: None})

# Global variables
MODELS_DIR = "models"
RENDERS_DIR = "renders"

# ------------------------------------
# Setup paths for external modules
# ------------------------------------

# Ensure addon dependencies are available
def setup_addon_modules():
    """Add the modules directory to sys.path for importing external dependencies"""
    # Get the directory of this file
    addon_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Check for 'modules' subdirectory
    modules_dir = os.path.join(addon_dir, "modules")
    
    # Create if it doesn't exist
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir, exist_ok=True)
    
    # Add to path if not already there
    if modules_dir not in sys.path:
        sys.path.append(modules_dir)

# Run setup
setup_addon_modules()

# Import requirements checker
class InstallDependencies(Operator):
    """Install required Python dependencies"""
    bl_idname = "arx.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required Python packages for ArX AssetsGen"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        try:
            import pip
            from pip._internal import main as pip_main
        except ImportError:
            self.report({'ERROR'}, "Pip not available. Please install pip first.")
            return {'CANCELLED'}
        
        addon_dir = os.path.dirname(os.path.realpath(__file__))
        modules_dir = os.path.join(addon_dir, "modules")
        
        # Create requirements file
        requirements = [
            "requests",
            "Pillow",
            "python-dotenv"
        ]
        
        requirements_file = os.path.join(addon_dir, "requirements.txt")
        with open(requirements_file, 'w') as f:
            f.write('\n'.join(requirements))
        
        # Install using pip to the modules directory
        try:
            pip_args = [
                "install",
                "--target", modules_dir,
                "-r", requirements_file
            ]
            pip_main(pip_args)
            self.report({'INFO'}, "Successfully installed dependencies")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error installing dependencies: {e}")
            return {'CANCELLED'}

# ------------------------------------
# API Integration (Placeholder)
# ------------------------------------

# In a production version, these would be proper implementations
# using the actual APIs from external files

class APIManager:
    """Manager for API interactions"""
    
    @staticmethod
    def generate_images_from_prompt(prompt, num_images=3):
        """Generate images from text prompt
        
        In production, this would call the actual API
        """
        # Create temporary image files
        image_paths = []
        for i in range(num_images):
            # In a real implementation, this would call the Replicate API
            # and save the resulting images
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_file.close()
            
            # Fill with dummy data for preview
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (512, 512), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)
                draw.text((20, 20), f"Generated from: {prompt}", fill=(200, 200, 200))
                draw.text((20, 50), f"Image {i+1}", fill=(200, 200, 200))
                draw.rectangle([(50, 100), (450, 400)], outline=(100, 100, 200))
                img.save(temp_file.name)
            except ImportError:
                # If PIL not available
                with open(temp_file.name, "wb") as f:
                    f.write(b"Dummy image data")
            
            image_paths.append(temp_file.name)
        
        return image_paths
    
    @staticmethod
    def generate_3d_model(image_path, model_type):
        """Generate 3D model from image
        
        In production, this would call Tripo/Trellis/etc API
        """
        # Create temporary model file
        temp_file = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        temp_file.close()
        
        # Create a dummy file for preview
        with open(temp_file.name, "wb") as f:
            f.write(b"Dummy GLB data")
        
        return temp_file.name

# ------------------------------------
# Addon Preferences
# ------------------------------------

class ArXAssetsGenPreferences(AddonPreferences):
    bl_idname = __name__
    
    openai_api_key: StringProperty(
        name="OpenAI API Key",
        description="API key for OpenAI services",
        default="",
        subtype="PASSWORD"
    )
    
    replicate_api_key: StringProperty(
        name="Replicate API Key",
        description="API key for Replicate services",
        default="",
        subtype="PASSWORD"
    )
    
    tripo_api_key: StringProperty(
        name="Tripo API Key",
        description="API key for Tripo 3D services",
        default="",
        subtype="PASSWORD"
    )
    
    output_dir: StringProperty(
        name="Output Directory",
        description="Directory to save generated files",
        default="//ArX_Output",  # Relative to blend file
        subtype="DIR_PATH"
    )
    
    def draw(self, context):
        layout = self.layout
        
        # API Keys
        box = layout.box()
        box.label(text="API Keys", icon='KEY')
        
        row = box.row()
        row.prop(self, "openai_api_key")
        
        row = box.row()
        row.prop(self, "replicate_api_key")
        
        row = box.row()
        row.prop(self, "tripo_api_key")
        
        # Output directory
        box = layout.box()
        box.label(text="Paths", icon='FOLDER')
        box.prop(self, "output_dir")
        
        # Dependencies
        box = layout.box()
        box.label(text="Dependencies", icon='PACKAGE')
        box.operator("arx.install_dependencies")

# ------------------------------------
# Property Groups
# ------------------------------------

class ArXGeneratedImage(PropertyGroup):
    """Properties for a generated image"""
    filepath: StringProperty(
        name="Image Path",
        description="Path to the generated image",
        default="",
        subtype="FILE_PATH"
    )
    
    is_selected: BoolProperty(
        name="Selected",
        description="Whether this image is selected",
        default=False
    )

class ArXAssetsGenProperties(PropertyGroup):
    """Property group for ArX AssetsGen"""
    
    # Concept & Images Stage
    prompt: StringProperty(
        name="Prompt",
        description="Text prompt for image generation",
        default=""
    )
    
    refine_prompt: BoolProperty(
        name="Refine Prompt",
        description="Use AI to refine the prompt",
        default=False
    )
    
    refined_prompt: StringProperty(
        name="Refined Prompt",
        description="AI-refined prompt",
        default=""
    )
    
    generated_images: CollectionProperty(
        type=ArXGeneratedImage,
        name="Generated Images",
        description="Collection of generated images"
    )
    
    selected_image_index: IntProperty(
        name="Selected Image Index",
        description="Index of the selected image",
        default=-1
    )
    
    # 3D Model Generation Stage
    model_type: EnumProperty(
        name="Model Type",
        description="Type of 3D model to generate",
        items=[
            ('TRIPO', "Tripo", "Use Tripo for 3D model generation"),
            ('TRELLIS', "Trellis", "Use Trellis for 3D model generation"),
            ('HUNYUAN', "Hunyuan", "Use Hunyuan for 3D model generation")
        ],
        default='TRIPO'
    )
    
    model_path: StringProperty(
        name="Model Path",
        description="Path to the generated 3D model",
        default="",
        subtype="FILE_PATH"
    )
    
    # Rendering Stage
    interactive_mode: BoolProperty(
        name="Interactive Mode",
        description="Enable interactive mode for rendering",
        default=False
    )
    
    render_path: StringProperty(
        name="Render Path",
        description="Path to the rendered output",
        default="",
        subtype="FILE_PATH"
    )
    
    # Processing Status
    is_processing: BoolProperty(
        name="Is Processing",
        description="Whether a process is currently running",
        default=False
    )
    
    progress: FloatProperty(
        name="Progress",
        description="Progress of the current operation",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="PERCENTAGE"
    )
    
    status_message: StringProperty(
        name="Status Message",
        description="Current status message",
        default="Ready"
    )

# ------------------------------------
# Operators
# ------------------------------------

class ARX_OT_GenerateImages(Operator):
    """Generate images from text prompt"""
    bl_idname = "arx.generate_images"
    bl_label = "Generate Images"
    bl_description = "Generate images from the provided text prompt"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.arx_assets_gen
        
        if not props.prompt:
            self.report({'ERROR'}, "Please enter a prompt first")
            return {'CANCELLED'}
        
        # Start processing
        props.is_processing = True
        props.progress = 0.0
        props.status_message = "Generating images..."
        
        # Clear existing images
        props.generated_images.clear()
        
        # Start thread for processing
        def process_images():
            try:
                # Step 1: Refine prompt if needed
                prompt_to_use = props.prompt
                if props.refine_prompt:
                    # In a real implementation, this would call the OpenAI API
                    # to refine the prompt
                    props.refined_prompt = f"Refined: {props.prompt}"
                    prompt_to_use = props.refined_prompt
                    props.progress = 0.3
                
                # Step 2: Generate images
                image_paths = APIManager.generate_images_from_prompt(
                    prompt_to_use, num_images=3
                )
                props.progress = 0.8
                
                # Step 3: Add images to the collection
                for i, path in enumerate(image_paths):
                    item = props.generated_images.add()
                    item.filepath = path
                    item.is_selected = (i == 0)  # Select the first image
                
                if len(props.generated_images) > 0:
                    props.selected_image_index = 0
                
                props.progress = 1.0
                props.status_message = "Images generated successfully"
            except Exception as e:
                props.status_message = f"Error: {str(e)}"
            finally:
                props.is_processing = False
                # Force a redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        
        # Start thread
        threading.Thread(target=process_images, daemon=True).start()
        
        return {'FINISHED'}

class ARX_OT_SelectImage(Operator):
    """Select an image from the generated images"""
    bl_idname = "arx.select_image"
    bl_label = "Select Image"
    bl_description = "Select this image for 3D model generation"
    bl_options = {'REGISTER', 'UNDO'}
    
    index: IntProperty(default=0)
    
    def execute(self, context):
        props = context.scene.arx_assets_gen
        
        # Make sure index is valid
        if self.index < 0 or self.index >= len(props.generated_images):
            self.report({'ERROR'}, "Invalid image index")
            return {'CANCELLED'}
        
        # Update selection
        for i, img in enumerate(props.generated_images):
            img.is_selected = (i == self.index)
        
        props.selected_image_index = self.index
        
        return {'FINISHED'}

class ARX_OT_ImportImage(Operator):
    """Import an image from disk"""
    bl_idname = "arx.import_image"
    bl_label = "Import Image"
    bl_description = "Import an image from your computer"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the image file",
        default="",
        subtype="FILE_PATH"
    )
    
    filter_glob: StringProperty(
        default="*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp",
        options={'HIDDEN'}
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        
        props = context.scene.arx_assets_gen
        
        # Create a copy of the image in a temporary location
        temp_file = tempfile.NamedTemporaryFile(suffix=os.path.splitext(self.filepath)[1], delete=False)
        temp_file.close()
        
        try:
            shutil.copy2(self.filepath, temp_file.name)
        except Exception as e:
            self.report({'ERROR'}, f"Error copying file: {e}")
            return {'CANCELLED'}
        
        # Clear existing images and add the imported one
        props.generated_images.clear()
        item = props.generated_images.add()
        item.filepath = temp_file.name
        item.is_selected = True
        props.selected_image_index = 0
        
        return {'FINISHED'}

class ARX_OT_Generate3DModel(Operator):
    """Generate a 3D model from the selected image"""
    bl_idname = "arx.generate_3d_model"
    bl_label = "Generate 3D Model"
    bl_description = "Generate a 3D model from the selected image"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.arx_assets_gen
        
        # Check if an image is selected
        if props.selected_image_index < 0 or props.selected_image_index >= len(props.generated_images):
            self.report({'ERROR'}, "Please select an image first")
            return {'CANCELLED'}
        
        # Get selected image path
        selected_image = props.generated_images[props.selected_image_index]
        image_path = selected_image.filepath
        
        if not image_path or not os.path.exists(image_path):
            self.report({'ERROR'}, "Selected image file not found")
            return {'CANCELLED'}
        
        # Start processing
        props.is_processing = True
        props.progress = 0.0
        props.status_message = f"Generating {props.model_type} 3D model..."
        
        # Start thread for processing
        def process_model():
            try:
                # Generate 3D model
                model_path = APIManager.generate_3d_model(
                    image_path, 
                    props.model_type
                )
                
                # Update properties
                props.model_path = model_path
                props.progress = 1.0
                props.status_message = "3D model generated successfully"
                
                # Import the model into Blender
                bpy.ops.wm.import_scene('EXEC_DEFAULT', filepath=model_path)
                
            except Exception as e:
                props.status_message = f"Error: {str(e)}"
            finally:
                props.is_processing = False
                # Force a redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        
        # Start thread
        threading.Thread(target=process_model, daemon=True).start()
        
        return {'FINISHED'}

class ARX_OT_RenderOutput(Operator):
    """Render the 3D model to create output animations"""
    bl_idname = "arx.render_output"
    bl_label = "Render Outputs"
    bl_description = "Render the 3D model to create output animations"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.arx_assets_gen
        
        # Check if a model is loaded
        if not props.model_path or not os.path.exists(props.model_path):
            self.report({'ERROR'}, "Please generate a 3D model first")
            return {'CANCELLED'}
        
        # Make sure we have something in the scene
        if len(bpy.context.scene.objects) == 0:
            self.report({'ERROR'}, "No objects in scene to render")
            return {'CANCELLED'}
        
        # Prepare rendering settings
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.film_transparent = True
        
        # Create a camera if none exists
        if not any(obj.type == 'CAMERA' for obj in bpy.context.scene.objects):
            bpy.ops.object.camera_add(location=(0, -5, 2), rotation=(1.2, 0, 0))
            cam = bpy.context.object
            bpy.context.scene.camera = cam
        
        # Create a light if none exists
        if not any(obj.type == 'LIGHT' for obj in bpy.context.scene.objects):
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 4))
            light = bpy.context.object
            light.data.energy = 5.0
        
        # Set up output path
        prefs = context.preferences.addons[__name__].preferences
        output_dir = bpy.path.abspath(prefs.output_dir)
        renders_dir = os.path.join(output_dir, RENDERS_DIR)
        os.makedirs(renders_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        render_path = os.path.join(renders_dir, f"render_{timestamp}")
        
        # Start processing
        props.is_processing = True
        props.progress = 0.0
        props.status_message = "Setting up rendering..."
        
        # Start thread for processing
        def process_render():
            try:
                # Set up rendering parameters
                bpy.context.scene.render.filepath = render_path
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                
                # Create animation (simple turntable)
                # Get all mesh objects
                mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
                
                if mesh_objects:
                    # Create empty for rotation if needed
                    empty = None
                    for obj in bpy.context.scene.objects:
                        if obj.name == "ArX_Turntable":
                            empty = obj
                            break
                    
                    if not empty:
                        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
                        empty = bpy.context.object
                        empty.name = "ArX_Turntable"
                    
                    # Parent mesh objects to empty
                    for obj in mesh_objects:
                        obj.select_set(True)
                        bpy.context.view_layer.objects.active = obj
                    
                    bpy.context.view_layer.objects.active = empty
                    bpy.ops.object.parent_set(type='OBJECT')
                    
                    # Add rotation animation
                    empty.rotation_euler = (0, 0, 0)
                    empty.keyframe_insert(data_path="rotation_euler", frame=1)
                    
                    empty.rotation_euler = (0, 0, 6.28319)  # 360 degrees
                    empty.keyframe_insert(data_path="rotation_euler", frame=48)
                    
                    # Set up animation settings
                    bpy.context.scene.frame_start = 1
                    bpy.context.scene.frame_end = 48
                    
                    # Render animation or just set up scene
                    if not props.interactive_mode:
                        # Render animation
                        for frame in range(1, 49):
                            bpy.context.scene.frame_set(frame)
                            bpy.context.scene.render.filepath = f"{render_path}_{frame:04d}"
                            bpy.ops.render.render(write_still=True)
                            props.progress = frame / 48
                            props.status_message = f"Rendering frame {frame}/48"
                        
                        # Create GIF from the rendered frames
                        gif_path = f"{render_path}.gif"
                        
                        # In a real implementation, this would use the images_to_gif utility
                        # For now, we'll just create a dummy file
                        with open(gif_path, "wb") as f:
                            f.write(b"Dummy GIF data")
                        
                        props.render_path = gif_path
                        
                        props.progress = 1.0
                        props.status_message = "Rendering complete"
                    else:
                        # Just set up the scene for interactive rendering
                        props.progress = 1.0
                        props.status_message = "Scene set up for interactive rendering"
                
            except Exception as e:
                props.status_message = f"Error: {str(e)}"
            finally:
                props.is_processing = False
                # Force a redraw
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        
        # Start thread
        threading.Thread(target=process_render, daemon=True).start()
        
        return {'FINISHED'}

class ARX_OT_OpenOutputFolder(Operator):
    """Open the output folder in the file explorer"""
    bl_idname = "arx.open_output_folder"
    bl_label = "Open Output Folder"
    bl_description = "Open the output folder in your system's file explorer"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        output_dir = bpy.path.abspath(prefs.output_dir)
        
        # Make sure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the directory with the appropriate command for the current platform
        if sys.platform == 'win32':  # Windows
            os.startfile(output_dir)
        elif sys.platform == 'darwin':  # macOS
            subprocess.Popen(['open', output_dir])
        else:  # Linux and other platforms
            subprocess.Popen(['xdg-open', output_dir])
        
        return {'FINISHED'}

class ARX_OT_ImportModel(Operator):
    """Import a 3D model from disk"""
    bl_idname = "arx.import_model"
    bl_label = "Import Model"
    bl_description = "Import an existing 3D model from your computer"
    bl_options = {'REGISTER', 'UNDO'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to the model file",
        default="",
        subtype="FILE_PATH"
    )
    
    filter_glob: StringProperty(
        default="*.glb;*.fbx;*.obj;*.blend",
        options={'HIDDEN'}
    )
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        if not self.filepath:
            self.report({'ERROR'}, "No file selected")
            return {'CANCELLED'}
        
        props = context.scene.arx_assets_gen
        
        # Create output directory if it doesn't exist
        prefs = context.preferences.addons[__name__].preferences
        output_dir = bpy.path.abspath(prefs.output_dir)
        models_dir = os.path.join(output_dir, MODELS_DIR)
        os.makedirs(models_dir, exist_ok=True)
        
        # Copy the model file to our models directory
        filename = os.path.basename(self.filepath)
        dest_path = os.path.join(models_dir, filename)
        
        try:
            shutil.copy2(self.filepath, dest_path)
            self.report({'INFO'}, f"Copied model to {dest_path}")
        except Exception as e:
            self.report({'ERROR'}, f"Error copying file: {e}")
            return {'CANCELLED'}
        
        # Update model path
        props.model_path = dest_path
        props.status_message = f"Model loaded: {filename}"
        
        # Import the model into Blender
        bpy.ops.wm.import_scene('EXEC_DEFAULT', filepath=dest_path)
        
        return {'FINISHED'}

# ------------------------------------
# UI Panels
# ------------------------------------

class ARX_PT_ConceptPanel(Panel):
    """Panel for concept and image generation"""
    bl_label = "1. Concept & Images"
    bl_idname = "ARX_PT_ConceptPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ArX'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.arx_assets_gen
        
        # Prompt input
        box = layout.box()
        box.label(text="Concept or Prompt")
        box.prop(props, "prompt", text="")
        
        # Refine toggle
        row = box.row()
        row.prop(props, "refine_prompt", text="Use AI to refine prompt")
        
        # Refined prompt (if available)
        if props.refined_prompt:
            box.label(text="Refined Prompt:")
            box.label(text=props.refined_prompt)
        
        # Generate button
        generate_row = layout.row(align=True)
        generate_row.scale_y = 1.5
        generate_row.operator("arx.generate_images", icon='IMAGE')
        
        # Import image button
        import_row = layout.row(align=True)
        import_row.operator("arx.import_image", icon='IMPORT')
        
        # Progress bar
        if props.is_processing:
            layout.prop(props, "progress", text="")
        
        # Images display
        if len(props.generated_images) > 0:
            box = layout.box()
            box.label(text="Generated Images")
            
            # Grid for images - 3 columns
            grid = box.grid_flow(row_major=True, columns=3, even_columns=True)
            
            for i, img in enumerate(props.generated_images):
                col = grid.column()
                col.template_icon(icon_value=self.get_preview_icon(context, img.filepath))
                
                select_op = col.operator("arx.select_image", text="Select" if not img.is_selected else "Selected")
                select_op.index = i
                
                # Highlight selected image
                if img.is_selected:
                    col.operator("arx.select_image", text="", icon='CHECKMARK')
    
    def get_preview_icon(self, context, filepath):
        """Get preview icon for image"""
        if not filepath or not os.path.exists(filepath):
            return 0  # Default icon
        
        try:
            # Try to use Blender's preview system
            icons = bpy.data.previews.get("arx_assets_gen_previews")
            
            if icons is None:
                # Use previews module through our compatibility layer
                icons = previews.new()
                bpy.data.previews["arx_assets_gen_previews"] = icons
            
            # Generate a unique key for this image path
            key = os.path.basename(filepath)
            
            if key not in icons:
                icons.load(key, filepath, 'IMAGE')
            
            return icons[key].icon_id
            
        except Exception as e:
            print(f"Error loading preview: {e}")
            return 0  # Default icon

class ARX_PT_ModelPanel(Panel):
    """Panel for 3D model generation"""
    bl_label = "2. 3D Model Generation"
    bl_idname = "ARX_PT_ModelPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ArX'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.arx_assets_gen
        
        # Model type selection
        box = layout.box()
        box.label(text="Model Type")
        box.prop(props, "model_type", text="")
        
        # Selected image
        box = layout.box()
        box.label(text="Selected Image")
        
        if props.selected_image_index >= 0 and props.selected_image_index < len(props.generated_images):
            img = props.generated_images[props.selected_image_index]
            box.template_icon(icon_value=self.get_preview_icon(context, img.filepath))
            box.label(text=os.path.basename(img.filepath))
        else:
            box.label(text="No image selected")
        
        # Buttons row - Generate and Import buttons
        row = layout.row(align=True)
        row.scale_y = 1.5
        
        # Generate button
        col = row.column()
        col.scale_x = 0.7
        generate_button = col.operator("arx.generate_3d_model", icon='MESH_MONKEY', text="Generate 3D Model")
        
        # Import model button
        col = row.column()
        col.scale_x = 0.3
        import_button = col.operator("arx.import_model", icon='IMPORT', text="Import")
        
        # Progress bar
        if props.is_processing:
            layout.prop(props, "progress", text="")
        
        # Model info (if available)
        if props.model_path:
            box = layout.box()
            box.label(text="Generated Model")
            box.label(text=os.path.basename(props.model_path))
    
    def get_preview_icon(self, context, filepath):
        """Get preview icon for image"""
        if not filepath or not os.path.exists(filepath):
            return 0  # Default icon
        
        try:
            # Try to use Blender's preview system
            icons = bpy.data.previews.get("arx_assets_gen_previews")
            
            if icons is None:
                # Use previews module through our compatibility layer
                icons = previews.new()
                bpy.data.previews["arx_assets_gen_previews"] = icons
            
            # Generate a unique key for this image path
            key = os.path.basename(filepath)
            
            if key not in icons:
                icons.load(key, filepath, 'IMAGE')
            
            return icons[key].icon_id
            
        except Exception as e:
            print(f"Error loading preview: {e}")
            return 0  # Default icon

class ARX_PT_RenderPanel(Panel):
    """Panel for rendering"""
    bl_label = "3. Rendering & Output"
    bl_idname = "ARX_PT_RenderPanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ArX'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.arx_assets_gen
        
        # Interactive mode toggle
        box = layout.box()
        box.prop(props, "interactive_mode", text="Interactive Mode")
        
        # Help text
        if props.interactive_mode:
            box.label(text="Scene will be set up for manual rendering")
        else:
            box.label(text="Full animation will be rendered automatically")
        
        # Render button
        render_row = layout.row(align=True)
        render_row.scale_y = 1.5
        render_row.operator("arx.render_output", icon='RENDER_ANIMATION')
        
        # Progress bar
        if props.is_processing:
            layout.prop(props, "progress", text="")
        
        # Output info (if available)
        if props.render_path:
            box = layout.box()
            box.label(text="Rendered Output")
            box.label(text=os.path.basename(props.render_path))
        
        # Open folder button
        layout.operator("arx.open_output_folder", icon='FOLDER')
        
        # Status message
        if props.status_message:
            layout.label(text=f"Status: {props.status_message}")

# ------------------------------------
# Registration
# ------------------------------------

classes = (
    InstallDependencies,
    ArXAssetsGenPreferences,
    ArXGeneratedImage,
    ArXAssetsGenProperties,
    ARX_OT_GenerateImages,
    ARX_OT_SelectImage,
    ARX_OT_ImportImage,
    ARX_OT_Generate3DModel,
    ARX_OT_RenderOutput,
    ARX_OT_OpenOutputFolder,
    ARX_OT_ImportModel,
    ARX_PT_ConceptPanel,
    ARX_PT_ModelPanel,
    ARX_PT_RenderPanel,
)

preview_collections = {}

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register properties
    bpy.types.Scene.arx_assets_gen = PointerProperty(type=ArXAssetsGenProperties)
    
    # Create preview collection for thumbnails using our compatibility layer
    if has_previews:
        preview_collections["main"] = previews.new()

def unregister():
    # Remove preview collection and all thumbnails
    if has_previews:
        for pcoll in preview_collections.values():
            previews.remove(pcoll)
    preview_collections.clear()
    
    # Unregister all properties
    del bpy.types.Scene.arx_assets_gen
    
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()