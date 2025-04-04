import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import sys
import io
import asyncio
import time
import shutil
import traceback
import subprocess
from pathlib import Path
from PIL import Image, ImageTk

# Calculate the project root
project_root = Path(__file__).resolve().parents[3]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

# Import necessary modules
try:
    # Blender Gen imports
    from src.IV_pipelines.blender_gen.blenderGen import BlenderGen
    
    # API imports - make sure we can import Tripo API with multiple fallbacks
    from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
    from src.I_integrations.replicate_API import ReplicateAPI
    
    # Try multiple import paths for Tripo API
    try:
        from src.I_integrations.tripo_API import TripoAPI
        print("Imported Tripo API from regular path")
    except ImportError:
        try:
            from I_integrations.tripo_API import TripoAPI
            print("Imported Tripo API from path without src prefix")
        except ImportError:
            try:
                from src.I_integrations.archive.tripo_API import TripoAPI
                print("Imported Tripo API from archive folder")
            except ImportError:
                print("WARNING: Failed to import TripoAPI, 3D generation might be limited")
                TripoAPI = None
    
    # Utilities imports - with fallbacks
    try:
        from src.VI_utils.video_utils import video_to_gif, image_to_video
        from src.VI_utils.image_utils import images_to_gif, get_image_info
    except ImportError as e:
        print(f"Error importing utility functions: {e}")
        # Define fallback functions if needed
        def images_to_gif(input_paths, output_path, duration=0.1, resize_factor=100.0, optimize=True, loop=0, max_colors=256):
            print("WARNING: Using fallback images_to_gif function!")
            try:
                # Simple implementation using PIL
                frames = []
                for path in input_paths:
                    img = Image.open(path)
                    if resize_factor != 100.0:
                        w, h = img.size
                        new_size = (int(w * resize_factor / 100), int(h * resize_factor / 100))
                        img = img.resize(new_size, Image.LANCZOS)
                    frames.append(img)
                
                if frames:
                    frames[0].save(
                        output_path, 
                        save_all=True, 
                        append_images=frames[1:], 
                        duration=int(duration * 1000), 
                        loop=loop,
                        optimize=optimize
                    )
                    return True
                return False
            except Exception as e:
                print(f"Error in fallback images_to_gif: {e}")
                return False
                
        def video_to_gif(input_path, output_path, fps=10, resize_factor=100.0, quality=85, optimize_size=True):
            print("WARNING: video_to_gif fallback not implemented")
            return False
            
        def image_to_video(frame_paths, output_path, fps=24, resolution=None, crf=20, preset="medium"):
            print("WARNING: image_to_video fallback not implemented")
            return False
            
        def get_image_info(image_path):
            try:
                with Image.open(image_path) as img:
                    return {
                        'width': img.width,
                        'height': img.height,
                        'format': img.format,
                        'mode': img.mode
                    }
            except Exception as e:
                print(f"Error in fallback get_image_info: {e}")
                return {'error': str(e)}
except ImportError as e:
    # Fallback for direct directory run
    print(f"Error importing modules: {e}")
    try:
        from blenderGen import BlenderGen
        # NOTE: Other imports may fail if running directly from the blender_gen directory
    except ImportError:
        print("Critical import error. Please run from the project root.")

class BlenderRunner:
    """Utility class to handle Blender rendering"""
    def __init__(self, blend_file, model_path, project_dir, material="blue_procedural_MAT", 
                 model_height=1.8, use_textures=True, interactive=False):
        self.blend_file = blend_file
        self.model_path = model_path
        self.project_dir = project_dir
        self.material = material
        self.model_height = model_height
        self.use_textures = use_textures
        self.interactive = interactive
        
    def run(self):
        """Run Blender with the given model and settings"""
        try:
            # Initialize BlenderGen
            blender_automation = BlenderGen()
            
            # Ensure needed directories exist
            os.makedirs(self.project_dir, exist_ok=True)
            
            # Run the pipeline
            result_dir = blender_automation.blenderGen_pipeline(
                template_path=self.blend_file,
                asset_path=self.model_path,
                mtl_name=self.material,
                texture_bool=self.use_textures,
                output_path=self.project_dir,
                height=self.model_height,
                interactive=self.interactive
            )
            
            if result_dir:
                print(f"Blender completed successfully: {result_dir}")
                return True, result_dir
            else:
                print("Blender process did not return a result directory")
                return False, None
                
        except Exception as e:
            print(f"Error running Blender: {str(e)}")
            traceback.print_exc()
            return False, None

class ComprehensiveGenUI:
    def __init__(self, root):
        self.root = root
        self.root.title("BlenderGen Pipeline UI")
        self.root.geometry("1000x800")  # Larger window for more content
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # State variables
        self.generated_images = []  # Store generated image paths/urls
        self.selected_image_index = None  # Track selected image
        self.generated_3d_model_path = None  # Store path to generated 3D model
        self.output_dir = None  # Base directory for outputs
        self.project_folder = None  # Blender project folder
        self.renders_folder = None  # Path to renders

        # Initialize default output directory
        self.output_dir = project_root / "data" / "output" / "blender_pipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if template exists in the user-requested location first, then fall back to other options
        user_template = project_root / "data/input/system_config/blender_templates/blenderGen_presentation_03.blend"
        default_template = project_root / "data/templates/blenderGen_presentation_03.blend"
        alt_template = project_root / "data/input/templates/blenderGen_presentation_03.blend"
        hardcoded_path = "/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_presentation_03.blend"
        
        # Initialize template path - try various options
        if user_template.exists():
            self.template_path = str(user_template)
            print(f"Using template from requested location: {user_template}")
        elif default_template.exists():
            self.template_path = str(default_template)
            print(f"Using default template at: {default_template}")
        elif alt_template.exists():
            self.template_path = str(alt_template)
            print(f"Found template at alternate location: {alt_template}")
        elif os.path.exists(hardcoded_path):
            self.template_path = hardcoded_path
            print(f"Using hardcoded template path: {hardcoded_path}")
        else:
            # No template found, but set a path to the user-requested location anyway
            self.template_path = str(user_template)
            print(f"WARNING: No template found. Using user-requested path that may not exist: {user_template}")
        
        # Hardcoded defaults for Blender
        self.material_name = "blue_procedural_MAT"  # Changed from grey to blue
        self.height_value = 1.8
        
        # Initialize API clients (on demand to avoid keys not being set)
        self.openai_client = None
        self.replicate_client = None 
        self.tripo_client = None
        
        # Create main container with tabs
        self.tab_view = ctk.CTkTabview(self.root)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add only 3 tabs now
        self.tab_view.add("1. Concept & Images")
        self.tab_view.add("2. 3D Model")
        self.tab_view.add("3. Output")
        
        # Make tabs fill width
        self.tab_view.tab("1. Concept & Images").grid_columnconfigure(0, weight=1)
        self.tab_view.tab("2. 3D Model").grid_columnconfigure(0, weight=1)
        self.tab_view.tab("3. Output").grid_columnconfigure(0, weight=1)
        
        # Set default tab
        self.tab_view.set("1. Concept & Images")
        
        # Initialize UI components
        self.setup_concept_tab()
        self.setup_3d_model_tab()
        self.setup_output_tab()
        self.setup_progress_area()

    def setup_concept_tab(self):
        """Setup the concept and image generation tab"""
        tab = self.tab_view.tab("1. Concept & Images")
        
        # Concept Input Frame
        concept_frame = ctk.CTkFrame(tab)
        concept_frame.pack(fill="x", padx=10, pady=10)
        
        # Tab selection - choose between text prompt or existing image
        tab_select_frame = ctk.CTkFrame(concept_frame)
        tab_select_frame.pack(fill="x", padx=5, pady=5)
        
        self.input_mode_var = tk.StringVar(value="prompt")
        
        # Create segmented button for tab selection
        input_modes = ctk.CTkSegmentedButton(
            tab_select_frame,
            values=["Generate from Prompt", "Use Existing Image"],
            command=self.switch_input_mode,
            variable=self.input_mode_var
        )
        input_modes.pack(fill="x", padx=5, pady=5)
        
        # Prompt Input Frame (initially visible)
        self.prompt_input_frame = ctk.CTkFrame(concept_frame)
        self.prompt_input_frame.pack(fill="x", padx=5, pady=5)
        
        # Concept/Prompt Input
        input_label = ctk.CTkLabel(self.prompt_input_frame, text="Concept or Prompt:", anchor="w")
        input_label.pack(fill="x", padx=5, pady=5)
        
        self.concept_entry = ctk.CTkTextbox(self.prompt_input_frame, height=80)
        self.concept_entry.pack(fill="x", padx=5, pady=5)
        self.concept_entry.insert("1.0", "Enter a concept or detailed description of what you want to create...")
        
        # Options frame with checkboxes and buttons
        options_frame = ctk.CTkFrame(self.prompt_input_frame)
        options_frame.pack(fill="x", padx=5, pady=5)
        
        # Mode selection
        self.use_concept_mode = tk.BooleanVar(value=True)
        concept_mode_cb = ctk.CTkCheckBox(options_frame, text="Refine with AI (concept mode)", 
                                          variable=self.use_concept_mode, 
                                          command=self.toggle_concept_mode)
        concept_mode_cb.pack(side="left", padx=10, pady=5)
        
        # Buttons
        button_frame = ctk.CTkFrame(self.prompt_input_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        self.generate_prompt_btn = ctk.CTkButton(button_frame, text="1. Generate Prompt", 
                                             command=self.generate_prompt)
        self.generate_prompt_btn.pack(side="left", padx=10, pady=5)
        
        self.generate_images_btn = ctk.CTkButton(button_frame, text="2. Generate Images", 
                                             command=self.generate_images, state="disabled")
        self.generate_images_btn.pack(side="left", padx=10, pady=5)
        
        # Enhanced prompt display (initially hidden)
        self.prompt_display_frame = ctk.CTkFrame(tab)
        self.prompt_label = ctk.CTkLabel(self.prompt_display_frame, text="Enhanced Prompt:", anchor="w")
        self.prompt_label.pack(fill="x", padx=5, pady=2)
        self.prompt_display = ctk.CTkTextbox(self.prompt_display_frame, height=60, state="disabled")
        self.prompt_display.pack(fill="x", padx=5, pady=5)
        
        # Existing Image Input Frame (initially hidden)
        self.image_input_frame = ctk.CTkFrame(concept_frame)
        
        # Existing image selection - simple browse button with explanation
        image_description = ctk.CTkLabel(
            self.image_input_frame, 
            text="Select an image to generate a 3D model from.\nThe image will be automatically processed (cropped to square and resized).",
            anchor="w",
            justify="left"
        )
        image_description.pack(fill="x", padx=5, pady=5)
        
        # Single browse button - large and prominent
        browse_btn = ctk.CTkButton(
            self.image_input_frame, 
            text="Browse for Image", 
            command=self.browse_for_image,
            height=40,
            font=("Helvetica", 14, "bold")
        )
        browse_btn.pack(fill="x", padx=20, pady=15)
        
        # Hidden entry for image path (for reference only)
        self.image_path_var = tk.StringVar()
        
        # Images display frame
        self.images_display_frame = ctk.CTkFrame(tab)
        self.images_display_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Label for images
        images_label = ctk.CTkLabel(self.images_display_frame, text="Generated/Selected Images:", anchor="w")
        images_label.pack(fill="x", padx=5, pady=5)
        
        # Container for image thumbnails
        self.images_container = ctk.CTkFrame(self.images_display_frame)
        self.images_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Button to proceed to 3D model tab (initially hidden)
        self.proceed_to_3d_btn = ctk.CTkButton(self.images_display_frame, text="Proceed to 3D Model Generation", 
                                            command=self.proceed_to_3d_model, state="disabled",
                                            height=40, font=("Helvetica", 14, "bold"))
        
        # Initialize image selection state
        self.image_var = tk.IntVar(value=-1)  # Store selected image index
        self.image_widgets = []  # List to store image widgets for cleanup

    def setup_3d_model_tab(self):
        """Setup the 3D model generation tab"""
        tab = self.tab_view.tab("2. 3D Model")
        
        # 3D Model Generation Frame
        model_frame = ctk.CTkFrame(tab)
        model_frame.pack(fill="x", padx=10, pady=10)
        
        # 3D Model Settings
        settings_label = ctk.CTkLabel(model_frame, text="3D Model Generation Settings:", anchor="w")
        settings_label.pack(fill="x", padx=5, pady=5)
        
        # Settings grid
        settings_grid = ctk.CTkFrame(model_frame)
        settings_grid.pack(fill="x", padx=5, pady=5)
        settings_grid.grid_columnconfigure(1, weight=1)
        
        # Model Type Selection
        model_type_label = ctk.CTkLabel(settings_grid, text="Model Type:")
        model_type_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.model_type_var = tk.StringVar(value="tripo")
        
        # Tripo option (Tripo API)
        tripo_radio = ctk.CTkRadioButton(settings_grid, text="Tripo API (best quality)", 
                                         variable=self.model_type_var, value="tripo")
        tripo_radio.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Trellis option (Replicate)
        trellis_radio = ctk.CTkRadioButton(settings_grid, text="Trellis (Replicate)", 
                                           variable=self.model_type_var, value="trellis")
        trellis_radio.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Hunyuan option (Replicate)
        hunyuan_radio = ctk.CTkRadioButton(settings_grid, text="Hunyuan (Replicate)", 
                                           variable=self.model_type_var, value="hunyuan")
        hunyuan_radio.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # All models option
        all_radio = ctk.CTkRadioButton(settings_grid, text="All Models (generate and compare)", 
                                      variable=self.model_type_var, value="all")
        all_radio.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Resolution setting
        resolution_label = ctk.CTkLabel(settings_grid, text="Max Faces:")
        resolution_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
        
        self.face_limit_var = tk.StringVar(value="200000")
        face_limit_entry = ctk.CTkEntry(settings_grid, textvariable=self.face_limit_var, width=120)
        face_limit_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        
        # Selected image display
        self.selected_img_frame = ctk.CTkFrame(tab)
        self.selected_img_frame.pack(fill="x", padx=10, pady=10)
        
        selected_img_label = ctk.CTkLabel(self.selected_img_frame, text="Selected Image:", anchor="w")
        selected_img_label.pack(fill="x", padx=5, pady=5)
        
        self.selected_img_display = ctk.CTkLabel(self.selected_img_frame, text="No image selected", height=200)
        self.selected_img_display.pack(fill="x", padx=5, pady=5)
        
        # Generate 3D model button
        self.generate_3d_btn = ctk.CTkButton(tab, text="Generate 3D Model", 
                                         command=self.generate_3d_model, state="disabled")
        self.generate_3d_btn.pack(padx=10, pady=10)
        
        # 3D model result display
        self.model_result_frame = ctk.CTkFrame(tab)
        self.model_result_frame.pack(fill="x", padx=10, pady=10)
        
        model_result_label = ctk.CTkLabel(self.model_result_frame, text="3D Model Result:", anchor="w")
        model_result_label.pack(fill="x", padx=5, pady=5)
        
        self.model_path_display = ctk.CTkLabel(self.model_result_frame, text="No model generated yet")
        self.model_path_display.pack(fill="x", padx=5, pady=5)
        
        # Preview and proceed buttons in a horizontal layout
        button_frame = ctk.CTkFrame(tab)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        # Preview button - open the 3D model with QuickLook
        self.preview_model_btn = ctk.CTkButton(button_frame, text="Preview 3D Model", 
                                          command=self.preview_3d_model, state="disabled")
        self.preview_model_btn.pack(side="left", padx=10, pady=5)
        
        # Button to proceed to output tab
        self.proceed_to_output_btn = ctk.CTkButton(button_frame, text="Proceed to Rendering & Output", 
                                               command=self.proceed_to_output, state="disabled")
        self.proceed_to_output_btn.pack(side="left", padx=10, pady=5)

    def setup_output_tab(self):
        """Setup the output processing tab"""
        tab = self.tab_view.tab("3. Output")
        
        # Output Frame
        output_frame = ctk.CTkFrame(tab)
        output_frame.pack(fill="x", padx=10, pady=10)
        
        # Title
        output_label = ctk.CTkLabel(output_frame, text="Blender Rendering & Output Processing:", anchor="w")
        output_label.pack(fill="x", padx=5, pady=5)
        
        # Blender Settings Frame
        blender_frame = ctk.CTkFrame(output_frame)
        blender_frame.pack(fill="x", padx=5, pady=5)
        
        # Title for Blender settings
        blender_settings_label = ctk.CTkLabel(blender_frame, text="Blender Rendering Settings:", anchor="w")
        blender_settings_label.pack(fill="x", padx=5, pady=5)
        
        # Settings grid for Blender
        blender_settings_grid = ctk.CTkFrame(blender_frame)
        blender_settings_grid.pack(fill="x", padx=5, pady=5)
        blender_settings_grid.grid_columnconfigure(1, weight=1)
        
        # Material selection
        material_label = ctk.CTkLabel(blender_settings_grid, text="Material:")
        material_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.material_var = tk.StringVar(value=self.material_name)
        material_options = ["blue_procedural_MAT", "red_procedural_MAT", "grey_procedural_MAT", "white_procedural_MAT"]
        material_dropdown = ctk.CTkOptionMenu(blender_settings_grid, values=material_options, variable=self.material_var)
        material_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Model height setting
        height_label = ctk.CTkLabel(blender_settings_grid, text="Model Height:")
        height_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.height_var = tk.StringVar(value=str(self.height_value))
        height_entry = ctk.CTkEntry(blender_settings_grid, textvariable=self.height_var, width=120)
        height_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Use textures checkbox
        self.use_textures_var = tk.BooleanVar(value=True)
        textures_checkbox = ctk.CTkCheckBox(blender_settings_grid, text="Use Textures", variable=self.use_textures_var)
        textures_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # Interactive mode checkbox
        self.interactive_mode_var = tk.BooleanVar(value=False)
        interactive_checkbox = ctk.CTkCheckBox(blender_settings_grid, text="Interactive Mode (open Blender UI)", variable=self.interactive_mode_var)
        interactive_checkbox.grid(row=3, column=1, padx=5, pady=5, sticky="w")
        
        # Display of 3D model info
        model_frame = ctk.CTkFrame(output_frame)
        model_frame.pack(fill="x", padx=5, pady=5)
        
        model_label = ctk.CTkLabel(model_frame, text="3D Model:")
        model_label.pack(side="left", padx=5, pady=5)
        
        self.output_model_display = ctk.CTkLabel(model_frame, text="No model selected")
        self.output_model_display.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        # Run Blender button
        self.run_blender_btn = ctk.CTkButton(output_frame, text="Run Blender Rendering", 
                                         command=self.run_blender, state="disabled")
        self.run_blender_btn.pack(fill="x", padx=10, pady=10)
        
        # Display of render folder
        folder_frame = ctk.CTkFrame(output_frame)
        folder_frame.pack(fill="x", padx=5, pady=5)
        
        folder_label = ctk.CTkLabel(folder_frame, text="Render Folder:")
        folder_label.pack(side="left", padx=5, pady=5)
        
        self.render_folder_display = ctk.CTkLabel(folder_frame, text="No render completed yet")
        self.render_folder_display.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        # Options for output processing
        options_frame = ctk.CTkFrame(output_frame)
        options_frame.pack(fill="x", padx=5, pady=5)
        
        # GIF options
        gif_label = ctk.CTkLabel(options_frame, text="GIF Options:", anchor="w")
        gif_label.pack(fill="x", padx=5, pady=5)
        
        gif_controls = ctk.CTkFrame(options_frame)
        gif_controls.pack(fill="x", padx=5, pady=5)
        
        duration_label = ctk.CTkLabel(gif_controls, text="Frame Duration (s):")
        duration_label.pack(side="left", padx=5, pady=5)
        
        self.duration_var = tk.StringVar(value="0.1")
        duration_entry = ctk.CTkEntry(gif_controls, textvariable=self.duration_var, width=60)
        duration_entry.pack(side="left", padx=5, pady=5)
        
        resize_label = ctk.CTkLabel(gif_controls, text="Resize (%):")
        resize_label.pack(side="left", padx=5, pady=5)
        
        self.resize_var = tk.StringVar(value="100")
        resize_entry = ctk.CTkEntry(gif_controls, textvariable=self.resize_var, width=60)
        resize_entry.pack(side="left", padx=5, pady=5)
        
        # Output buttons
        button_frame = ctk.CTkFrame(output_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        self.create_gif_btn = ctk.CTkButton(button_frame, text="Create GIF", 
                                        command=self.create_gif, state="disabled")
        self.create_gif_btn.pack(side="left", padx=10, pady=5)
        
        self.create_mp4_btn = ctk.CTkButton(button_frame, text="Create MP4 from Frames", 
                                         command=self.create_mp4, state="disabled")
        self.create_mp4_btn.pack(side="left", padx=10, pady=5)
        
        # Output result display
        self.output_result_frame = ctk.CTkFrame(tab)
        self.output_result_frame.pack(fill="x", padx=10, pady=10)
        
        output_result_label = ctk.CTkLabel(self.output_result_frame, text="Output Results:", anchor="w")
        output_result_label.pack(fill="x", padx=5, pady=5)
        
        self.output_path_display = ctk.CTkLabel(self.output_result_frame, text="No outputs generated yet")
        self.output_path_display.pack(fill="x", padx=5, pady=5)

    def setup_progress_area(self):
        """Setup progress display area (shared across tabs)"""
        # Progress Frame - fixed at bottom of window
        self.progress_frame = ctk.CTkFrame(self.root)
        self.progress_frame.pack(fill="x", padx=10, pady=10, side="bottom")
        
        # Progress message label
        self.progress_message = ctk.CTkLabel(self.progress_frame, text="Ready", anchor="w")
        self.progress_message.pack(fill="x", padx=5, pady=2)
        
        # Progress text
        self.progress_text = ctk.CTkTextbox(self.progress_frame, height=100, state="disabled", wrap="word")
        self.progress_text.pack(fill="x", padx=5, pady=5)
        
        # Progress bar (initially hidden)
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.configure(mode="indeterminate")
        # Will be shown when operations start

    def browse_file(self, entry_widget, title, filetypes):
        """Browse for a file and update the entry widget"""
        initial_dir = os.path.dirname(entry_widget.get()) if entry_widget.get() else str(project_root)
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initial_dir)
        if file_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, file_path)
            
            # If browsing for a 3D model, enable the next step
            if "3D Models" in str(filetypes):
                self.model_entry.configure(state="normal")
                self.model_entry.delete(0, tk.END)
                self.model_entry.insert(0, file_path)
                self.model_entry.configure(state="disabled")
                self.generated_3d_model_path = file_path
                self.run_blender_btn.configure(state="normal")

    def browse_directory(self, entry_widget, title):
        """Browse for a directory and update the entry widget"""
        initial_dir = entry_widget.get() if entry_widget.get() else str(project_root)
        dir_path = filedialog.askdirectory(title=title, initialdir=initial_dir)
        if dir_path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, dir_path)

    def log_progress(self, message):
        """Log a message to the progress text widget"""
        def _update_text():
            self.progress_text.configure(state="normal")
            self.progress_text.insert(tk.END, message + "\n")
            self.progress_text.configure(state="disabled")
            self.progress_text.see(tk.END)  # Auto-scroll
        # Schedule in main thread
        self.root.after(0, _update_text)

    def toggle_progress_bar(self, show=True, message=None):
        """Toggle the visibility of the progress bar"""
        if show:
            # Create progress bar frame if it doesn't exist
            if not hasattr(self, 'progress_frame'):
                self.progress_frame = ctk.CTkFrame(self.root)
                self.progress_frame.pack(side="bottom", fill="x", padx=10, pady=10)
                
                # Add a label for status messages
                self.progress_message = ctk.CTkLabel(self.progress_frame, text="Processing...")
                self.progress_message.pack(padx=5, pady=5)
                
                # Add the actual progress bar
                self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
                self.progress_bar.pack(fill="x", padx=5, pady=5)
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start()
            else:
                # If it exists but was hidden, show it again
                self.progress_frame.pack(side="bottom", fill="x", padx=10, pady=10)
                self.progress_bar.start()
            
            # Update message if provided
            if message and hasattr(self, 'progress_message'):
                self.progress_message.configure(text=message)
                
            # Also log the message
            if message:
                self.log_progress(message)
        else:
            # Hide progress bar if it exists
            if hasattr(self, 'progress_frame'):
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.stop()
                self.progress_frame.pack_forget()

    def update_ui_state(self, state, **kwargs):
        """Update UI elements based on the current state"""
        if state == "prompt_generated":
            self.generate_prompt_btn.configure(state="normal")
            self.generate_images_btn.configure(state="normal")
            # Show the prompt display
            self.prompt_display_frame.pack(fill="x", padx=10, pady=5)
            # Update prompt display
            if "prompt" in kwargs:
                self.prompt_display.configure(state="normal")
                self.prompt_display.delete("1.0", tk.END)
                self.prompt_display.insert("1.0", kwargs["prompt"])
                self.prompt_display.configure(state="disabled")
                
        elif state == "images_generated":
            # Enable selection UI for generated images
            if "images" in kwargs and kwargs["images"]:
                self.display_images(kwargs["images"])
                self.proceed_to_3d_btn.configure(state="normal")
                self.proceed_to_3d_btn.pack(padx=10, pady=10)
                
        elif state == "3d_model_generated":
            # Update 3D model display
            if "model_path" in kwargs:
                self.model_path_display.configure(text=f"Generated: {os.path.basename(kwargs['model_path'])}")
                self.generated_3d_model_path = kwargs["model_path"]
                
        elif state == "blender_complete":
            # Update render folder display and enable output processing
            if "render_folder" in kwargs:
                self.render_folder_display.configure(text=kwargs["render_folder"])
                # Enable output processing buttons
                self.create_gif_btn.configure(state="normal")
                self.create_mp4_btn.configure(state="normal")
                # Switch to output tab
                self.tab_view.set("3. Output")
                
        elif state == "output_complete":
            # Update output path display
            if "output_path" in kwargs:
                self.output_path_display.configure(text=f"Generated: {kwargs['output_path']}")

    def proceed_to_3d_model(self):
        """Switch to the 3D model tab and update selected image display"""
        if self.selected_image_index is None:
            messagebox.showerror("Input Error", "Please select an image first.")
            return
            
        # Explicitly switch to the 3D model tab
        self.tab_view.set("2. 3D Model")
        
        # Force update of the tab to ensure it's visible
        self.root.update_idletasks()
        
        # Enable 3D generation button
        self.generate_3d_btn.configure(state="normal")
        
        self.log_progress("Proceeding to 3D model generation tab")

    def proceed_to_output(self):
        """Proceed to the output tab with the generated 3D model"""
        if not hasattr(self, 'generated_3d_model_path') or not self.generated_3d_model_path:
            self.show_error("Error", "No 3D model has been generated yet")
            return
        
        # Set the selected model in the output tab
        self.output_model_display.configure(text=os.path.basename(self.generated_3d_model_path))
        
        # Enable the Blender button in the output tab
        self.run_blender_btn.configure(state="normal")
        
        # Switch to the output tab
        self.tab_view.set("3. Output")
        
        self.log_progress("Proceeding to output tab")

    def display_images(self, image_paths):
        """Display generated images as selectable options"""
        # Clear existing images first
        self.clear_generated_images()

        # Create frame for images
        images_frame = ctk.CTkFrame(self.images_container)
        images_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.image_widgets.append(images_frame) # Track the frame itself for clearing

        # Create a scrollable canvas for images
        # Use CTkScrollableFrame for better integration
        canvas_frame = ctk.CTkScrollableFrame(images_frame, orientation="horizontal", height=240) # Set fixed height
        canvas_frame.pack(fill="x", expand=False, padx=5, pady=5) # Fill horizontally, don't expand vertically
        self.image_widgets.append(canvas_frame) # Track the scrollable frame

        # Reset selected image state
        self.image_var.set(-1)
        self.selected_image_index = None

        # Add images to the canvas
        for i, img_path in enumerate(image_paths):
            try:
                # Create frame for each image + radio button
                img_frame = ctk.CTkFrame(canvas_frame)
                img_frame.pack(side="left", padx=10, pady=5)

                # Load and resize image
                img = Image.open(img_path)
                # Calculate resize to fit in frame (max 200px height)
                width, height = img.size
                new_height = 200
                new_width = int((new_height / height) * width)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Convert to PhotoImage
                ctk_image = ImageTk.PhotoImage(img) # Use ImageTk for CTk compatibility

                # Create label and attach image
                img_label = ctk.CTkLabel(img_frame, text="", image=ctk_image)
                img_label.image = ctk_image  # Keep reference
                img_label.pack(padx=5, pady=5)

                # Add selection radio button
                radio_btn = ctk.CTkRadioButton(img_frame, text=f"Image {i+1}",
                                             variable=self.image_var, value=i,
                                             command=lambda idx=i, path=img_path: self.select_image(idx, path))
                radio_btn.pack(padx=5, pady=5)

                # Track individual image widgets for cleanup
                self.image_widgets.extend([img_frame, img_label, radio_btn])

            except Exception as e:
                self.log_progress(f"Error displaying image {i}: {str(e)}")

        # Store image paths
        self.generated_images = image_paths

        # Ensure the proceed button is visible if images were displayed
        if image_paths:
            self.proceed_to_3d_btn.pack(padx=10, pady=10)
            self.proceed_to_3d_btn.configure(state="disabled") # Start disabled until selection
        else:
            self.proceed_to_3d_btn.pack_forget() # Hide if no images

    def select_image(self, index, path):
        """Handle image selection"""
        self.selected_image_index = index
        self.log_progress(f"Selected image {index+1}")
        
        # Update selected image in 3D tab
        try:
            # Load and resize image for display
            img = Image.open(path)
            # Calculate resize to fit in frame (max 200px height)
            width, height = img.size
            new_height = 200
            new_width = int((new_height / height) * width)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update the image display
            self.selected_img_display.configure(image=photo, text="")
            self.selected_img_display.image = photo  # Keep reference
            
        except Exception as e:
            self.log_progress(f"Error displaying selected image: {str(e)}")
            self.selected_img_display.configure(text=f"Error displaying image: {str(e)}", image=None)
        
        # Enable proceed button
        self.proceed_to_3d_btn.configure(state="normal")

    def toggle_concept_mode(self):
        """Toggle between concept mode and direct prompt mode"""
        if self.use_concept_mode.get():
            self.concept_entry.configure(height=80)
            self.generate_prompt_btn.configure(state="normal")
            self.generate_images_btn.configure(state="disabled")
        else:
            self.concept_entry.configure(height=120)
            self.generate_prompt_btn.configure(state="disabled") 
            self.generate_images_btn.configure(state="normal")
            # Show prompt display frame when in direct mode
            self.prompt_display_frame.pack(fill="x", padx=10, pady=5)

    def generate_prompt(self):
        """Generate an enhanced prompt using OpenAI based on user concept"""
        concept = self.concept_entry.get("1.0", tk.END).strip()
        if not concept or concept == "Enter a concept or detailed description of what you want to create...":
            messagebox.showerror("Input Error", "Please enter a concept or description first.")
            return
            
        # Disable button and show progress
        self.generate_prompt_btn.configure(state="disabled")
        self.toggle_progress_bar(True)
        self.log_progress(f"Generating enhanced prompt for: {concept}")
        
        # Start thread for API call
        thread = threading.Thread(target=self._generate_prompt_thread, args=(concept,), daemon=True)
        thread.start()
    
    def _generate_prompt_thread(self, concept):
        """Thread function for prompt generation"""
        try:
            # Initialize OpenAI client if needed
            if self.openai_client is None:
                self.openai_client = OpenAIResponsesAPI()
                
            # Prepare system prompt based on guidelines
            with open(project_root / "data/input/system_config/promptGen_guideline.md", "r") as f:
                guidelines = f.read()
                
            # Extract the "promptGen Agent Template" section
            if "promptGen Agent Template" in guidelines:
                template_section = guidelines.split("## promptGen Agent Template")[1]
                # Extract system prompt
                system_prompt = template_section.split("```")[1].strip()
                # Extract user prompt template
                user_prompt_template = template_section.split("```")[3].strip()
                
                # Replace {input} with actual concept
                user_prompt = user_prompt_template.replace("{input}", concept)
                
                self.log_progress("Using promptGen guidelines to enhance prompt...")
                
                # Call OpenAI with system and user prompts
                response = self.openai_client.response(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.7
                )
                
                # Clean response if needed (OpenAI sometimes adds quotes/formatting)
                enhanced_prompt = response.strip()
                if enhanced_prompt.startswith('"') and enhanced_prompt.endswith('"'):
                    enhanced_prompt = enhanced_prompt[1:-1]
                
                # Store the enhanced prompt
                self.enhanced_prompt = enhanced_prompt
                
                # Update UI with the enhanced prompt
                self.root.after(0, lambda: self.update_ui_state("prompt_generated", prompt=enhanced_prompt))
                self.log_progress("✓ Enhanced prompt generated successfully!")
                
                # Re-enable buttons
                self.root.after(0, lambda: self.generate_prompt_btn.configure(state="normal"))
                self.root.after(0, lambda: self.generate_images_btn.configure(state="normal"))
            else:
                # Fallback if template not found
                self.log_progress("! promptGen guideline template not found, using default prompt enhancement")
                # Simple prompt enhancement
                enhanced_prompt = f"Professional {concept}, detailed high-quality image with studio lighting, 8K resolution, photorealistic"
                self.enhanced_prompt = enhanced_prompt
                self.root.after(0, lambda: self.update_ui_state("prompt_generated", prompt=enhanced_prompt))
                
        except Exception as e:
            self.log_progress(f"Error generating prompt: {str(e)}")
            traceback.print_exc()
            # Re-enable button
            self.root.after(0, lambda: self.generate_prompt_btn.configure(state="normal"))
        finally:
            # Stop progress bar
            self.root.after(0, lambda: self.toggle_progress_bar(False))
            
    def generate_images(self):
        """Generate images using Replicate API based on the prompt"""
        # Get prompt based on mode
        if self.use_concept_mode.get():
            # Use the enhanced prompt
            if not hasattr(self, 'enhanced_prompt') or not self.enhanced_prompt:
                messagebox.showerror("Input Error", "Please generate an enhanced prompt first.")
                return
            prompt = self.enhanced_prompt
        else:
            # Use the raw input as prompt
            prompt = self.concept_entry.get("1.0", tk.END).strip()
            if not prompt or prompt == "Enter a concept or detailed description of what you want to create...":
                messagebox.showerror("Input Error", "Please enter a prompt first.")
                return
        
        # Disable button and show progress
        self.generate_images_btn.configure(state="disabled")
        self.toggle_progress_bar(True)
        self.log_progress(f"Generating images with prompt: {prompt}")

        # Clear previous images before starting generation
        self.clear_generated_images()

        # Create output directory
        img_output_dir = self.output_dir / "images"
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start thread for API call
        thread = threading.Thread(target=self._generate_images_thread, args=(prompt, img_output_dir), daemon=True)
        thread.start()
    
    def _generate_images_thread(self, prompt, output_dir):
        """Thread function for image generation"""
        try:
            # Initialize Replicate client if needed
            if self.replicate_client is None:
                self.replicate_client = ReplicateAPI()
            
            # Generate multiple images
            image_paths = []
            num_images = 3  # Generate 3 images to choose from
            
            for i in range(num_images):
                self.log_progress(f"Generating image {i+1}/{num_images}...")
                
                # Generate image with Replicate
                image_url = self.replicate_client.generate_image(
                    prompt=prompt,
                    model="flux-dev",  # Use flux-dev model for good quality
                    aspect_ratio="1:1",  # Square aspect ratio works best for 3D models
                    safety_tolerance=6
                )
                
                if image_url:
                    # Download the image
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = self.replicate_client.download_file(
                        image_url,
                        output_dir=str(output_dir),
                        filename=f"concept_{i+1}_{timestamp}.jpg"
                    )
                    
                    if image_path:
                        self.log_progress(f"✓ Image {i+1} downloaded to {image_path}")
                        image_paths.append(image_path)
                    else:
                        self.log_progress(f"! Failed to download image {i+1}")
                else:
                    self.log_progress(f"! Failed to generate image {i+1}")
            
            # Update UI with generated images (using display_images handles UI state now)
            if image_paths:
                # Schedule UI update in main thread
                self.root.after(0, lambda paths=image_paths: self.display_images(paths))
                self.log_progress(f"✓ Generated {len(image_paths)} images successfully! Please select one.")
            else:
                self.log_progress("! No images were generated successfully.")
                # Ensure proceed button is hidden if generation failed
                self.root.after(0, lambda: self.proceed_to_3d_btn.pack_forget())

            # Re-enable button
            self.root.after(0, lambda: self.generate_images_btn.configure(state="normal"))

        except Exception as e:
            self.log_progress(f"Error generating images: {str(e)}")
            traceback.print_exc()
            # Re-enable button
            self.root.after(0, lambda: self.generate_images_btn.configure(state="normal"))
        finally:
            # Hide progress bar
            self.root.after(0, lambda: self.toggle_progress_bar(False))

    def generate_3d_model(self):
        """Generate 3D model without running Blender"""
        if self.selected_image_index is None or not self.generated_images:
            self.show_error("Error", "Please generate and select an image first")
            return
            
        # Get settings
        model_type = self.model_type_var.get()
        face_limit = int(self.face_limit_var.get())
        
        # Get selected image path
        selected_image = self.generated_images[self.selected_image_index]
        
        # Check if image is a full path or just a filename
        if os.path.isabs(selected_image):
            selected_img_path = selected_image
        else:
            # If we have image_output_dir, use it, otherwise assume image is in output_dir/images
            if hasattr(self, 'image_output_dir'):
                selected_img_path = os.path.join(self.image_output_dir, selected_image)
            else:
                img_dir = os.path.join(self.output_dir, "images")
                selected_img_path = os.path.join(img_dir, selected_image)
        
        # Check if the file exists
        if not os.path.exists(selected_img_path):
            self.log_progress(f"Warning: Image not found at expected path: {selected_img_path}")
            # Try to find by filename in case it's stored somewhere else
            selected_img_path = selected_image
            
        # Create threading event for signaling completion
        self.generation_completed = threading.Event()
        
        # Start the generation process in a separate thread
        threading.Thread(
            target=self._generate_3d_thread,
            args=(selected_img_path, model_type, face_limit),
            daemon=True
        ).start()
        
        # Show progress bar
        self.toggle_progress_bar(True, "Generating 3D model...")
        
        # Schedule a check for completion
        self.root.after(100, self._check_model_generation_completed)
    
    def _generate_3d_thread(self, image_path, model_type, face_limit):
        """Thread function for 3D model generation"""
        try:
            # Create a new project directory for this run
            self.project_dir = self._create_project_dir()
            self.exports_dir = os.path.join(self.project_dir, "exports")
            os.makedirs(self.exports_dir, exist_ok=True)
            
            # Copy concept image to refs folder
            refs_dir = os.path.join(self.project_dir, "refs")
            os.makedirs(refs_dir, exist_ok=True)
            concept_file = os.path.basename(image_path)
            concept_path = os.path.join(refs_dir, concept_file)
            shutil.copy2(image_path, concept_path)
            print(f"Copied concept image to: {concept_path}")
            
            # Generate 3D model
            self.log_progress(f"Generating 3D model from image: {concept_file}")
            self.log_progress(f"Settings: Model Type={model_type}, Face Limit={face_limit}")
            
            model_path = None
            
            # Generate model based on selected type
            if model_type == "tripo":
                model_path = self._generate_with_tripo(image_path, self.exports_dir, face_limit)
            elif model_type in ["trellis", "hunyuan"]:
                model_path = self._generate_with_replicate(image_path, self.exports_dir, model_type, face_limit)
            elif model_type == "all":
                # Try all models - stop at first successful one
                for try_model in ["tripo", "trellis", "hunyuan"]:
                    self.log_progress(f"Trying {try_model} model...")
                    
                    try:
                        if try_model == "tripo":
                            model_path = self._generate_with_tripo(image_path, self.exports_dir, face_limit)
                        else:
                            model_path = self._generate_with_replicate(image_path, self.exports_dir, try_model, face_limit)
                            
                        # If model generation succeeded, stop trying other models
                        if model_path:
                            self.log_progress(f"✓ Successfully generated model with {try_model}")
                            break
                    except Exception as e:
                        self.log_progress(f"Failed to generate model with {try_model}: {e}")
            
            if model_path:
                self.generated_3d_model_path = model_path
                self.log_progress("✓ 3D model generated successfully!")
            else:
                self.log_progress("! Failed to generate 3D model")
                
        except Exception as e:
            self.log_progress(f"Error generating 3D model: {str(e)}")
            traceback.print_exc()
        finally:
            # Signal that the generation process has completed
            self.generation_completed.set()
    
    def _check_model_generation_completed(self):
        """Check if the 3D model generation has completed"""
        if self.generation_completed.is_set():
            self.toggle_progress_bar(False)
            
            # Update the UI to display the result
            if hasattr(self, 'generated_3d_model_path') and self.generated_3d_model_path:
                self.model_path_display.configure(
                    text=f"Model generated: {os.path.basename(self.generated_3d_model_path)}"
                )
                
                # Enable preview and output buttons
                self.preview_model_btn.configure(state="normal")
                self.proceed_to_output_btn.configure(state="normal")
            else:
                self.model_path_display.configure(text="Failed to generate 3D model")
        else:
            # Check again after a delay
            self.root.after(100, self._check_model_generation_completed)
    
    def preview_3d_model(self):
        """Open the 3D model with QuickLook for preview"""
        if not hasattr(self, 'generated_3d_model_path') or not self.generated_3d_model_path:
            self.show_error("Error", "No 3D model has been generated yet")
            return
        
        if not os.path.exists(self.generated_3d_model_path):
            self.show_error("Error", "Generated model file not found")
            return
        
        try:
            # On macOS, use QuickLook to preview the 3D model
            if sys.platform == 'darwin':
                subprocess.run(['qlmanage', '-p', self.generated_3d_model_path], 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.log_progress(f"Opening model with QuickLook: {os.path.basename(self.generated_3d_model_path)}")
            else:
                # For other platforms, show a message that preview is not available
                self.show_info("Preview Not Available", 
                            "3D model preview is currently only supported on macOS.\n"
                            f"The model is located at: {self.generated_3d_model_path}")
        except Exception as e:
            self.show_error("Preview Error", f"Could not preview the model: {str(e)}")
    
    def run_blender(self):
        """Run Blender to render the 3D model"""
        if not hasattr(self, 'generated_3d_model_path') or not self.generated_3d_model_path:
            self.show_error("Error", "No 3D model has been generated yet")
            return
        
        # Get Blender settings
        material_name = self.material_var.get()
        height_value = float(self.height_var.get())
        use_textures = self.use_textures_var.get()
        interactive_mode = self.interactive_mode_var.get()
        
        # Create threading event for signaling completion
        self.blender_completed = threading.Event()
        
        # Start the Blender process in a separate thread
        threading.Thread(
            target=self._run_blender_thread,
            args=(self.generated_3d_model_path, self.project_dir, material_name, height_value, use_textures, interactive_mode),
            daemon=True
        ).start()
        
        # Show progress bar
        self.toggle_progress_bar(True, "Running Blender...")
        
        # Schedule a check for completion
        self.root.after(100, self._check_blender_completed)
    
    def _run_blender_thread(self, model_path, project_dir, material_name, height_value, use_textures, interactive_mode):
        """Thread function for running Blender"""
        try:
            # Run Blender with the provided arguments
            self.log_progress(f"Running Blender with model: {os.path.basename(model_path)}")
            self.log_progress(f"Settings: Material={material_name}, Height={height_value}, Textures={use_textures}, Interactive={interactive_mode}")
            
            # Create Blender runner
            blender_runner = BlenderRunner(
                blend_file=self.template_path,
                model_path=model_path,
                project_dir=project_dir,
                material=material_name,
                model_height=height_value,
                use_textures=use_textures,
                interactive=interactive_mode
            )
            
            # Run Blender
            success, render_dir = blender_runner.run()
            
            if success and render_dir:
                self.render_dir = render_dir
                self.log_progress(f"✓ Blender rendering completed successfully!")
                self.log_progress(f"Render output directory: {render_dir}")
            else:
                self.log_progress("! Blender rendering failed")
                
        except Exception as e:
            self.log_progress(f"Error running Blender: {str(e)}")
            traceback.print_exc()
        finally:
            # Signal that the Blender process has completed
            self.blender_completed.set()
    
    def _check_blender_completed(self):
        """Check if the Blender process has completed"""
        if hasattr(self, 'blender_completed') and self.blender_completed.is_set():
            self.toggle_progress_bar(False)
            
            # Update the UI to display the result
            if hasattr(self, 'render_dir') and self.render_dir:
                self.render_folder_display.configure(text=self.render_dir)
                
                # Enable the output processing buttons
                self.create_gif_btn.configure(state="normal")
                self.create_mp4_btn.configure(state="normal")
            else:
                self.render_folder_display.configure(text="Blender rendering failed")
        else:
            # Check again after a delay
            self.root.after(100, self._check_blender_completed)
    
    def create_gif(self):
        """Create a GIF from the rendered frames using image_utils"""
        if not hasattr(self, 'render_dir') or not self.render_dir:
            self.show_error("Error", "No render folder available")
            return
        
        try:
            # Find rendered frames
            render_path = os.path.join(self.render_dir, "renders")
            if not os.path.exists(render_path):
                self.show_error("Error", f"Render folder not found: {render_path}")
                return
            
            # Find all image files in the renders directory
            image_files = []
            for file in sorted(os.listdir(render_path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(render_path, file))
            
            if not image_files:
                self.show_error("Error", "No image frames found in the render folder")
                return
            
            # Get settings
            duration = float(self.duration_var.get())
            resize_factor = float(self.resize_var.get())
            
            # Create GIF output filename
            gif_output = os.path.join(self.project_dir, f"animation_{time.strftime('%Y%m%d-%H%M%S')}.gif")
            
            # Show progress
            self.toggle_progress_bar(True, "Creating GIF from frames...")
            
            # Create GIF in a separate thread
            threading.Thread(
                target=self._create_gif_thread,
                args=(image_files, gif_output, duration, resize_factor),
                daemon=True
            ).start()
            
        except Exception as e:
            self.toggle_progress_bar(False)
            self.show_error("GIF Creation Error", f"Error creating GIF: {str(e)}")
    
    def _create_gif_thread(self, image_files, output_path, duration, resize_factor):
        """Thread function for creating a GIF"""
        try:
            # Create the GIF (using our imported or fallback function)
            success = images_to_gif(
                input_paths=image_files,
                output_path=output_path,
                duration=duration,
                resize_factor=resize_factor,
                optimize=True
            )
            
            # Update UI
            if success:
                self.output_path_display.configure(text=f"GIF created: {output_path}")
                self.log_progress(f"✓ GIF created successfully: {output_path}")
                
                # Show the GIF
                try:
                    if sys.platform == 'darwin':
                        subprocess.run(['open', output_path], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    self.log_progress(f"Error opening GIF: {str(e)}")
            else:
                self.log_progress("! Failed to create GIF")
        except Exception as e:
            self.log_progress(f"Error creating GIF: {str(e)}")
        finally:
            # Hide progress bar
            self.root.after(0, lambda: self.toggle_progress_bar(False))
    
    def create_mp4(self):
        """Create an MP4 video from the rendered frames using video_utils"""
        if not hasattr(self, 'render_dir') or not self.render_dir:
            self.show_error("Error", "No render folder available")
            return
        
        try:
            # Find rendered frames
            render_path = os.path.join(self.render_dir, "renders")
            if not os.path.exists(render_path):
                self.show_error("Error", f"Render folder not found: {render_path}")
                return
            
            # Find all image files in the renders directory
            image_files = []
            for file in sorted(os.listdir(render_path)):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(render_path, file))
            
            if not image_files:
                self.show_error("Error", "No image frames found in the render folder")
                return
            
            # Create MP4 output filename
            mp4_output = os.path.join(self.project_dir, f"animation_{time.strftime('%Y%m%d-%H%M%S')}.mp4")
            
            # Show progress
            self.toggle_progress_bar(True, "Creating MP4 from frames...")
            
            # Create MP4 in a separate thread
            threading.Thread(
                target=self._create_mp4_thread,
                args=(image_files, mp4_output),
                daemon=True
            ).start()
            
        except Exception as e:
            self.toggle_progress_bar(False)
            self.show_error("MP4 Creation Error", f"Error creating MP4: {str(e)}")
    
    def _create_mp4_thread(self, image_files, output_path):
        """Thread function for creating an MP4 video"""
        try:
            # Create the MP4 (using our imported or fallback function)
            fps = 24  # Default frame rate
            success = image_to_video(
                frame_paths=image_files,
                output_path=output_path,
                fps=fps
            )
            
            # Update UI
            if success:
                self.output_path_display.configure(text=f"MP4 created: {output_path}")
                self.log_progress(f"✓ MP4 created successfully: {output_path}")
                
                # Show the MP4
                try:
                    if sys.platform == 'darwin':
                        subprocess.run(['open', output_path], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    self.log_progress(f"Error opening MP4: {str(e)}")
            else:
                self.log_progress("! Failed to create MP4")
        except Exception as e:
            self.log_progress(f"Error creating MP4: {str(e)}")
        finally:
            # Hide progress bar
            self.root.after(0, lambda: self.toggle_progress_bar(False))

    def _create_project_dir(self):
        """Create a new project directory for the current run"""
        # Create timestamp for unique folder name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        project_name = f"blender_project_{timestamp}"
        
        # Create project directory in the output folder
        project_dir = os.path.join(self.output_dir, project_name)
        os.makedirs(project_dir, exist_ok=True)
        
        # Log the created directory
        self.log_progress(f"Created project directory: {project_dir}")
        
        return project_dir
        
    async def _generate_with_tripo_async(self, image_path, output_dir, face_limit):
        """Async function for 3D model generation with Tripo API"""
        try:
            # Initialize Tripo API
            tripo_api = TripoAPI()
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create output path
            output_path = os.path.join(output_dir, f"model_tripo_{timestamp}.glb")
            
            # Local file path is fine for Tripo API's generate_threed method
            self.log_progress("Starting 3D model generation with Tripo API...")
            model_path = await tripo_api.generate_threed(
                image_path=image_path,  # Tripo accepts a local file path
                output_path=output_path,
                texture_quality="detailed",  # Always use detailed quality
                face_limit=face_limit,
                texture=True,
                pbr=True
            )
            
            if model_path:
                self.log_progress("3D model generated successfully with Tripo API")
            else:
                self.log_progress("Failed to generate 3D model with Tripo API")
                
            return model_path
        except Exception as e:
            self.log_progress(f"Error in Tripo API: {str(e)}")
            traceback.print_exc()
            raise
    
    def _generate_with_tripo(self, image_path, output_dir, face_limit):
        """Generate 3D model using Tripo API"""
        self.log_progress("Generating 3D model with Tripo API...")
        
        # Set up async event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            model_path = loop.run_until_complete(
                self._generate_with_tripo_async(image_path, output_dir, face_limit)
            )
            return model_path
        except Exception as e:
            self.log_progress(f"Error with Tripo generation: {e}")
            return None
        finally:
            loop.close()
    
    def _generate_with_replicate(self, image_path, output_dir, model_name, face_limit):
        """Generate 3D model using Replicate API with specified model"""
        self.log_progress(f"Generating 3D model with Replicate API ({model_name})...")
        
        try:
            # Initialize Replicate client if needed
            if self.replicate_client is None:
                self.replicate_client = ReplicateAPI()
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Create output path
            output_path = os.path.join(output_dir, f"model_{model_name}_{timestamp}.glb")
            
            # Select model based on name
            if model_name == "trellis":
                replicate_model = "cjwbw/trelliscompute:828c752da1d2f73ec75f85e86cb85e7cd58922337b217a4fb8934bf7a0b43a2e"
            elif model_name == "hunyuan":
                replicate_model = "tencent-hunyuan/image-to-3d-object:6d3caf573e8a3ab0fc523e064360457985ed59b425c93c23e0fb129a75db67ba"
            else:
                self.log_progress(f"Unknown Replicate model: {model_name}")
                return None
            
            # We need to upload the local file to get a URL first
            try:
                import replicate
                self.log_progress(f"Uploading local image to Replicate: {image_path}")
                with open(image_path, "rb") as f:
                    image_url = replicate.upload(f)
                    self.log_progress(f"Image successfully uploaded with URL: {image_url}")
            except Exception as upload_error:
                self.log_progress(f"Error uploading image to Replicate: {upload_error}")
                return None
            
            # Generate model with Replicate
            self.log_progress(f"Starting 3D model generation with {model_name}...")
            model_url = self.replicate_client.generate_threed(
                image_url=image_url,  # Use the uploaded image URL
                model=model_name,
                seed=1234,
                steps=50,
                guidance_scale=5.5,
                remove_background=True,
                texture_size=1024
            )
            
            if model_url:
                self.log_progress(f"3D model generated successfully with {model_name}")
                # Download the model
                model_path = self.replicate_client.download_file(
                    url=model_url,
                    output_dir=str(output_dir),
                    filename=os.path.basename(output_path)
                )
                return model_path
            else:
                self.log_progress(f"Failed to generate 3D model with {model_name}")
                return None
                
        except Exception as e:
            self.log_progress(f"Error with {model_name} generation: {e}")
            traceback.print_exc()
            return None
            
    def show_error(self, title, message):
        """Show an error message box"""
        messagebox.showerror(title, message)
        
    def show_info(self, title, message):
        """Show an information message box"""
        messagebox.showinfo(title, message)

    def switch_input_mode(self, mode=None):
        """Switch between prompt and existing image input modes"""
        mode = self.input_mode_var.get()
        
        if "Prompt" in mode:
            # Show prompt input, hide image input
            self.prompt_input_frame.pack(fill="x", padx=5, pady=5)
            self.image_input_frame.pack_forget()
        else:
            # Show image input, hide prompt input
            self.prompt_input_frame.pack_forget()
            self.image_input_frame.pack(fill="x", padx=5, pady=5)
            
            # Reset the preview area
            self.clear_generated_images()
    
    def browse_for_image(self):
        """Browse for an image file and automatically process it for 3D generation"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image", 
            filetypes=filetypes
        )
        
        if file_path:
            # Show progress
            self.toggle_progress_bar(True, "Processing image...")
            
            # Process image in a thread to avoid UI freeze
            threading.Thread(
                target=self._process_and_use_image,
                args=(file_path,),
                daemon=True
            ).start()
    
    def _process_and_use_image(self, image_path):
        """Process the image and prepare it for 3D generation"""
        try:
            # Update UI with selected image path
            self.image_path_var.set(image_path)
            
            # Create output directory if it doesn't exist
            if not hasattr(self, 'output_dir'):
                self.output_dir = Path(project_root) / "data" / "output" / "blender_pipeline"
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
            # Create images directory if needed
            img_output_dir = self.output_dir / "images"
            img_output_dir.mkdir(parents=True, exist_ok=True)
            self.image_output_dir = img_output_dir
            
            # Process the image (crop & resize to square)
            processed_path = self.process_image_for_3d(image_path, img_output_dir)
            
            if processed_path:
                # Clear any previously displayed images
                self.clear_generated_images()
                
                # Store as the selected image (even though only one, keep structure consistent)
                self.generated_images = [processed_path]
                # self.selected_image_index = 0 # Don't auto-select, let user click

                # Display the image and allow selection
                self.display_images([processed_path])
                # self.image_var.set(0)  # Don't auto-select
                # self.select_image(0, processed_path) # User needs to select

                # Remove preview call
                # self.preview_selected_image(processed_path)
                self.log_progress(f"✓ Image processed and ready for selection")

                # Update UI with guidance
                self.root.after(0, lambda: self.log_progress("Please select the processed image to continue"))
            else:
                self.log_progress("! Failed to process the image")
        
        except Exception as e:
            self.log_progress(f"Error processing image: {str(e)}")
            traceback.print_exc()
        finally:
            # Hide progress bar
            self.root.after(0, lambda: self.toggle_progress_bar(False))
    
    def process_image_for_3d(self, image_path, output_dir):
        """Process an image for 3D generation (crop, resize to square)"""
        try:
            # Load the image
            img = Image.open(image_path)
            
            # Create a square crop (taking the center)
            width, height = img.size
            
            if width != height:
                # Get the smallest dimension
                min_dim = min(width, height)
                
                # Calculate crop box (center crop)
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                
                # Crop the image
                img = img.crop((left, top, right, bottom))
                
            # Resize to 1000x1000
            img = img.resize((1000, 1000), Image.LANCZOS)
            
            # Save the processed image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_processed_{timestamp}.jpg")
            
            # Save with high quality
            img.save(output_path, "JPEG", quality=95)
            
            return output_path
            
        except Exception as e:
            self.log_progress(f"Error processing image: {str(e)}")
            traceback.print_exc()
            return None
    
    def clear_generated_images(self):
        """Clear any previously generated images"""
        try:
            print("Clearing generated images...") # Debug print
            # Clear any existing image widgets
            # Iterate backwards to avoid index issues when removing
            for i in range(len(self.image_widgets) -1, -1, -1):
                widget = self.image_widgets.pop(i)
                try:
                    widget.destroy()
                except Exception as e:
                    # Ignore errors when destroying widgets (Tcl errors can occur)
                    print(f"Minor error destroying widget: {e}")
                    pass

            self.image_widgets = [] # Ensure list is empty

            # Reset selection state
            self.image_var.set(-1)
            self.selected_image_index = None
            self.generated_images = []

            # Hide proceed button
            try:
                self.proceed_to_3d_btn.pack_forget()
            except Exception:
                # Ignore if button is not packed
                pass

        except Exception as e:
            print(f"Error clearing generated images: {e}")
            # Don't re-raise, just continue


if __name__ == "__main__":
    try:
        # Check if template file exists, if not print warning
        user_template = Path(project_root) / "data/input/system_config/blender_templates/blenderGen_presentation_03.blend"
        if not user_template.exists():
            print(f"WARNING: Template file not found at the requested location: {user_template}")
            print("Checking alternative locations or using fallback paths...")
        else:
            print(f"Using Blender template from: {user_template}")
        
        root = ctk.CTk()
        app = ComprehensiveGenUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start UI: {e}")
        # Add a fallback simple message box if CTk fails
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw() # Hide the main window
            messagebox.showerror("UI Error", f"Failed to initialize CustomTkinter UI: {e}\n\nPlease check your installation.")
        except Exception as fallback_e:
            print(f"Fallback Tkinter message box also failed: {fallback_e}") 