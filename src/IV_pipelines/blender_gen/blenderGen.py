import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime

# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[3]  # Changed from 2 to 3 to get correct project root
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

class BlenderGen:
    def __init__(self, version="4.1.1"):
        self.version = version
        self.blender_executable = self.get_blender_executable_path()
        self.project_folder = None

    def get_blender_executable_path(self):
        if os.name == 'nt':  # Windows
            path = f"C:/Program Files/Blender Foundation/Blender {self.version}/blender.exe"
            print(f"Checking Windows path: {path}")
        elif os.name == 'posix':  # macOS/Linux
            if os.uname().sysname == 'Darwin':  # macOS
                path = f"/Applications/Blender.app/Contents/MacOS/Blender"
                print(f"Checking macOS path: {path}")
            else:  # Linux
                path = "/usr/bin/blender"
                print(f"Checking Linux path: {path}")
        else:
            raise EnvironmentError("Unsupported operating system")

        if not os.path.exists(path):
            print(f"Blender executable not found at {path}")
            raise FileNotFoundError(f"Blender executable not found at {path}")
        print(f"Blender executable found at {path}")
        return path
        
    def create_project_folder(self, base_output, asset_path):
        """Create a timestamped project folder with standard subfolders"""
        # Create unique project folder with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name = Path(asset_path).stem
        project_folder = Path(base_output) / f"{model_name}_{timestamp}"
        
        # Create standard folder structure
        folders = [
            "scenes",
            "exports",
            "refs",
            "renders",
            "textures"
        ]
        
        for folder in folders:
            (project_folder / folder).mkdir(parents=True, exist_ok=True)
            print(f"Created folder: {project_folder / folder}")
            
        return project_folder

    def launch_blender_with_script(self, script_path, env_vars, interactive=False):
        """Launch Blender with a Python script
        
        Args:
            script_path: Path to the Python script to run
            env_vars: Environment variables to pass to the script
            interactive: If True, launch Blender UI. If False, run in background.
        """
        script_path = os.path.abspath(script_path)
        
        # Add memory optimization environment variables
        memory_env = {
            'PYTHONDONTWRITEBYTECODE': '1',
            'BLENDER_FILE_CACHE_SIZE': '4096'
        }
        
        # Update environment with memory settings
        merged_env = {**os.environ, **env_vars, **memory_env}
        
        # Command with memory optimization flags
        command = [self.blender_executable]
        
        # Add background flag only if not interactive
        if not interactive:
            command.append('--background')
            command.append('--debug-python')
            command.append('--factory-startup')  # Start with factory settings to reduce memory usage
        
        # Add the Python script
        command.extend(['--python', script_path, '--'])
        
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, env=merged_env)

    def blenderGen_pipeline(self, template_path, asset_path, mtl_name, texture_bool, output_path, height, interactive):
        """Process a 3D model in Blender
        
        Args:
            template_path: Path to Blender template file
            asset_path: Path to 3D model file
            mtl_name: Material name to use
            texture_bool: Whether to use textures
            output_path: Base directory where the project folder will be created
            height: Height to scale the model to
            interactive: If True, open Blender UI. If False, render in background.
        """
        print(">>>BLENDER SETTINGS:")
        print(f"Template: {template_path}")
        print(f"Asset: {asset_path}")
        print(f"Material: {mtl_name}")
        print(f"Base Output: {output_path}")
        print(f"Height: {height}")
        print(f"Use Textures: {texture_bool}")
        print(f"Interactive Mode: {interactive}")
        
        # Verify asset file exists
        if not Path(asset_path).exists():
            raise FileNotFoundError(f"3D model file not found: {asset_path}")
        
        # Create project folder structure
        self.project_folder = self.create_project_folder(output_path, asset_path)
        # Define specific paths for subfolders
        scenes_path = self.project_folder / "scenes"
        renders_path = self.project_folder / "renders"
        exports_path = self.project_folder / "exports"
        
        print(f"Project folder created at: {self.project_folder}")
        print(f"Scenes will be saved to: {scenes_path}")
        print(f"Renders will be saved to: {renders_path}")
        print(f"Exports will be saved to: {exports_path}")

        # Set up environment variables with specific paths
        env_vars = {
            'TEMPLATE_PATH': template_path,
            'ASSET_PATH': asset_path,
            'MTL_NAME': mtl_name,
            'SCENES_SAVE_PATH': str(scenes_path),    # Path for .blend files
            'RENDERS_SAVE_PATH': str(renders_path),   # Path for render output
            'EXPORTS_SAVE_PATH': str(exports_path),   # Path for GLB exports
            'HEIGHT': str(height),
            'TEXTURE_BOOL': str(texture_bool),
            'SAVE_BLEND': 'True',
            'INTERACTIVE_MODE': str(interactive)
        }

        # Choose which script to run
        script_path = str(project_root / "src/IV_pipelines/blender_gen/bpy_BlenderGen.py")
        
        # Launch Blender
        self.launch_blender_with_script(script_path, env_vars, interactive)
        
        return self.project_folder
            
if __name__ == "__main__":
    # Define your paths
    template_path = str("/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_presentation_03.blend")
    asset_path = str(project_root / "data/output/test_3d_models/test_3d_recraft_20250402-135607.glb")
    mtl_name = "blue_procedural_MAT"
    # This is the BASE output directory where the timestamped project folder will be created
    output_path = str(project_root / "data/output/blender") 
    height = 0.7
    texture_bool = True
    # Set to True to open Blender UI, False to render in background
    interactive_mode = False

    print(">>>BLENDERGEN LAUNCH SETTINGS:")
    print(f"Template: {template_path}")
    print(f"Asset: {asset_path}")
    print(f"Material: {mtl_name}")
    print(f"Base Output Dir: {output_path}")
    print(f"Height: {height}")
    print(f"Use Textures: {texture_bool}")
    print(f"Interactive Mode: {interactive_mode}")
    
    try:
        # Verify asset file exists
        if not Path(asset_path).exists():
            print(f"WARNING: Asset file not found at: {asset_path}")
            print("Please check the path and try again.")
            sys.exit(1)
            
        # Create base output directory if it doesn't exist
        # The create_project_folder method handles subdirectories
        os.makedirs(output_path, exist_ok=True) 
        
        # Initialize the BlenderGen instance
        blender_automation = BlenderGen()

        # Process the model (either interactively or in background)
        project_folder = blender_automation.blenderGen_pipeline(
            template_path, asset_path, mtl_name, texture_bool, output_path, height, interactive_mode
        )
        
        if interactive_mode:
             print(f"✓ Blender opened interactively. Project setup in: {project_folder}")
        else:
            print(f"✓ Background processing complete! Files saved to: {project_folder}")
            
    except Exception as e:
        print(f"!!! ERROR in blenderGen.py: {e}")
        sys.exit(1)
