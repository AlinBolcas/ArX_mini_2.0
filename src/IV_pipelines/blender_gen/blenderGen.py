import subprocess
import os


import sys
from pathlib import Path
# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

class BlenderGen:
    def __init__(self, version="4.1.1"):
        self.version = version
        self.blender_executable = self.get_blender_executable_path()

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

    def launch_blender_with_script(self, script_path, env_vars):
        script_path = os.path.abspath(script_path)
        command = [self.blender_executable, '--background', '--python', script_path, '--']
        print(f"Running command: {' '.join(command)}")
        subprocess.run(command, check=True, env={**os.environ, **env_vars})

    def render_model(self, template_path, obj_path, mtl_name, texture_bool, output_path, height):
        print(">>>BLENDER RENDER META-SETTINGS:\n", template_path, obj_path, mtl_name, output_path, height)

        env_vars = {
            'TEMPLATE_PATH': template_path,
            'OBJ_PATH': obj_path,
            'MTL_NAME': mtl_name,
            'OUTPUT_PATH': output_path,
            'HEIGHT': str(height),
            'TEXTURE_BOOL': str(texture_bool),
            'SAVE_BLEND': 'True'
        }

        self.launch_blender_with_script(str(project_root / "modules/blender_gen/autoRender_mats.py"), env_vars)

    def batch_render_models(self, batch_file):
        with open(batch_file, 'r') as file:
            lines = file.readlines()

        for line in lines:
            template_path, obj_path, mtl_name, output_path, height = line.strip().split(',')
            self.render_model(template_path, obj_path, mtl_name, output_path, float(height))
            
if __name__ == "__main__":
    # Define your paths
    
    template_path = str("/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_realTime_02.blend")
    obj_path = str(project_root / "output/threedModels/test.obj")
    mtl_name = "grey_procedural_MAT"
    output_path = str(project_root / "output/threedModels/renders_TEST")
    height = 1.6
    texture_bool = True

    print(">>>BLENDER RENDER META-SETTINGS:\n", 
          ">>>TEMPLATE_PATH\n:", template_path, "\n",
          ">>>OBJ_path:", obj_path, "\n",
          ">>>MTL_name:", mtl_name, "\n",
          ">>>OUTPUT_PATH:", output_path, "\n",
          ">>>HEIGHT:", height, "\n",
          ">>>TEXTURE_BOOL:", texture_bool, "\n")
    
    try:
        blender_automation = BlenderGen()
        blender_automation.render_model(template_path, obj_path, mtl_name, texture_bool, output_path, height)
    except Exception as e:
        print(f"An error occurred: {e}")
