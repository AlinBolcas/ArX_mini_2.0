import subprocess
import os
import sys
from pathlib import Path

# Calculate the project root (which is three directories up from this file)
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

class BlenderViewer:
    def __init__(self, version="4.1.1"):
        self.version = version
        self.blender_executable = self.get_blender_executable_path()

    def get_blender_executable_path(self):
        if os.name == 'posix':  # macOS/Linux
            if os.uname().sysname == 'Darwin':  # macOS
                path = f"/Applications/Blender.app/Contents/MacOS/Blender"
            else:  # Linux
                path = "/usr/bin/blender"
        else:
            raise EnvironmentError("Unsupported operating system")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Blender executable not found at {path}")
        return path

    def launch_blender_with_script(self, script_path, glb_file, output_path):
        script_path = os.path.abspath(script_path)
        command = [
            self.blender_executable,
            "/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_realTime_05.blend",
            "--python", script_path,
            "--",
            glb_file,
            output_path
        ]
        subprocess.run(command, check=True)

if __name__ == "__main__":
    viewer = BlenderViewer()
    glb_file = "/Users/arvolve/Coding/ARX_02/output/lol.glb"
    output_path = "change this to data/output/ whatever name is good here"
    script_path = os.path.join(os.path.dirname(__file__), "bpy_script.py")
    viewer.launch_blender_with_script(script_path, glb_file, output_path)