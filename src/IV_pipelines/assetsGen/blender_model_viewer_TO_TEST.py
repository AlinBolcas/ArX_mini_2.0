import os
import sys
import subprocess
from pathlib import Path
import logging
import tempfile
import shutil
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def find_blender_executable():
    """Find Blender executable on different platforms."""
    possible_paths = [
        # macOS
        "/Applications/Blender.app/Contents/MacOS/Blender",
        # Common Linux paths
        "/usr/bin/blender",
        "/usr/local/bin/blender",
        # Common Windows paths
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
    ]
    
    # Add any path from environment variable
    if os.environ.get('BLENDER_PATH'):
        possible_paths.insert(0, os.environ.get('BLENDER_PATH'))
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found Blender at: {path}")
            return path
    
    # If not found, try to find in PATH
    try:
        # Try 'which blender' or 'where blender' depending on platform
        if sys.platform.startswith('win'):
            result = subprocess.run(["where", "blender"], capture_output=True, text=True)
        else:
            result = subprocess.run(["which", "blender"], capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout:
            blender_path = result.stdout.strip().split("\n")[0]
            logger.info(f"Found Blender in PATH: {blender_path}")
            return blender_path
    except Exception as e:
        logger.warning(f"Error finding Blender in PATH: {e}")
    
    logger.error("Could not find Blender executable. Please install Blender or set BLENDER_PATH environment variable.")
    return None

def render_3d_model(model_path, output_dir=None, output_format="PNG", resolution=(1920, 1080), 
                    template_blend=None, camera_views=None, bg_color=(0.05, 0.05, 0.05)):
    """
    Render a 3D model using Blender.
    
    Args:
        model_path: Path to the GLB model file
        output_dir: Directory to save renders (default: same as model directory)
        output_format: Format to render (PNG, JPEG, etc.)
        resolution: Tuple of (width, height) for render
        template_blend: Optional path to template Blender file
        camera_views: Optional list of camera positions (default rotations if None)
        bg_color: Background color as RGB tuple (0-1 range)
        
    Returns:
        List of paths to rendered images
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Set default output directory if not specified
    if not output_dir:
        output_dir = model_path.parent / "renders"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get path to the Blender script
    blender_script_path = Path(__file__).parent / "blender_viewer.py"
    
    # Get path to Blender executable
    blender_path = find_blender_executable()
    if not blender_path:
        return None
    
    # Set default template blend if not specified
    if not template_blend:
        template_blend = Path(__file__).parent / "templates" / "studio_template.blend"
        # Create a default template if needed
        if not template_blend.exists():
            logger.info("Template blend file not found. Using empty scene.")
            template_blend = None
    
    # Prepare camera views if not specified
    if not camera_views:
        # Default to four camera angles at 45 degree intervals
        camera_views = [
            (3, 0, 30),     # Front-right, slightly elevated
            (0, 3, 30),     # Back-right, slightly elevated
            (-3, 0, 30),    # Back-left, slightly elevated
            (0, -3, 30),    # Front-left, slightly elevated
        ]
    
    # Prepare command-line arguments 
    cmd = [
        blender_path,
        "--background"
    ]
    
    # Add template file if it exists
    if template_blend and Path(template_blend).exists():
        cmd.extend(["--python-expr", f"import bpy; bpy.ops.wm.open_mainfile(filepath='{template_blend}')"])
    
    # Add the script and its arguments
    cmd.extend([
        "--python", str(blender_script_path),
        "--", 
        "--model", str(model_path),
        "--output", str(output_dir),
        "--format", output_format,
        "--resolution", f"{resolution[0]},{resolution[1]}",
        "--bg-color", f"{bg_color[0]},{bg_color[1]},{bg_color[2]}"
    ])
    
    # Add camera views
    for i, view in enumerate(camera_views):
        cmd.extend(["--camera-view", f"{view[0]},{view[1]},{view[2]}"])
    
    # Log the command
    logger.info(f"Running Blender with command: {' '.join(str(arg) for arg in cmd)}")
    
    try:
        # Run Blender process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Check for errors
        if return_code != 0:
            stderr = process.stderr.read()
            logger.error(f"Blender returned error code {return_code}: {stderr}")
            return None
        
        # Collect rendered image paths
        render_files = list(output_dir.glob(f"*.{output_format.lower()}"))
        logger.info(f"Rendered {len(render_files)} images to {output_dir}")
        
        return [str(f) for f in render_files]
    
    except Exception as e:
        logger.error(f"Error running Blender: {e}")
        return None

def create_turntable_animation(model_path, output_path=None, fps=30, duration=5.0, 
                              template_blend=None, resolution=(1920, 1080), quality=90):
    """
    Create a turntable animation of a 3D model using Blender.
    
    Args:
        model_path: Path to the GLB model file
        output_path: Path for the output video (default: model_directory/model_name_turntable.mp4)
        fps: Frames per second
        duration: Duration in seconds
        template_blend: Optional path to template Blender file
        resolution: Tuple of (width, height) for render
        quality: Video encoding quality (0-100)
        
    Returns:
        Path to the rendered video file
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return None
    
    # Set default output path if not specified
    if not output_path:
        output_path = model_path.parent / f"{model_path.stem}_turntable.mp4"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get path to the Blender script
    blender_script_path = Path(__file__).parent / "blender_viewer.py"
    
    # Get path to Blender executable
    blender_path = find_blender_executable()
    if not blender_path:
        return None
    
    # Set default template blend if not specified
    if not template_blend:
        template_blend = Path(__file__).parent / "templates" / "studio_template.blend"
        # Create a default template if needed
        if not template_blend.exists():
            logger.info("Template blend file not found. Using empty scene.")
            template_blend = None
    
    # Prepare command-line arguments
    cmd = [
        blender_path,
        "--background"
    ]
    
    # Add template file if it exists
    if template_blend and Path(template_blend).exists():
        cmd.extend(["--python-expr", f"import bpy; bpy.ops.wm.open_mainfile(filepath='{template_blend}')"])
    
    # Add the script and its arguments
    cmd.extend([
        "--python", str(blender_script_path),
        "--", 
        "--model", str(model_path),
        "--output", str(output_path),
        "--turntable",
        "--fps", str(fps),
        "--duration", str(duration),
        "--resolution", f"{resolution[0]},{resolution[1]}",
        "--quality", str(quality)
    ])
    
    # Log the command
    logger.info(f"Running Blender with command: {' '.join(str(arg) for arg in cmd)}")
    
    try:
        # Run Blender process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(output.strip())
        
        # Get return code
        return_code = process.poll()
        
        # Check for errors
        if return_code != 0:
            stderr = process.stderr.read()
            logger.error(f"Blender returned error code {return_code}: {stderr}")
            return None
        
        # Wait for file to be fully written
        if output_path.exists():
            logger.info(f"Rendered turntable animation to {output_path}")
            return str(output_path)
        else:
            logger.error(f"Expected output file not found: {output_path}")
            return None
    
    except Exception as e:
        logger.error(f"Error running Blender: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render 3D models with Blender")
    parser.add_argument("model_path", help="Path to the 3D model file (GLB/GLTF/OBJ)")
    parser.add_argument("--output", help="Output directory for renders", default=None)
    parser.add_argument("--turntable", action="store_true", help="Create turntable animation")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of turntable animation in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for animation")
    parser.add_argument("--template", help="Path to template Blender file", default=None)
    parser.add_argument("--resolution", help="Render resolution (WxH)", default="1920x1080")
    
    args = parser.parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    if args.turntable:
        # Generate turntable animation
        output_path = create_turntable_animation(
            args.model_path,
            output_path=args.output,
            fps=args.fps,
            duration=args.duration,
            template_blend=args.template,
            resolution=(width, height)
        )
        
        if output_path:
            print(f"Turntable animation created: {output_path}")
        else:
            print("Failed to create turntable animation")
    else:
        # Generate still images
        output_paths = render_3d_model(
            args.model_path,
            output_dir=args.output,
            template_blend=args.template,
            resolution=(width, height)
        )
        
        if output_paths:
            print(f"Rendered {len(output_paths)} images:")
            for path in output_paths:
                print(f"  - {path}")
        else:
            print("Failed to render model")