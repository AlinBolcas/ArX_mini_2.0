import os
import re
import sys
import shutil # Added for cleanup
import platform # Added for OS-specific paths
from pathlib import Path
import io
import contextlib
import concurrent.futures
import traceback
import logging
import subprocess # Added for running commands
from typing import List, Tuple, Optional, Union # Added missing imports from typing

# Basic logging setup
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- File System Utilities ---

def get_project_root() -> Path:
    """Find the project root directory (assuming it contains a '.git' folder or a known file)."""
    current_path = Path(__file__).resolve()
    while not (current_path / '.git').exists() and not (current_path / 'README.md').exists():
        if current_path.parent == current_path:
            # Reached the filesystem root without finding the project marker
            raise FileNotFoundError("Could not determine project root. Place a .git folder or README.md at the root.")
        current_path = current_path.parent
    return current_path

def search_file(filename: str, start_dir: Path = None) -> str | None:
    """
    Recursively search for a file within the project directory.

    Args:
        filename: The name of the file to search for.
        start_dir: The directory to start searching from (defaults to project root).

    Returns:
        The absolute path to the file if found, otherwise None.
    """
    if start_dir is None:
        start_dir = get_project_root()

    logger.info(f"Searching for '{filename}' starting from '{start_dir}'...")
    for root, _, files in os.walk(start_dir):
        if filename in files:
            found_path = os.path.abspath(os.path.join(root, filename))
            logger.info(f"Found '{filename}' at: {found_path}")
            return found_path
    logger.warning(f"File '{filename}' not found in project.")
    return None

def save_file(content: str, base_filename: str, version: int = 0, output_subdir: str = "data/output/code") -> str | None:
    """
    Save content to a versioned file within a subdirectory of the project root.
    Creates the subdirectory if it doesn't exist.

    Args:
        content: The string content to save.
        base_filename: The base name of the file (e.g., "script.py").
        version: The version number (0 for initial, 1+ for revisions).
        output_subdir: The subdirectory relative to the project root.

    Returns:
        The full absolute path to the saved file, or None if saving failed.
    """
    try:
        project_root = get_project_root()
        full_output_dir = project_root / output_subdir
        full_output_dir.mkdir(parents=True, exist_ok=True)

        # Construct versioned filename
        file_stem = Path(base_filename).stem
        file_suffix = Path(base_filename).suffix
        # Use _v0, _v1 etc convention. If version is 0, don't add suffix initially?
        # Let's always add version suffix for clarity, starting from v1 in the main loop.
        # If version is passed as 0, maybe save as base_filename? Let's stick to _v{N}
        version_suffix = f"_v{version}" if version > 0 else "" # Initial save (v0) might not need suffix?
        # Decision: Let's *always* add version, starting v1 from factory loop.
        versioned_filename = f"{file_stem}_v{version}{file_suffix}"
        
        full_path = full_output_dir / versioned_filename

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"File saved successfully at: {full_path}")
        return str(full_path)
    except Exception as e:
        logger.error(f"Error saving file '{versioned_filename}' to '{output_subdir}': {e}")
        return None

# --- Code Processing Utilities ---

def extract_code(text: str, lang: str = "python") -> list[str]:
    """
    Extracts code blocks for a specified language from markdown formatted text.

    Args:
        text: The markdown text containing code blocks.
        lang: The language identifier for the code blocks (e.g., "python", "bash").

    Returns:
        A list of extracted code strings.
    """
    code_blocks = []
    # Pattern to match ```lang ... ``` or ``` ... ``` (if lang is generic)
    pattern = rf"```{lang}?.*?\n([\s\S]*?)\n```"
    matches = re.finditer(pattern, text)

    for match in matches:
        code_blocks.append(match.group(1).strip())

    if not code_blocks:
         # Fallback if no language tag is used, but be cautious as this might grab non-code blocks
         pattern = r"```\n([\s\S]*?)\n```"
         matches = re.finditer(pattern, text)
         for match in matches:
             code_blocks.append(match.group(1).strip())

    if not code_blocks:
        logger.warning("No code blocks found in the provided text.")

    return code_blocks

def test_code_local(code: str, timeout: int = 15) -> tuple[bool, str]:
    """
    (Formerly test_code) Executes Python code string in the *current* environment's memory.
    Captures stdout/stderr. WARNING: Not isolated.

    Args:
        code: The Python code string to execute.
        timeout: Maximum execution time in seconds.

    Returns:
        A tuple containing: (success, output_string)
    """
    logger.info(f"Testing code snippet locally (timeout={timeout}s)...")
    output_buffer = io.StringIO()
    local_namespace = {} # Fresh namespace for each execution

    def run_code_safely():
        try:
            # Redirect stdout and stderr to the buffer
            with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
                exec(code, local_namespace)
            return True # Indicate success
        except Exception:
            # Capture the full traceback if an exception occurs
            output_buffer.write(traceback.format_exc())
            return False # Indicate failure

    success = False
    try:
        # Use ThreadPoolExecutor for timeout capability
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_code_safely)
            success = future.result(timeout=timeout)
        output = output_buffer.getvalue()
        if success:
            logger.info("Local code execution successful.")
            logger.debug(f"Local Execution Output:\n{output}")
        else:
            logger.warning(f"Local code execution failed. Error:\n{output}")

    except concurrent.futures.TimeoutError:
        success = False
        output = f"Error: Local code execution timed out after {timeout} seconds."
        logger.error(output)
    except Exception as e:
        success = False
        output = f"Error: An unexpected error occurred during local code execution setup: {e}\n{traceback.format_exc()}"
        logger.error(output)
    finally:
        output_buffer.close()

    return success, output

# --- Venv Sandboxed Execution ---

def _is_safe_package_name(package_name: str) -> bool:
    """Basic check for potentially unsafe package names."""
    # Allows letters, numbers, underscore, hyphen, dot (for versions/extras like requests[security])
    # Disallows paths, shell characters, etc.
    # WARNING: This is NOT a foolproof security measure.
    safe_pattern = r"^[a-zA-Z0-9_\-\.\[\]]+$"
    return bool(re.match(safe_pattern, package_name))

def test_code_in_venv(
    project_dir: str,               # New: Path to the versioned project directory
    main_script_name: str,        # New: Name of the main script to execute
    requirements: List[str],        # Still needed: List of packages to install
    timeout: int = 120,             # Increased timeout for potentially complex projects
    cleanup_venv: bool = False      # << CHANGED: Keep venv by default for debugging
) -> tuple[bool, str]:
    """
    Tests the main Python script within a project directory in a dedicated venv.
    1. Creates a venv within a temporary location or designated venv dir.
    2. Installs requirements from the provided list (or reads from project_dir/requirements.txt).
    3. Runs the specified main script from the project directory.
    4. Captures output/errors.
    5. Cleans up the venv (optional).

    Args:
        project_dir: The absolute path to the project directory containing the code and requirements.txt.
        main_script_name: The filename of the main Python script to execute within the project directory.
        requirements: A list of package names to install (e.g., ['requests', 'numpy']).
                      If empty, attempts to install from requirements.txt if it exists.
        timeout: Timeout in seconds for the code execution subprocess.
        cleanup_venv: If True, delete the venv directory after execution.

    Returns:
        A tuple containing: (success, output_string)
    """
    project_path = Path(project_dir).resolve() # Ensure absolute path
    main_script_path = project_path / main_script_name
    requirements_path = project_path / "requirements.txt"
    project_name = project_path.name # Use the directory name (e.g., my_project_v1)

    # Define venv location (could be within project_dir or a central place)
    # Let's keep it separate for cleanliness
    venv_base_dir = get_project_root() / "data" / "output" / "venvs"
    venv_path = venv_base_dir / f"{project_name}_venv"

    logger.info(f"Preparing venv sandbox for project: {project_path} at {venv_path}")
    if not cleanup_venv:
        logger.warning(f"Venv cleanup is disabled. Directory will remain at: {venv_path}")

    # --- Input Validation ---
    if not project_path.is_dir():
        return False, f"Sandbox Setup Error: Project directory not found: {project_path}"
    if not main_script_path.is_file():
        return False, f"Sandbox Setup Error: Main script not found: {main_script_path}"

    # Ensure venv base directory exists
    venv_base_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine Requirements --- 
    # Use provided requirements list preferentially. If empty, try reading from file.
    reqs_to_install = requirements
    if not reqs_to_install and requirements_path.is_file():
        try:
            with open(requirements_path, "r", encoding="utf-8") as f:
                reqs_to_install = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f"Read {len(reqs_to_install)} requirements from {requirements_path}")
        except Exception as e:
            logger.warning(f"Could not read requirements file {requirements_path}, proceeding without: {e}")
            reqs_to_install = [] # Ensure it's an empty list on error
    
    safe_requirements = [pkg for pkg in reqs_to_install if _is_safe_package_name(pkg)]
    unsafe_skipped = [pkg for pkg in reqs_to_install if not _is_safe_package_name(pkg)]
    if unsafe_skipped:
        logger.warning(f"Skipping potentially unsafe packages: {unsafe_skipped}")

    # --- Create Venv --- (Same as before, using venv_path)
    try:
        logger.info(f"Creating virtual environment at: {venv_path}")
        create_venv_cmd = [sys.executable, '-m', 'venv', str(venv_path)]
        result = subprocess.run(create_venv_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Venv creation failed: {result.stderr}")
        logger.info("Venv created successfully.")
    except Exception as e:
        logger.error(f"Error creating venv: {e}")
        if venv_path.exists() and cleanup_venv: shutil.rmtree(venv_path, ignore_errors=True)
        return False, f"Sandbox Setup Error: Could not create venv - {e}"

    # --- Determine Venv Executable Paths --- (Same as before)
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    # --- Install Requirements --- (Installs safe_requirements list)
    install_output = ""
    pip_install_success = True # Assume success initially
    if safe_requirements:
        try:
            logger.info(f"Installing {len(safe_requirements)} requirements using {pip_path}...")
            # Create a temporary requirements file within the venv dir for pip
            temp_req_path = venv_path / "temp_requirements.txt"
            with open(temp_req_path, "w", encoding="utf-8") as f:
                f.write("\n".join(safe_requirements))
            
            install_cmd = [str(pip_path), 'install', '-r', str(temp_req_path)]
            # Increased timeout for pip install
            result = subprocess.run(install_cmd, capture_output=True, text=True, check=False, timeout=600)
            install_output = result.stdout + "\n--- STDERR ---\n" + result.stderr # Combine outputs
            
            # Clean up temporary file
            temp_req_path.unlink(missing_ok=True)
            
            if result.returncode != 0:
                 pip_install_success = False
                 # Log the error prominently
                 logger.error(f"pip install command failed with exit code {result.returncode}.")
                 logger.error(f"Pip Install Output:\n{install_output}")
                 # Raise runtime error to stop execution here, message includes output
                 raise RuntimeError(f"pip install failed. See logs for details.\nOutput:\n{install_output}")
            else:
                 # Log success and full output at INFO level for visibility
                 logger.info("Requirements installation command completed successfully.")
                 logger.info(f"Pip Install Output:\n{install_output}")

        except subprocess.TimeoutError as e:
             logger.error(f"Error installing requirements: Timeout after {e.timeout} seconds.")
             if venv_path.exists() and cleanup_venv: shutil.rmtree(venv_path, ignore_errors=True)
             return False, f"Sandbox Setup Error: Failed to install requirements - Timeout. Output:\n{install_output}"

        except Exception as e:
            # Catch the explicit RuntimeError or other exceptions
            logger.error(f"Error installing requirements: {e}")
            # Ensure install_output is included if available
            error_message = f"Sandbox Setup Error: Failed to install requirements - {e}"
            if install_output:
                error_message += f"\nOutput:\n{install_output}"
            if venv_path.exists() and cleanup_venv: shutil.rmtree(venv_path, ignore_errors=True)
            return False, error_message
    else:
        logger.info("No safe requirements to install.")

    # --- Execute Code --- (Run main_script_path from project_path working directory)
    success = False
    exec_output = ""
    try:
        logger.info(f"Executing code: {main_script_path} using {python_path} from CWD: {project_path}")
        # Execute the script with the project directory as the CWD
        exec_cmd = [str(python_path), str(main_script_name)] # Execute by name relative to CWD
        result = subprocess.run(
            exec_cmd, 
            cwd=str(project_path), # Set the Current Working Directory
            capture_output=True, 
            text=True, 
            check=False, 
            timeout=timeout
        )
        # Return FULL output (stdout + stderr) regardless of success/fail for better debugging
        exec_output = f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"

        if result.returncode == 0:
            logger.info("Code execution in venv successful.")
            logger.info(f"Execution Output:\n{exec_output}") # Log full output on success too
            success = True
        else:
            logger.warning(f"Code execution in venv failed with exit code {result.returncode}.")
            logger.warning(f"Execution Output:\n{exec_output}") # Log full output on failure
            success = False # Explicitly set

    except subprocess.TimeoutExpired:
        logger.error(f"Code execution timed out after {timeout} seconds.")
        exec_output = f"Execution Error: Timeout after {timeout} seconds."
        success = False
    except Exception as e:
        logger.error(f"Error during code execution in venv: {e}")
        exec_output = f"Execution Error: {e}\n{traceback.format_exc()}"
        success = False

    # --- Cleanup ---
    if cleanup_venv:
        try:
            logger.info(f"Cleaning up venv: {venv_path}")
            shutil.rmtree(venv_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup venv {venv_path}: {e}") 

    logger.info(f"Venv test completed. Success: {success}")
    return success, exec_output.strip()

# --- Path Fixing for Imports ---
# Add project root to sys.path to allow imports like `from src.I_integrations...`
try:
    project_root_path = str(get_project_root())
    if project_root_path not in sys.path:
        sys.path.insert(0, project_root_path)
        logger.debug(f"Added project root to sys.path: {project_root_path}")
except FileNotFoundError as e:
    logger.error(f"Could not add project root to path: {e}")
