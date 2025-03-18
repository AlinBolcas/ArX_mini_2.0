import os
import sys
import socket
from pathlib import Path

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

# Create necessary directories
output_dirs = [
    "output/images",
    "output/descriptions",
    "output/music_analysis"
]

for dir_path in output_dirs:
    os.makedirs(dir_path, exist_ok=True)

# Function to find first available port
def find_available_port(start_port, max_attempts=10):
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    return start_port + max_attempts

# Print server URLs
def print_server_urls(port):
    print("\n" + "="*50)
    print(f"🚀 Ozart server running at:")
    print(f"   http://localhost:{port}")
    try:
        print(f"   http://{socket.gethostbyname(socket.gethostname())}:{port}")
    except:
        pass
    print("="*50 + "\n")

if __name__ == "__main__":
    if "--streamlit" in sys.argv:
        print("Starting Streamlit UI...")
        from streamlit.web.cli import main as streamlit_main
        sys.argv = ["streamlit", "run", str(ROOT_DIR / "code" / "IV_ui" / "ui_streamlit.py")]
        streamlit_main()
    else:
        # Find an available port
        port = find_available_port(5001)
        
        # Import Flask app directly
        import src.IV_ui.ui_flask as flask_ui
        
        # Print server information
        print_server_urls(port)
        
        # Run Flask app directly without debug mode
        flask_ui.app.run(host='0.0.0.0', port=port, debug=False)
