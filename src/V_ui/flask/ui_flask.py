import sys
import os
from pathlib import Path

# Add parent directory to Python path and set ROOT_DIR
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import Flask and extensions
from flask import Flask, render_template, request, jsonify, send_from_directory, abort, session
from flask_cors import CORS

# Import Ozart agent
from src.III_ozart.ozart_simple import OzartAgent

# Rest of the imports
from PIL import Image
import logging
import socket
import requests
import threading
import queue
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure Flask - minimal config
app.config.update(
    SECRET_KEY=os.urandom(24)
)

# Configure logging to be less verbose
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# Define storage directories
OUTPUT_DIR = os.path.abspath("output")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
DESC_DIR = os.path.join(OUTPUT_DIR, "descriptions")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "music_analysis")
CONFIG_DIR = os.path.join(OUTPUT_DIR, "config")

# Create the directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Create static folders and fallback image
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
STATIC_DATA_DIR = os.path.join(STATIC_DIR, 'data')
os.makedirs(STATIC_DATA_DIR, exist_ok=True)

FALLBACK_PATH = os.path.join(STATIC_DATA_DIR, 'fallback.png')
if not os.path.exists(FALLBACK_PATH):
    img = Image.new('RGB', (100, 100), color='black')
    img.save(FALLBACK_PATH)

# Create favicon
FAVICON_PATH = os.path.join(STATIC_DIR, 'images')
os.makedirs(FAVICON_PATH, exist_ok=True)
if not os.path.exists(os.path.join(FAVICON_PATH, 'favicon.png')):
    img = Image.new('RGB', (32, 32), color='black')
    img.save(os.path.join(FAVICON_PATH, 'favicon.png'))

# Path for custom prompts
CUSTOM_PROMPTS_PATH = os.path.join(CONFIG_DIR, "custom_prompts.json")

# Path for default prompts
DEFAULT_PROMPTS_PATH = Path("data/default_prompts.json")

# Load custom prompts if they exist
custom_prompts = None
if os.path.exists(CUSTOM_PROMPTS_PATH):
    try:
        with open(CUSTOM_PROMPTS_PATH, 'r', encoding='utf-8') as f:
            custom_prompts = json.load(f)
            print("✓ Loaded custom prompts from", CUSTOM_PROMPTS_PATH)
    except Exception as e:
        print(f"Error loading custom prompts: {e}")

# Initialize agent
ozart_agent = OzartAgent(debug=False, custom_prompts=custom_prompts)
print("✓ Ozart Agent initialized in ui_flask.py")
print(f"API Keys configured: OpenAI={ozart_agent.has_openai_key}, Replicate={ozart_agent.has_replicate_key}")

# Thread-safe queues
processing_queue = queue.Queue()
error_queue = queue.Queue()

@app.route('/')
def index():
    artworks = ozart_agent.get_all_artworks()
    artwork_count = len(artworks)
    recent_title = "None yet"
    
    if artworks:
        metadata = artworks[0].get("metadata", {})
        song_data = metadata.get("song", {})
        recent_title = song_data.get("title", "None yet")
    
    return render_template('index.html', 
                         artwork_count=artwork_count,
                         recent_title=recent_title)

@app.route('/process', methods=['POST'])
def process_song():
    """Process song"""
    song_input = request.json.get('song')
    if not song_input:
        return jsonify({'error': 'No song provided'})
    
    # Start processing in background thread without any key checks
    thread = threading.Thread(target=process_song_thread, args=(song_input,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'processing'})

@app.route('/status')
def get_status():
    try:
        errors = []
        while not error_queue.empty():
            try:
                errors.append(error_queue.get_nowait())
            except queue.Empty:
                break
        
        artworks = ozart_agent.get_all_artworks()
        formatted_artworks = [format_artwork(artwork) for artwork in artworks]
        
        return jsonify({
            'processing': not processing_queue.empty(),
            'errors': errors,
            'artwork_count': len(artworks),
            'artworks': formatted_artworks
        })
    except Exception as e:
        return jsonify({
            'processing': False,
            'errors': [str(e)],
            'artwork_count': 0,
            'artworks': []
        })

@app.route('/output/<path:filename>')
def serve_file(filename):
    """Serve files from output directory"""
    try:
        requested_path = os.path.abspath(os.path.join(OUTPUT_DIR, filename))
        if not requested_path.startswith(OUTPUT_DIR):
            abort(403)
        
        directory = os.path.dirname(requested_path)
        file_name = os.path.basename(requested_path)
        
        return send_from_directory(directory, file_name)
    except Exception:
        abort(404)

def format_artwork(artwork):
    """Format artwork data for JSON response"""
    metadata = artwork.get("metadata", {})
    image_path = metadata.get("image", {}).get("path", "")
    song_data = metadata.get("song", {})
    
    # Get audio path from session path
    audio_path = None
    if "session_path" in song_data:
        session_dir = song_data["session_path"]
        if os.path.exists(session_dir):
            for file in os.listdir(session_dir):
                if file.endswith('.wav'):
                    audio_path = os.path.join(session_dir, file)
                    audio_path = os.path.relpath(audio_path, OUTPUT_DIR)
                    break
    
    return {
        'image_path': image_path,
        'song_title': song_data.get('title', 'Unknown'),
        'artist': song_data.get('artist', 'Unknown'),
        'interpretation': song_data.get('description', ''),
        'prompt': metadata.get("image", {}).get("prompt", ""),
        'human_time': metadata.get("human_time", "Unknown time"),
        'audio_path': audio_path
    }

def process_song_thread(song_input):
    """Process song in background thread"""
    try:
        processing_queue.put(song_input)
        result = ozart_agent.process_single_song(song_input)
        
        if 'error' in result:
            error_queue.put(result['error'])
    except Exception as e:
        error_queue.put(str(e))
    finally:
        try:
            processing_queue.get_nowait()
        except queue.Empty:
            pass

@app.route('/check-keys')
def check_keys():
    """Check if API keys are available"""
    # Always return true to bypass key input requirement
    return jsonify({
        'has_openai': True,
        'has_replicate': True,
        'keys_required': False
    })

@app.route('/save-keys', methods=['POST'])
def save_api_keys():
    """Save API keys to session"""
    # Just return success without saving to session
    return jsonify({'success': True})

@app.route('/get-prompts')
def get_prompts():
    """Get current prompts (custom or default)"""
    try:
        # First try to load from custom prompts file
        if os.path.exists(CUSTOM_PROMPTS_PATH):
            with open(CUSTOM_PROMPTS_PATH, 'r') as f:
                custom_prompts = json.load(f)
                return jsonify({
                    'success': True, 
                    'prompts': custom_prompts,
                    'is_custom': True
                })
        
        # Otherwise load default prompts
        if DEFAULT_PROMPTS_PATH.exists():
            with open(DEFAULT_PROMPTS_PATH, 'r') as f:
                default_prompts = json.load(f)
                return jsonify({
                    'success': True, 
                    'prompts': default_prompts,
                    'is_custom': False
                })
                
        return jsonify({'success': False, 'error': 'Default prompts not found'})
    except Exception as e:
        print(f"Error getting prompts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get-default-prompts')
def get_default_prompts():
    """Get default prompts (for restore defaults button)"""
    try:
        # Load default prompts
        if DEFAULT_PROMPTS_PATH.exists():
            with open(DEFAULT_PROMPTS_PATH, 'r') as f:
                default_prompts = json.load(f)
                return jsonify({
                    'success': True, 
                    'prompts': default_prompts
                })
                
        return jsonify({'success': False, 'error': 'Default prompts not found'})
    except Exception as e:
        print(f"Error getting default prompts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/save-prompts', methods=['POST'])
def save_prompts():
    """Save custom prompts to a file"""
    try:
        data = request.json
        
        # Validate that all required fields are present
        required_fields = [
            'music_analysis_system_prompt', 
            'music_analysis_user_prompt',
            'image_prompt_system_prompt',
            'image_prompt_user_prompt'
        ]
        
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    'success': False, 
                    'error': f'Missing required field: {field}'
                })
        
        # Create prompts object
        prompts = {
            'music_analysis_system_prompt': data.get('music_analysis_system_prompt', ''),
            'music_analysis_user_prompt': data.get('music_analysis_user_prompt', ''),
            'image_prompt_system_prompt': data.get('image_prompt_system_prompt', ''),
            'image_prompt_user_prompt': data.get('image_prompt_user_prompt', '')
        }
        
        # Save to file
        with open(CUSTOM_PROMPTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(prompts, f, ensure_ascii=False, indent=2)
        
        # Also save to session for immediate use
        session['custom_prompts'] = prompts
        
        # Update the agent with new prompts
        ozart_agent.load_prompts(prompts)
        ozart_agent.music_analyzer.load_prompts(prompts)
        
        print(f"✓ Custom prompts saved to {CUSTOM_PROMPTS_PATH}")
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error saving prompts: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from data directory"""
    data_dir = os.path.join(ROOT_DIR, 'data')
    try:
        return send_from_directory(data_dir, filename)
    except Exception as e:
        print(f"Error serving {filename}: {str(e)}")
        abort(404)

# Add a simple test route
@app.route('/test')
def test_route():
    """Test route to confirm app is running"""
    return "Ozart is running!"

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(error):
    print(f"Server error: {error}")
    return jsonify({"error": "Internal server error"}), 500 