import streamlit as st
import requests
import time
import os
import json
from PIL import Image
from io import BytesIO
from datetime import datetime
import random
import uuid
import sys
import threading
import queue  # Add this back for thread communication

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Ozart agent
from III_ozart.ozart_simple import OzartAgent

# Create storage directories in project root
IMG_DIR = os.path.join("output", "images")
DESC_DIR = os.path.join("output", "descriptions")
AUDIO_DIR = os.path.join("output", "music_analysis")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DESC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# Create a global error queue for thread-safe communication
error_queue = queue.Queue()

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "errors" not in st.session_state:
    st.session_state.errors = []
if "ozart_agent" not in st.session_state:
    # Initialize agent with just debug flag
    st.session_state.ozart_agent = OzartAgent(debug=False)
if "last_processed" not in st.session_state:
    st.session_state.last_processed = None
if "song_queue" not in st.session_state:
    st.session_state.song_queue = []

def display_processing_message():
    """Display a simple processing message."""
    st.info("Please wait about 1 minute, your song is processing ...")

def display_errors():
    """Display any errors from processing."""
    # Check the error queue and add to session state
    while not error_queue.empty():
        try:
            error = error_queue.get_nowait()
            st.session_state.errors.append(error)
        except queue.Empty:
            break
            
    if st.session_state.errors:
        for error in st.session_state.errors:
            st.error(error)
        # Clear errors after displaying
        st.session_state.errors = []

def process_songs_thread(agent, songs):
    """Process songs in a background thread"""
    try:
        for song in songs:
            try:
                result = agent.process_single_song(song)
                
                if "error" in result:
                    error_msg = f"Error processing {song}: {result['error']}"
                    error_queue.put(error_msg)
                    continue
                
                # Instead of directly updating session state, store result in queue
                st.session_state.last_result = result
                
            except Exception as e:
                error_msg = f"Error processing {song}: {str(e)}"
                error_queue.put(error_msg)
                
    except Exception as e:
        error_msg = f"Thread error: {str(e)}"
        error_queue.put(error_msg)
    finally:
        # Signal completion through session state
        st.session_state.processing = False
        st.session_state.needs_refresh = True  # Add this flag

def start_processing(songs_input):
    """Start background processing of songs"""
    if st.session_state.processing:
        return False
    
    songs = [s.strip() for s in songs_input.split(",") if s.strip()]
    if not songs:
        return False
    
    # Get a copy of the agent to pass to thread
    agent = st.session_state.ozart_agent
    
    # Start processing in background thread
    st.session_state.processing = True
    thread = threading.Thread(target=process_songs_thread, args=(agent, songs))
    thread.daemon = True
    thread.start()
    
    # Store thread reference
    st.session_state.processing_thread = thread
    
    return True

def find_audio_file(song_title, artwork_metadata=None):
    """Find audio file for a song, first checking metadata if available"""
    if not song_title:
        return None
    
    # First check if metadata already has audio_path
    if artwork_metadata and "audio_path" in artwork_metadata:
        audio_path = artwork_metadata.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            return audio_path
    
    # Check the dedicated audio directory first
    audio_dir = os.path.join(AUDIO_DIR, "audio")
    if os.path.exists(audio_dir):
        # Clean the song title for filename matching
        clean_title = ''.join(c if c.isalnum() or c in ' -_' else '_' for c in song_title).lower()
        
        # First try exact match in the audio directory
        for file in os.listdir(audio_dir):
            if file.endswith((".wav", ".mp3", ".webm")):
                if clean_title in file.lower():
                    return os.path.join(audio_dir, file)
    
    # Try all music_analysis subdirectories as a fallback
    for root, dirs, files in os.walk(AUDIO_DIR):
        for file in files:
            if file.endswith((".wav", ".mp3", ".webm")):
                # Try to match the song title
                if song_title.lower() in file.lower():
                    return os.path.join(root, file)
    
    return None

def display_artwork(artwork, idx, cols):
    """Display a single artwork in the UI"""
    # Extract metadata and image
    metadata = artwork["metadata"]
    image = artwork["image"]
    
    # Calculate which column to use
    col_idx = idx % len(cols)
    
    with cols[col_idx]:
        # Display the image
        st.image(image, width=750)
        
        # Display artwork info in expandable section
        with st.expander("Show details"):
            # Get song info from metadata
            song_title = metadata.get('song', {}).get('title', 'Unknown Song')
            artist = metadata.get('song', {}).get('artist', 'Unknown Artist')
            st.markdown(f"### {song_title} by {artist}")
            
            # Try to find and display audio player
            audio_path = metadata.get('song', {}).get('audio_path')
            if audio_path and os.path.exists(audio_path):
                st.markdown("#### Listen to the song:")
                try:
                    st.audio(audio_path)
                except Exception as e:
                    st.error(f"Could not play audio: {str(e)}")
            
            # Show the song analysis
            st.markdown("### Song Analysis")
            analysis = metadata.get('song', {}).get('interpretation', 'No analysis available')
            st.markdown(analysis)
            
            # Show the generation prompt
            st.markdown("### Image Prompt")
            st.markdown(f"```\n{metadata.get('image', {}).get('prompt', 'No prompt available')}\n```")
            
            # Show creation time
            st.markdown(f"**Created**: {metadata.get('human_time', 'Unknown time')}")

def main():
    # Initialize needs_refresh flag if not exists
    if "needs_refresh" not in st.session_state:
        st.session_state.needs_refresh = False

    # Check for processing completion first
    if 'processing_thread' in st.session_state and st.session_state.processing:
        if not st.session_state.processing_thread.is_alive():
            st.session_state.processing = False
            st.session_state.last_processed = datetime.now()
            st.session_state.needs_refresh = True
            time.sleep(0.5)  # Small delay to ensure files are written
            st.rerun()

    # Apply custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .stat-header {
        font-size: 0.8rem;
        color: #888;
        margin-bottom: 0px;
    }
    .stat-value {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 0px;
    }
    /* Hide "No artworks yet" message */
    .empty-gallery-message {
        display: none;
    }
    /* Custom button styling */
    .stButton button {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Layout: header with simple image
    cover_paths = [
        "cover_02.png",
        "data/cover_02.png", 
        "../data/cover_02.png",
        "/app/data/cover_02.png"
    ]
    
    cover_loaded = False
    for path in cover_paths:
        if os.path.exists(path):
            cover_img = Image.open(path)
            col1, col2, col3 = st.columns([1, 10, 1])
            with col2:
                st.image(cover_img, width=1200)
            cover_loaded = True
            break

    # Layout: sidebar with stats
    with st.sidebar:
        # Try to find and display profile image
        profile_paths = [
            "profile_02.jpg",
            "data/profile_02.jpg", 
            "/app/data/profile_02.jpg",
            "../data/profile_02.jpg"
        ]
        
        for path in profile_paths:
            if os.path.exists(path):
                try:
                    profile_img = Image.open(path)
                    st.image(profile_img, width=None)
                    break
                except Exception:
                    pass
        
        # Add separator
        st.markdown("---")
        
        # Ozart description
        st.markdown("""
        I am an autonomous AI artist interpreting songs and transforming them into artworks. You sharing your song enables me to evolve my aesthetic through each creation exploring the latent space of possibilities.
        """)
        
        # Statistics section
        st.markdown("---")
        
        # Total artworks
        artworks = st.session_state.ozart_agent.get_all_artworks()
        artwork_count = len(artworks)
        st.markdown(f"<p class='stat-header'>Total Artworks</p><p class='stat-value'>{artwork_count}</p>", unsafe_allow_html=True)
        
        # Get artworks for stats
        artworks = st.session_state.ozart_agent.get_all_artworks()
        
        # Recent Song
        if artworks:
            recent_metadata = artworks[0]["metadata"] 
            recent_title = recent_metadata.get("song_title", "None yet")
            st.markdown(f"<p class='stat-header'>Recent Song</p><p class='stat-value'>{recent_title}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='stat-header'>Recent Song</p><p class='stat-value'>None yet</p>", unsafe_allow_html=True)

    # Input section with minimal styling
    st.markdown("## Name your song")
    
    # Container for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        song_input = st.text_input(
            "Song names", 
            placeholder="E.g., song name, Spotify link, YouTube link", 
            label_visibility="collapsed",
            key="song_input_field"  # Use consistent key
        )
    
    with col2:
        process_btn = st.button("Check In", disabled=st.session_state.processing, use_container_width=True)

    # Process button clicks - simpler logic
    if process_btn and song_input and not st.session_state.processing:
        # Start processing the song
        if start_processing(song_input):
            st.success(f"Started processing: {song_input}")
            time.sleep(0.5)  # Small delay to ensure thread starts
            st.rerun()
    
    # Display the progress bar and status updates
    if st.session_state.processing:
        display_processing_message()
    else:
        # Display any errors from previous processing
        display_errors()

    # After the input field, add a separator and artwork count
    st.markdown("---")

    # Display artwork count in a more subtle way
    st.markdown(f"""
    <p style="color: #999999; font-size: 14px; margin-bottom: 20px;">
        Total Artworks: {artwork_count}
    </p>
    """, unsafe_allow_html=True)

    # Display all artworks and handle refresh
    artworks = st.session_state.ozart_agent.get_all_artworks()
    if artworks:
        cols = st.columns(3)
        for idx, artwork in enumerate(artworks):
            display_artwork(artwork, idx, cols)
    
    # Keep checking for completion while showing artworks
    if st.session_state.processing:
        time.sleep(2)  # Check every second
        st.rerun()

if __name__ == "__main__":
    # Set page config for better appearance
    st.set_page_config(
        page_title="Ozart",
        page_icon="⬜",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    main() 