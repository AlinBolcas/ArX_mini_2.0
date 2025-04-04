#!/usr/bin/env python3
"""
Video Utilities

A collection of video manipulation utilities for:
- Creating seamless loops
- Speeding up videos
- Extracting frames from videos
- Converting videos to GIFs
- Optimizing video file size
- Converting MOV to MP4

Requirements:
- FFmpeg installed and accessible in PATH
- OpenCV (cv2)
- PIL (Pillow)
- moviepy
"""

import subprocess
import os
import sys
import tempfile
from pathlib import Path
import time
import cv2
from PIL import Image
import numpy as np
from typing import List, Tuple, Optional, Union


def check_ffmpeg() -> bool:
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print("Error: FFmpeg is not properly installed or not in PATH.")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
            return False
        return True
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html")
        return False


def get_video_duration(video_path: str) -> Optional[float]:
    """Get the duration of a video file using FFmpeg."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error getting video duration: {result.stderr}")
            return None
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return None


def get_video_info(video_path: str) -> dict:
    """Get video information like duration, fps, resolution."""
    info = {}
    try:
        cap = cv2.VideoCapture(video_path)
        info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['duration'] = info['frame_count'] / info['fps']
        cap.release()
        return info
    except Exception as e:
        print(f"Error getting video info: {str(e)}")
        return {'error': str(e)}


def video_loop(input_path: str, output_path: str, fade_duration: float = 1.0) -> bool:
    """
    Create a seamless looped video with overlay blending.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the output looped video
        fade_duration: Duration of the fade transition in seconds
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    duration = get_video_duration(input_path)
    if duration is None:
        return False

    if fade_duration >= duration / 2:
        print(f"Error: Fade duration ({fade_duration}s) too long for video ({duration}s)")
        return False

    try:
        # Base filter chain - extract segments
        base_filter = (
            f"[0:v]split=3[main][end][begin];"
            f"[main]trim=start={fade_duration}:end={duration-fade_duration},setpts=PTS-STARTPTS[middle];"
            f"[end]trim=start={duration-fade_duration},setpts=PTS-STARTPTS[end_clip];"
            f"[begin]trim=0:{fade_duration},setpts=PTS-STARTPTS[begin_clip];"
        )
        
        # Standard overlay method
        transition_filter = (
            f"[begin_clip]format=yuva420p,fade=t=in:st=0:d={fade_duration}:alpha=1[overlay_fade];"
            f"[end_clip][overlay_fade]overlay=(main_w-overlay_w)/2:(main_h-overlay_h)/2:format=auto,"
            f"trim=0:{fade_duration}[transition];"
        )

        # Finalize filter with concatenation
        final_concat = f"[middle][transition]concat=n=2:v=1:a=0"
        
        # Complete filter chain
        filter_complex = base_filter + transition_filter + final_concat

        # Run FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", filter_complex,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-x264-params", "keyint=15:min-keyint=15:scenecut=0",
            "-an", output_path
        ]

        print(f"Creating seamless loop...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error creating loop: {result.stderr}")
            return False

        print(f"✅ Seamless loop created: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating loop: {str(e)}")
        return False


def create_extended_loop(input_path: str, output_path: str, repeats: int = 2) -> bool:
    """
    Create a longer video by seamlessly repeating the input video multiple times.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the extended video
        repeats: Number of times to repeat the video
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    try:
        # Convert to absolute path to avoid issues with temporary directory
        abs_input_path = os.path.abspath(input_path)
        
        # Create a temporary file for the concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_list:
            concat_list_path = concat_list.name
            
            # Create the concat file with multiple entries of the same video
            for _ in range(repeats):
                concat_list.write(f"file '{abs_input_path}'\n")
            concat_list.flush()
            
            # Concatenate the video with itself
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-c", "copy",
                output_path
            ]
            
            print(f"Creating extended loop ({repeats}x)...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Clean up temp file
            os.unlink(concat_list_path)
            
            if result.returncode != 0:
                print(f"Error creating extended loop: {result.stderr}")
                return False
            
            print(f"✅ Extended loop created: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error creating extended loop: {str(e)}")
        return False


def video_speedup(input_path: str, output_path: str, speed_factor: float = 4.0) -> bool:
    """
    Speed up a video by the specified factor.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the output sped-up video
        speed_factor: Factor by which to speed up the video (e.g., 2.0 = twice as fast)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    try:
        # FFmpeg command to speed up video using the setpts filter
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter:v", f"setpts=PTS/{speed_factor}",
            "-an" if speed_factor > 2.0 else f"-filter:a", f"atempo={min(2.0, speed_factor)}" if speed_factor <= 2.0 else "-an",
            output_path
        ]
        
        # For audio, atempo only works in range 0.5-2.0, so for higher speeds we just remove audio
        
        print(f"Speeding up video by factor of {speed_factor}...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error speeding up video: {result.stderr}")
            return False
        
        print(f"✅ Video speed increased: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error speeding up video: {str(e)}")
        return False


def video_to_frames(
    input_path: str, 
    output_folder: str, 
    frame_rate: int = 1, 
    rotation_angle: int = 0
) -> Tuple[bool, int]:
    """
    Extract frames from a video file as PNG images.
    
    Args:
        input_path: Path to the input video
        output_folder: Directory to save extracted frames
        frame_rate: Extract every nth frame
        rotation_angle: Rotation angle (0, 90, 180, 270)
    
    Returns:
        Tuple[bool, int]: Success status and number of frames extracted
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Open video file
        cap = cv2.VideoCapture(input_path)
        frame_count = 0
        saved_count = 0
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every nth frame based on frame_rate
            if frame_count % frame_rate == 0:
                # Apply rotation if needed
                rotated_frame = frame.copy()
                if rotation_angle == 90:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_angle == 180:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation_angle == 270:
                    rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                # Save frame as PNG with proper color format (BGR in OpenCV)
                output_path = os.path.join(output_folder, f"frame_{saved_count:05d}.png")
                cv2.imwrite(output_path, rotated_frame)
                
                saved_count += 1
            
            frame_count += 1
            
            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, saved {saved_count}")
        
        cap.release()
        
        print(f"✅ Extracted {saved_count} frames to {output_folder}")
        return True, saved_count
        
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return False, 0


def video_to_gif(
    input_path: str, 
    output_path: str, 
    fps: int = 10, 
    resize_factor: float = 100.0,
    quality: int = 85,
    optimize_size: bool = True,
    sample_every: int = 1,
    max_colors: int = 256
) -> bool:
    """
    Convert a video file to an animated GIF with color preservation and size optimization.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the output GIF
        fps: Frames per second in the output GIF
        resize_factor: Resize percentage (e.g., 50 = half size, 100 = original size)
        quality: Quality of the output GIF (1-100, higher is better)
        optimize_size: Whether to apply additional optimization to reduce file size
        sample_every: Sample every Nth frame (higher values = smaller file size)
        max_colors: Maximum number of colors to use (2-256, lower values = smaller file size)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Converting video to GIF: {input_path}")
        print(f"Sampling every {sample_every} frame(s) at {fps} FPS")
        
        # Open video file
        video = cv2.VideoCapture(input_path)
        
        # Get video info
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate resize percentage
        resize_factor = float(resize_factor) / 100
        
        if resize_factor != 1:
            new_width = int(width * resize_factor)
            new_height = int(height * resize_factor)
            print(f"Resizing from {width}x{height} to {new_width}x{new_height}")
        else:
            print(f"Maintaining original resolution: {width}x{height}")
        
        # Limit max_colors to valid range
        max_colors = max(2, min(256, max_colors))
        
        # Process frames
        frames = []
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            # Sample every Nth frame
            if frame_count % sample_every == 0:
                # Convert BGR to RGB with proper color handling
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                # Resize if needed
                if resize_factor != 1:
                    new_size = (int(pil_frame.width * resize_factor), 
                              int(pil_frame.height * resize_factor))
                    pil_frame = pil_frame.resize(new_size, Image.LANCZOS)
                
                # Apply adaptive palette if we're restricting colors
                if max_colors < 256:
                    pil_frame = pil_frame.convert('P', palette=Image.ADAPTIVE, colors=max_colors)
                
                frames.append(pil_frame)
                saved_count += 1
                
                # Print progress every 20 frames
                if saved_count % 20 == 0:
                    print(f"Processed {frame_count+1} frames, saved {saved_count}")
            
            frame_count += 1
        
        video.release()
        
        # Save as GIF
        if frames:
            # Calculate number of frames being saved vs. total
            print(f"Saving GIF with {len(frames)} frames ({len(frames)/frame_count*100:.1f}% of original)")
            
            # Calculate appropriate duration for the desired FPS
            duration = int(1000 / fps)  # milliseconds per frame
            
            # Save with optimizations
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=optimize_size,
                quality=quality,
                duration=duration,
                loop=0  # 0 = loop forever
            )
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"✅ GIF created successfully: {output_path} ({file_size:.2f} MB)")
            return True
        else:
            print("No frames were extracted from the video.")
            return False
        
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")
        return False


def video_optimise(
    input_path: str,
    output_path: str,
    resolution: Optional[Tuple[int, int]] = None,
    crf: int = 28,
    preset: str = "medium",
    codec: str = "libx264"
) -> bool:
    """
    Optimize a video file to reduce its size while maintaining perceptual quality.
    
    Args:
        input_path: Path to the input video
        output_path: Path for the optimized video
        resolution: Output resolution as (width, height), or None to keep original
        crf: Constant Rate Factor (0-51, lower is better quality, 18-28 is good)
        preset: x264 preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        codec: Video codec to use (libx264, libx265, etc.)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    try:
        # Get input file size for comparison
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        
        # Start building the command
        cmd = ["ffmpeg", "-y", "-i", input_path]
        
        # Add scale filter if resolution is specified
        if resolution:
            width, height = resolution
            cmd.extend(["-vf", f"scale={width}:{height}"])
        
        # Add codec and quality parameters
        cmd.extend([
            "-c:v", codec,
            "-crf", str(crf),
            "-preset", preset,
            # Maintain audio quality
            "-c:a", "aac",
            "-b:a", "128k",
            output_path
        ])
        
        print(f"Optimizing video...")
        print(f"Input size: {input_size:.2f} MB")
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error optimizing video: {result.stderr}")
            return False
        
        # Get output file size for comparison
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✅ Video optimized: {output_path}")
        print(f"Original size: {input_size:.2f} MB")
        print(f"Optimized size: {output_size:.2f} MB")
        print(f"Size reduction: {(1 - output_size/input_size) * 100:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Error optimizing video: {str(e)}")
        return False


def convert_mov_to_mp4(
    input_path: str,
    output_path: str,
    quality: str = "high",
    preserve_audio: bool = True
) -> bool:
    """
    Convert a MOV file to MP4 format with configurable quality.
    
    Args:
        input_path: Path to the input MOV file
        output_path: Path for the output MP4 file
        quality: Quality preset ('high', 'medium', 'low')
        preserve_audio: Whether to include audio in the output file
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    if not input_path.lower().endswith('.mov'):
        print(f"Warning: Input file does not appear to be a MOV file: {input_path}")
    
    # Define quality presets
    quality_presets = {
        "high": {"crf": "18", "preset": "slow"},
        "medium": {"crf": "23", "preset": "medium"},
        "low": {"crf": "28", "preset": "fast"}
    }
    
    # Use medium quality if the specified quality is not in presets
    preset_config = quality_presets.get(quality.lower(), quality_presets["medium"])
    
    try:
        # Build FFmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-crf", preset_config["crf"],
            "-preset", preset_config["preset"],
            "-pix_fmt", "yuv420p"  # Ensure compatibility
        ]
        
        # Handle audio settings
        if preserve_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.append("-an")
        
        # Add output path
        cmd.append(output_path)
        
        # Get input file size for comparison
        input_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        
        print(f"Converting MOV to MP4...")
        print(f"Input: {input_path} ({input_size:.2f} MB)")
        print(f"Quality: {quality} (CRF: {preset_config['crf']}, Preset: {preset_config['preset']})")
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print(f"Error converting MOV to MP4: {result.stderr}")
            return False
        
        # Get output file size for comparison
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        print(f"✅ MOV to MP4 conversion successful: {output_path}")
        print(f"Original size: {input_size:.2f} MB")
        print(f"Converted size: {output_size:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"Error converting MOV to MP4: {str(e)}")
        return False


if __name__ == "__main__":
    import time
    from pathlib import Path
    
    def prompt(message, default=None):
        """Helper function to get user input with optional default value."""
        if default:
            result = input(f"{message} [{default}]: ")
            return result if result.strip() else default
        return input(f"{message}: ")

    def setup_test_paths():
        """Setup default test paths and directories."""
        # Default test video from video_loop.py
        input_path = Path("data/output/test_videos/test_wan-i2v-480p.mp4")
        
        # Try to find alternative test videos if the default doesn't exist
        if not input_path.exists():
            print(f"Default test video not found: {input_path}")
            
            # Check in current directory and subdirectories
            alternatives = list(Path(".").glob("**/*.mp4"))
            if alternatives:
                input_path = alternatives[0]
                print(f"Using alternative video: {input_path}")
            else:
                # Ask user to provide a path
                user_path = prompt("Please enter a path to a test video (MP4 file)")
                input_path = Path(user_path)
                if not input_path.exists():
                    print(f"Error: Video file not found: {input_path}")
                    return None, None
        
        # Create output directory
        output_dir = Path("data/output/video_utils_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return str(input_path), str(output_dir)

    def video_loop_test(input_path, output_dir):
        """Test the video_loop function."""
        print("\n=== Testing Video Loop ===")
        output_path = os.path.join(output_dir, "loop.mp4")
        
        # Get video info
        info = get_video_info(input_path)
        print(f"Video info: {info}")
        
        # Ask for fade duration
        fade_duration = float(prompt("Fade duration (seconds)", "1.0"))
        
        print(f"Creating seamless loop from: {input_path}")
        
        # Create video loop
        success = video_loop(
            input_path=input_path,
            output_path=output_path,
            fade_duration=fade_duration
        )
        
        if success:
            print(f"Video loop created: {output_path}")
            # Offer to create extended loop
            if prompt("Create extended loop? (y/n)", "y").lower() == "y":
                extended_path = os.path.join(output_dir, "loop_extended.mp4")
                repeats = int(prompt("Number of repeats", "2"))
                
                extended_success = create_extended_loop(
                    input_path=output_path,
                    output_path=extended_path,
                    repeats=repeats
                )
                
                if extended_success:
                    print(f"Extended loop created: {extended_path}")
        
        return success

    def video_speedup_test(input_path, output_dir):
        """Test the video_speedup function."""
        print("\n=== Testing Video Speed-up ===")
        output_path = os.path.join(output_dir, "speedup.mp4")
        
        # Ask for speed factor
        speed_factor = float(prompt("Speed factor (e.g., 2.0 = twice as fast)", "2.0"))
        
        print(f"Speeding up video: {input_path}")
        
        # Speed up video
        success = video_speedup(
            input_path=input_path,
            output_path=output_path,
            speed_factor=speed_factor
        )
        
        if success:
            print(f"Speed-up video created: {output_path}")
        
        return success

    def video_to_frames_test(input_path, output_dir):
        """Test the video_to_frames function."""
        print("\n=== Testing Video to Frames ===")
        frames_dir = os.path.join(output_dir, "frames")
        
        # Create frames directory
        os.makedirs(frames_dir, exist_ok=True)
        
        # Ask for frame rate and rotation
        frame_rate = int(prompt("Extract every Nth frame", "10"))
        rotation_options = {
            "0": "No rotation",
            "90": "Rotate 90° clockwise",
            "180": "Rotate 180°",
            "270": "Rotate 90° counter-clockwise"
        }
        
        print("Rotation options:")
        for key, value in rotation_options.items():
            print(f"{key}: {value}")
        
        rotation = int(prompt("Rotation angle", "0"))
        
        print(f"Extracting frames from: {input_path}")
        
        # Extract frames
        success, num_frames = video_to_frames(
            input_path=input_path,
            output_folder=frames_dir,
            frame_rate=frame_rate,
            rotation_angle=rotation
        )
        
        if success:
            print(f"Extracted {num_frames} frames to: {frames_dir}")
        
        return success, frames_dir

    def video_to_gif_test(input_path, output_dir):
        """Test the video_to_gif function."""
        print("\n=== Testing Video to GIF ===")
        output_path = os.path.join(output_dir, "output.gif")
        
        # Get video info for context
        info = get_video_info(input_path)
        print(f"Video info: {info}")
        
        # Ask for fps and resize factor
        fps = int(prompt("Frames per second", "10"))
        resize_factor = float(prompt("Resize factor (percentage, e.g., 50 = half size, 100 = original)", "100"))
        
        # Ask for compression options
        print("\nCompression options:")
        sample_every = int(prompt("Sample every N frames (higher = smaller file)", "1"))
        quality = int(prompt("Quality (1-100, higher = better quality)", "85"))
        max_colors = int(prompt("Max colors (2-256, lower = smaller file)", "256"))
        
        # Convert to GIF
        success = video_to_gif(
            input_path=input_path,
            output_path=output_path,
            fps=fps,
            resize_factor=resize_factor,
            quality=quality,
            optimize_size=True,
            sample_every=sample_every,
            max_colors=max_colors
        )
        
        if success:
            print(f"GIF created: {output_path}")
        
        return success

    def video_optimise_test(input_path, output_dir):
        """Test the video_optimise function."""
        print("\n=== Testing Video Optimisation ===")
        output_path = os.path.join(output_dir, "optimised.mp4")
        
        # Get original video info
        info = get_video_info(input_path)
        original_width = info.get('width', 0)
        original_height = info.get('height', 0)
        
        print(f"Original resolution: {original_width}x{original_height}")
        
        # Ask for optimization parameters
        change_resolution = prompt("Change resolution? (y/n)", "n").lower() == "y"
        resolution = None
        
        if change_resolution:
            new_width = int(prompt("New width", str(original_width // 2)))
            new_height = int(prompt("New height", str(original_height // 2)))
            resolution = (new_width, new_height)
        
        # Quality settings
        crf = int(prompt("CRF (Constant Rate Factor, 0-51, lower is better quality)", "23"))
        
        preset_options = {
            "ultrafast": "Very fast encoding, largest file size",
            "superfast": "Super fast encoding",
            "veryfast": "Very fast encoding",
            "faster": "Faster encoding",
            "fast": "Fast encoding",
            "medium": "Default preset, balanced speed/quality",
            "slow": "Slower encoding, better quality",
            "slower": "Even slower encoding",
            "veryslow": "Very slow encoding, best quality"
        }
        
        print("Preset options:")
        for key, value in preset_options.items():
            print(f"{key}: {value}")
        
        preset = prompt("Preset", "medium")
        
        codec_options = {
            "libx264": "H.264, good compatibility",
            "libx265": "H.265/HEVC, better compression but less compatible"
        }
        
        print("Codec options:")
        for key, value in codec_options.items():
            print(f"{key}: {value}")
        
        codec = prompt("Codec", "libx264")
        
        print(f"Optimizing video: {input_path}")
        
        # Optimize video
        success = video_optimise(
            input_path=input_path,
            output_path=output_path,
            resolution=resolution,
            crf=crf,
            preset=preset,
            codec=codec
        )
        
        if success:
            print(f"Optimized video created: {output_path}")
        
        return success
    
    def convert_mov_to_mp4_test(input_path, output_dir):
        """Test the convert_mov_to_mp4 function."""
        print("\n=== Testing MOV to MP4 Conversion ===")
        
        # Check if input is MOV file or ask for one
        if not input_path.lower().endswith('.mov'):
            print("Current input is not a MOV file.")
            mov_path = prompt("Please enter a path to a MOV file")
            input_path = mov_path
        
        output_path = os.path.join(output_dir, "converted.mp4")
        
        # Ask for quality settings
        quality_options = {
            "high": "High quality, larger file size",
            "medium": "Medium quality, balanced file size (default)",
            "low": "Lower quality, smaller file size"
        }
        
        print("Quality options:")
        for key, value in quality_options.items():
            print(f"{key}: {value}")
        
        quality = prompt("Quality", "medium")
        preserve_audio = prompt("Preserve audio? (y/n)", "y").lower() == "y"
        
        # Convert MOV to MP4
        success = convert_mov_to_mp4(
            input_path=input_path,
            output_path=output_path,
            quality=quality,
            preserve_audio=preserve_audio
        )
        
        if success:
            print(f"MOV to MP4 conversion completed: {output_path}")
        
        return success

    # Test menu
    print("\n" + "="*60)
    print("📹 VIDEO UTILS TEST SUITE")
    print("="*60)
    
    # Setup test paths
    input_path, output_dir = setup_test_paths()
    if not input_path or not output_dir:
        print("Error setting up test paths. Exiting.")
        sys.exit(1)
    
    # Define test functions dictionary
    test_functions = {
        "1": ("Video Loop", lambda: video_loop_test(input_path, output_dir)),
        "2": ("Video Speed-up", lambda: video_speedup_test(input_path, output_dir)),
        "3": ("Video to Frames", lambda: video_to_frames_test(input_path, output_dir)),
        "4": ("Video to GIF", lambda: video_to_gif_test(input_path, output_dir)),
        "5": ("Video Optimisation", lambda: video_optimise_test(input_path, output_dir)),
        "6": ("Display Video Info", lambda: print(f"Video info: {get_video_info(input_path)}")),
        "7": ("MOV to MP4 Conversion", lambda: convert_mov_to_mp4_test(input_path, output_dir))
    }
    
    # Main test loop
    while True:
        print("\nAvailable Tests:")
        for key, (name, _) in test_functions.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        choice = prompt("\nSelect a test to run", "0")
        
        if choice == "0":
            print("\nExiting test suite.")
            break
        elif choice in test_functions:
            try:
                name, test_func = test_functions[choice]
                print(f"\nRunning test: {name}")
                test_func()
            except Exception as e:
                print(f"\n❌ Error running test: {str(e)}")
                import traceback
                traceback.print_exc()
            
            input("\nPress Enter to continue...")
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE COMPLETED")
    print("="*60) 