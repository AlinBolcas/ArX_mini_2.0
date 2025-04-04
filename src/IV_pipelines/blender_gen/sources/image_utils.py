#!/usr/bin/env python3
"""
Image Utilities

A collection of image and GIF manipulation utilities for:
- Converting images to GIFs
- Optimizing GIFs
- Creating videos from image sequences
- Optimizing/resizing images
- Converting image formats

Requirements:
- PIL (Pillow)
- FFmpeg (for video creation)
- OpenCV (cv2) for advanced image processing
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import cv2
from PIL import Image, ImageSequence
import numpy as np
from typing import List, Tuple, Optional, Union, Dict


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


def get_image_info(image_path: str) -> Dict:
    """Get basic information about an image file."""
    try:
        with Image.open(image_path) as img:
            info = {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'file_size': os.path.getsize(image_path) / 1024,  # KB
                'path': image_path
            }
            return info
    except Exception as e:
        print(f"Error getting image info for {image_path}: {str(e)}")
        return {'error': str(e), 'path': image_path}


def images_to_gif(
    input_paths: List[str],
    output_path: str,
    duration: float = 0.1,
    resize_factor: float = 100.0,
    optimize: bool = True,
    loop: int = 0,
    max_colors: int = 256
) -> bool:
    """
    Create an animated GIF from a list of images.
    
    Args:
        input_paths: List of paths to input images
        output_path: Path for the output GIF
        duration: Duration of each frame in seconds
        resize_factor: Resize percentage (100 = original size)
        optimize: Whether to optimize the GIF
        loop: Number of loops (0 = infinite)
        max_colors: Maximum number of colors to use (2-256)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not input_paths:
            print("Error: No input images provided.")
            return False
            
        print(f"Creating GIF from {len(input_paths)} images...")
        
        # Calculate resize percentage
        resize_factor = float(resize_factor) / 100
        
        # Load and potentially resize images
        frames = []
        for i, path in enumerate(input_paths):
            try:
                # Print progress for every 10th image or first/last
                if i % 10 == 0 or i == len(input_paths) - 1:
                    print(f"Processing image {i+1}/{len(input_paths)}: {path}")
                
                # Open and convert to RGBA to preserve transparency if present
                img = Image.open(path).convert('RGBA')
                
                # Resize if needed
                if resize_factor != 1:
                    new_size = (int(img.width * resize_factor), int(img.height * resize_factor))
                    img = img.resize(new_size, Image.LANCZOS)
                
                frames.append(img)
            except Exception as e:
                print(f"Warning: Couldn't process {path}: {str(e)}")
        
        if not frames:
            print("Error: No valid frames found.")
            return False
        
        # Convert duration to milliseconds
        duration_ms = int(duration * 1000)
        
        # Prepare palette mode for better color handling if requested
        if max_colors < 256:
            # Convert frames to indexed color mode (palette) with adaptive palette
            for i, frame in enumerate(frames):
                frames[i] = frame.convert('P', palette=Image.ADAPTIVE, colors=max_colors)
        
        # Save the GIF
        print(f"Saving GIF with {len(frames)} frames...")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=optimize,
            duration=duration_ms,
            loop=loop
        )
        
        # Report final size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ GIF created successfully: {output_path} ({file_size:.2f} MB)")
        return True
        
    except Exception as e:
        print(f"Error creating GIF: {str(e)}")
        return False


def gif_optimise(
    input_path: str,
    output_path: str,
    color_reduction: int = 128,
    scale_factor: float = 75.0,
    speed_factor: float = 100.0,
    optimize: bool = True
) -> bool:
    """
    Optimize a GIF by reducing colors, size, and/or speed.
    
    Args:
        input_path: Path to input GIF
        output_path: Path for optimized GIF
        color_reduction: Number of colors to use (2-256)
        scale_factor: Resize percentage (100 = original size)
        speed_factor: Speed percentage (100 = original speed, 50 = half speed, 200 = double speed)
        optimize: Use PIL's built-in optimization
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Optimizing GIF: {input_path}")
        
        # Calculate resize factor
        scale_factor = float(scale_factor) / 100
        
        # Calculate speed factor (inverse of duration factor)
        duration_factor = 100.0 / speed_factor
        
        # Open the GIF
        with Image.open(input_path) as img:
            # Get original size and other info
            original_width, original_height = img.size
            frame_count = 0
            for frame in ImageSequence.Iterator(img):
                frame_count += 1
            
            # Get original duration
            try:
                original_duration = img.info.get('duration', 100)  # Default to 100ms if not specified
            except:
                original_duration = 100
                
            print(f"Original: {frame_count} frames, {original_width}x{original_height}, {original_duration}ms/frame")
            
            # Process frames
            frames = []
            durations = []
            
            for frame in ImageSequence.Iterator(img):
                # Color reduction (convert to palette mode)
                frame = frame.convert("P", palette=Image.ADAPTIVE, colors=color_reduction)
                
                # Resize if needed
                if scale_factor != 1.0:
                    new_size = (int(frame.width * scale_factor), int(frame.height * scale_factor))
                    frame = frame.resize(new_size, Image.LANCZOS)
                    
                frames.append(frame)
                
                # Adjust frame duration
                frame_duration = int(original_duration * duration_factor)
                durations.append(frame_duration)
            
            # New dimensions
            new_width, new_height = frames[0].size
            
            # Save optimized GIF
            print(f"Saving optimized GIF with {len(frames)} frames...")
            print(f"New size: {new_width}x{new_height} ({scale_factor*100:.0f}% of original)")
            print(f"Colors: {color_reduction}")
            print(f"Speed: {speed_factor:.0f}% of original")
            
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                optimize=optimize,
                loop=0,  # Loop forever
                duration=durations
            )
            
            # Report file size reduction
            original_size = os.path.getsize(input_path) / 1024  # KB
            new_size = os.path.getsize(output_path) / 1024  # KB
            
            print(f"✅ GIF optimized successfully: {output_path}")
            print(f"Original size: {original_size:.1f} KB")
            print(f"Optimized size: {new_size:.1f} KB")
            print(f"Size reduction: {(1 - new_size/original_size) * 100:.1f}%")
            
            return True
            
    except Exception as e:
        print(f"Error optimizing GIF: {str(e)}")
        return False


def image_to_video(
    frame_paths: List[str],
    output_path: str,
    fps: int = 24,
    resolution: Optional[Tuple[int, int]] = None,
    crf: int = 20,
    preset: str = "medium"
) -> bool:
    """
    Create a video from a list of image frames.
    
    Args:
        frame_paths: List of paths to image frames
        output_path: Path for the output video
        fps: Frames per second for the output video
        resolution: Output resolution as (width, height), or None to use first image's resolution
        crf: Constant Rate Factor (0-51, lower is better quality, 18-28 is good)
        preset: x264 preset (e.g., medium, slow)
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not check_ffmpeg():
        return False
        
    if not frame_paths:
        print("Error: No frames provided")
        return False
        
    try:
        print(f"Creating video from {len(frame_paths)} images...")
        
        # Create a temporary file for the concat list
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as concat_list:
            concat_list_path = concat_list.name
            
            # Determine resolution if not specified
            if resolution is None:
                # Try different methods to get image size
                first_img = None
                try:
                    # Try PIL first
                    first_img = Image.open(frame_paths[0])
                    width, height = first_img.size
                except:
                    try:
                        # Try OpenCV if PIL fails
                        first_img = cv2.imread(frame_paths[0])
                        if first_img is None:
                            raise ValueError(f"Could not read image at {frame_paths[0]}")
                        height, width = first_img.shape[:2]
                    except:
                        print(f"Error: Could not determine image dimensions for {frame_paths[0]}")
                        return False
                
                resolution = (width, height)
                print(f"Using resolution from first image: {width}x{height}")
            else:
                width, height = resolution
                print(f"Using specified resolution: {width}x{height}")
            
            # Write the list of frames to the concat file
            for frame_path in frame_paths:
                # Make sure paths are absolute and escape spaces
                abs_path = os.path.abspath(frame_path).replace('\\', '/')
                # Each image needs to be listed with a duration to match the desired FPS
                concat_list.write(f"file '{abs_path}'\n")
                concat_list.write(f"duration {1/fps}\n")
            
            # Add the last frame again (required by the format)
            # Fix: Pre-format the path outside of the f-string to avoid backslash issues
            last_frame_path = os.path.abspath(frame_paths[-1]).replace('\\', '/')
            concat_list.write(f"file '{last_frame_path}'\n")
            concat_list.flush()
            
            # Create video using FFmpeg
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_list_path,
                "-vsync", "vfr",
                "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", str(crf),
                "-preset", preset,
                output_path
            ]
            
            print(f"Running FFmpeg command...")
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Clean up temp file
            os.unlink(concat_list_path)
            
            if result.returncode != 0:
                print(f"Error creating video: {result.stderr}")
                return False
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f"✅ Video created: {output_path} ({file_size:.2f} MB)")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps}")
            print(f"Duration: {len(frame_paths)/fps:.2f} seconds")
            
            return True
            
    except Exception as e:
        print(f"Error creating video from frames: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def images_optimise(
    input_paths: List[str],
    output_dir: Optional[str] = None,
    output_format: str = "jpg",
    scale_factor: float = 75.0,
    quality: int = 80,
    prefix: str = "",
    suffix: str = "_optimized"
) -> List[str]:
    """
    Optimize and resize a list of images.
    
    Args:
        input_paths: List of paths to input images
        output_dir: Directory to save optimized images (if None, use same dir as input)
        output_format: Output format (jpg, png, webp, etc.)
        scale_factor: Resize percentage (100 = original size)
        quality: JPEG/WebP quality (0-100, higher is better quality)
        prefix: Prefix to add to output filenames
        suffix: Suffix to add to output filenames before extension
        
    Returns:
        List[str]: List of paths to the optimized images
    """
    try:
        if not input_paths:
            print("Error: No input images provided.")
            return []
            
        print(f"Optimizing {len(input_paths)} images...")
        
        # Calculate resize percentage
        scale_factor = float(scale_factor) / 100
        
        # Setup output paths
        output_paths = []
        successful_count = 0
        
        # Process images
        for i, input_path in enumerate(input_paths):
            try:
                # Create output path
                input_file = Path(input_path)
                
                if output_dir:
                    # Use specified output directory
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir_path / f"{prefix}{input_file.stem}{suffix}.{output_format.lower()}"
                else:
                    # Use same directory as input
                    output_file = input_file.parent / f"{prefix}{input_file.stem}{suffix}.{output_format.lower()}"
                
                output_path = str(output_file)
                
                # Print progress for every 10th image or first/last
                if i % 10 == 0 or i == len(input_paths) - 1:
                    print(f"Processing {i+1}/{len(input_paths)}: {input_path}")
                
                # Open image
                with Image.open(input_path) as img:
                    # Get original size
                    original_width, original_height = img.width, img.height
                    original_size = os.path.getsize(input_path) / 1024  # KB
                    
                    # Convert to RGB if image has alpha channel or is in palette mode
                    # unless output format supports alpha
                    if output_format.lower() in ('jpg', 'jpeg'):
                        if img.mode == 'RGBA' or img.mode == 'P':
                            # Create white background for transparent images when converting to JPEG
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                                img = background
                            else:
                                img = img.convert('RGB')
                    
                    # Resize if needed
                    if scale_factor != 1:
                        new_width = int(img.width * scale_factor)
                        new_height = int(img.height * scale_factor)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Determine save parameters based on format
                    save_kwargs = {}
                    if output_format.lower() in ('jpg', 'jpeg'):
                        save_kwargs = {'quality': quality, 'optimize': True}
                    elif output_format.lower() == 'png':
                        save_kwargs = {'optimize': True}
                    elif output_format.lower() == 'webp':
                        save_kwargs = {'quality': quality, 'method': 6}  # Higher method = better compression
                    
                    # Save optimized image
                    img.save(output_path, **save_kwargs)
                    
                    # Get optimized size
                    optimized_size = os.path.getsize(output_path) / 1024  # KB
                    
                    # Print detailed info for every 10th image or for large compressions
                    if i % 10 == 0 or (original_size > 0 and optimized_size / original_size < 0.5):
                        print(f"  Original: {original_width}x{original_height}, {original_size:.1f} KB")
                        print(f"  Optimized: {new_width if scale_factor != 1 else original_width}x"
                              f"{new_height if scale_factor != 1 else original_height}, "
                              f"{optimized_size:.1f} KB ({(optimized_size/original_size*100):.1f}%)")
                    
                    output_paths.append(output_path)
                    successful_count += 1
                    
            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
        
        print(f"✅ Successfully optimized {successful_count}/{len(input_paths)} images")
        return output_paths
        
    except Exception as e:
        print(f"Error in images_optimise: {str(e)}")
        return []


def images_to_format(
    input_paths: List[str],
    output_format: str,
    output_dir: Optional[str] = None,
    quality: int = 95,
    preserve_size: bool = True,
    prefix: str = "",
    suffix: str = ""
) -> List[str]:
    """
    Convert images to a different format.
    
    Args:
        input_paths: List of paths to input images
        output_format: Target format (jpg, png, webp, etc.)
        output_dir: Directory to save converted images (if None, use same dir as input)
        quality: Quality setting for lossy formats (0-100)
        preserve_size: Whether to maintain the original dimensions
        prefix: Prefix to add to output filenames
        suffix: Suffix to add to output filenames before extension
        
    Returns:
        List[str]: List of paths to the converted images
    """
    try:
        if not input_paths:
            print("Error: No input images provided.")
            return []
            
        # Normalize output format
        output_format = output_format.lower().strip('.')
        
        print(f"Converting {len(input_paths)} images to {output_format.upper()}...")
        
        # Setup output paths
        output_paths = []
        successful_count = 0
        
        # Process images
        for i, input_path in enumerate(input_paths):
            try:
                # Create output path
                input_file = Path(input_path)
                
                if output_dir:
                    # Use specified output directory
                    output_dir_path = Path(output_dir)
                    output_dir_path.mkdir(parents=True, exist_ok=True)
                    output_file = output_dir_path / f"{prefix}{input_file.stem}{suffix}.{output_format}"
                else:
                    # Use same directory as input
                    output_file = input_file.parent / f"{prefix}{input_file.stem}{suffix}.{output_format}"
                
                output_path = str(output_file)
                
                # Print progress for every 10th image or first/last
                if i % 10 == 0 or i == len(input_paths) - 1:
                    print(f"Converting {i+1}/{len(input_paths)}: {input_path}")
                
                # Open image
                with Image.open(input_path) as img:
                    # Handle format-specific conversions
                    if output_format in ('jpg', 'jpeg'):
                        # For JPG conversion, ensure RGB mode and handle transparency
                        if img.mode in ('RGBA', 'LA'):
                            # White background for transparent images
                            background = Image.new('RGB', img.size, (255, 255, 255))
                            if img.mode == 'RGBA':
                                background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                            else:
                                background.paste(img, mask=img.split()[1])  # Use alpha channel as mask
                            img = background
                        elif img.mode != 'RGB':
                            img = img.convert('RGB')
                            
                        # Save with quality setting
                        img.save(output_path, format='JPEG', quality=quality, optimize=True)
                        
                    elif output_format == 'png':
                        # PNG supports all modes, just save it
                        img.save(output_path, format='PNG', optimize=True)
                        
                    elif output_format == 'webp':
                        # WebP with quality setting
                        img.save(output_path, format='WEBP', quality=quality, method=6)
                        
                    elif output_format == 'gif':
                        # If source has transparency, preserve it
                        if img.mode == 'RGBA':
                            # Convert to P mode with transparency
                            img = img.convert('P', palette=Image.ADAPTIVE, colors=255)
                        else:
                            img = img.convert('P', palette=Image.ADAPTIVE)
                        
                        img.save(output_path, format='GIF')
                        
                    else:
                        # For other formats, use PIL's built-in conversion
                        img.save(output_path, format=output_format.upper())
                    
                    print(f"  Converted: {input_file.name} → {output_file.name}")
                    
                    output_paths.append(output_path)
                    successful_count += 1
                    
            except Exception as e:
                print(f"Error converting {input_path}: {str(e)}")
        
        print(f"✅ Successfully converted {successful_count}/{len(input_paths)} images to {output_format.upper()}")
        return output_paths
        
    except Exception as e:
        print(f"Error in images_to_format: {str(e)}")
        return []


if __name__ == "__main__":
    def prompt(message, default=None):
        """Helper function to get user input with optional default value."""
        if default:
            result = input(f"{message} [{default}]: ")
            return result if result.strip() else default
        return input(f"{message}: ")
    
    def get_test_images(frames_dir=None):
        """Find test images in the specified directory or alternatives."""
        if frames_dir is None:
            # Default test frames path
            frames_dir = "/Users/arvolve/Coding/ArX_mini_2.0/ArX_mini_2.0/data/output/video_utils_test/frames"
            
            # Fallback to relative path if absolute path doesn't exist
            if not os.path.exists(frames_dir):
                frames_dir = "data/output/video_utils_test/frames"
        
        # Check if directory exists
        if not os.path.exists(frames_dir):
            print(f"Warning: Test frames directory not found: {frames_dir}")
            return []
        
        # Find image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
        image_paths = [
            os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))
            if f.lower().endswith(image_extensions)
        ]
        
        if image_paths:
            print(f"Found {len(image_paths)} test images in {frames_dir}")
        else:
            print(f"No test images found in {frames_dir}")
        
        return image_paths
    
    # Test/demo menu
    print("\n" + "="*60)
    print("🖼️ IMAGE UTILS TEST SUITE")
    print("="*60)
    
    # Setup default directories and test images
    output_dir = Path("data/output/image_utils_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get test images
    test_images = get_test_images()
    test_images_str = ','.join(test_images) if test_images else ""
    
    # If we have test images, show some details
    if test_images:
        print(f"Test images available: {len(test_images)}")
        print(f"First image: {os.path.basename(test_images[0])}")
        print(f"Sample image info:")
        print(get_image_info(test_images[0]))
    
    # Define test functions dictionary
    test_functions = {
        "1": ("Images to GIF", lambda: images_to_gif(
            input_paths=prompt("Enter image paths (comma-separated)", test_images_str).split(','),
            output_path=str(output_dir / "test_output.gif"),
            duration=float(prompt("Duration per frame (seconds)", "0.1")),
            resize_factor=float(prompt("Resize factor (%)", "100")),
            max_colors=int(prompt("Max colors (2-256)", "256"))
        )),
        "2": ("Optimize GIF", lambda: gif_optimise(
            input_path=prompt("Enter input GIF path", str(output_dir / "test_output.gif") if os.path.exists(str(output_dir / "test_output.gif")) else ""),
            output_path=str(output_dir / "optimized.gif"),
            color_reduction=int(prompt("Color reduction (2-256)", "128")),
            scale_factor=float(prompt("Scale factor (%)", "75")),
            speed_factor=float(prompt("Speed factor (%)", "100"))
        )),
        "3": ("Images to Video", lambda: image_to_video(
            frame_paths=prompt("Enter image paths (comma-separated)", test_images_str).split(','),
            output_path=str(output_dir / "output_video.mp4"),
            fps=int(prompt("Frames per second", "24")),
            crf=int(prompt("Quality (0-51, lower is better)", "20"))
        )),
        "4": ("Optimize Images", lambda: images_optimise(
            input_paths=prompt("Enter image paths (comma-separated)", test_images_str if len(test_images) <= 10 else test_images[0]).split(','),
            output_dir=str(output_dir),
            output_format=prompt("Output format (jpg, png, webp)", "jpg"),
            scale_factor=float(prompt("Scale factor (%)", "75")),
            quality=int(prompt("Quality (0-100)", "80"))
        )),
        "5": ("Convert Image Format", lambda: images_to_format(
            input_paths=prompt("Enter image paths (comma-separated)", test_images_str if len(test_images) <= 10 else test_images[0]).split(','),
            output_format=prompt("Output format (jpg, png, webp, gif)", "jpg"),
            output_dir=str(output_dir),
            quality=int(prompt("Quality (0-100)", "95"))
        )),
        "6": ("Display Image Info", lambda: print(
            get_image_info(prompt("Enter image path", test_images[0] if test_images else ""))
        ))
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