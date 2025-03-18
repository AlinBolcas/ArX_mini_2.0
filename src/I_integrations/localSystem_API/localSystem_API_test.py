import sys
from pathlib import Path
import logging
from typing import List, Dict

import os
import subprocess

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import required modules and functions
utils = finder.import_module('utils.py')
printColoured = utils.printColoured
quick_look = utils.quick_look

# Custom formatter for prettier logging
class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if 'Testing' in record.msg:
            return f"\n{printColoured('>> ' + record.msg, 'magenta')}"
        elif 'Response:' in record.msg:
            return f"\n{printColoured('Response:', 'yellow')}\n{record.msg.replace('Response:', '')}"
        elif 'Error:' in record.msg:
            return f"\n{printColoured('❌ ' + record.msg, 'red')}"
        elif 'Success:' in record.msg:
            return f"\n{printColoured('✓ ' + record.msg, 'green')}"
        else:
            return f"{printColoured('>', 'blue')} {record.msg}"

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

from modules.I_integrations.localSystem_API.localSystem_API import LocalSystem
import numpy as np
import time

def preview_media(file_path: str) -> None:
    """Safely preview media files."""
    try:
        # For video files, use system default player directly
        if file_path.endswith(('.mp4', '.avi', '.mov')):
            logger.info(f"Video saved to: {file_path}")
            # On macOS, use 'open' command which is more reliable for videos
            if sys.platform == "darwin":
                subprocess.run(["open", "-a", "QuickTime Player", file_path])
            elif sys.platform == "win32":
                os.startfile(file_path)
            else:
                subprocess.run(["xdg-open", file_path])
            return
            
        # For images and other files, use QuickLook
        quick_look(file_path)
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")

def test_webcam_capture(local_sys: LocalSystem) -> None:
    """Test webcam image capture functionality"""
    logger.info("Testing Webcam Image Capture")
    
    try:
        # Test single image capture with preview
        logger.info("Capturing single image (press SPACE to capture, ESC to cancel)...")
        image_path = local_sys.capture_webcam(show_preview=True)
        
        if image_path and os.path.exists(image_path):
            logger.info(f"Success: Captured and saved image to: {image_path}")
            preview_media(image_path)
        else:
            logger.error("Failed to capture webcam image")
    finally:
        local_sys.release_webcam()
        logger.info("Released webcam resource")

def test_webcam_video(local_sys: LocalSystem) -> None:
    """Test webcam video capture functionality"""
    logger.info("Testing Webcam Video Capture")
    
    try:
        logger.info("Recording 3-second video (press ESC to cancel)...")
        video_path = local_sys.capture_webcam_video(duration=3.0, fps=30, show_preview=True)
        
        if video_path and os.path.exists(video_path):
            logger.info(f"Success: Captured video saved to: {video_path}")
            preview_media(video_path)
        else:
            logger.error("Failed to capture webcam video")
    finally:
        local_sys.release_webcam()
        logger.info("Released webcam resource")

def test_screenshot(local_sys: LocalSystem) -> None:
    """Test screenshot functionality"""
    logger.info("Testing Screenshot Capture")
    
    # Test full screenshot
    logger.info("Capturing full screenshot...")
    screenshot_path = local_sys.capture_screenshot()
    
    if screenshot_path and os.path.exists(screenshot_path):
        logger.info(f"Success: Full screenshot saved to: {screenshot_path}")
        preview_media(screenshot_path)
    else:
        logger.error("Failed to capture screenshot")
    
    # Add small delay between screenshots
    time.sleep(1)
    
    # Test region screenshot
    logger.info("Capturing region screenshot...")
    region = (0, 0, 500, 500)  # Capture 500x500 region from top-left
    region_path = local_sys.capture_screenshot(region=region)
    
    if region_path and os.path.exists(region_path):
        logger.info(f"Success: Region screenshot saved to: {region_path}")
        preview_media(region_path)
    else:
        logger.error("Failed to capture region screenshot")

def test_screen_recording(local_sys: LocalSystem) -> None:
    """Test screen recording functionality"""
    logger.info("Testing Screen Recording")
    
    logger.info("Recording 3-second screen capture (press ESC to cancel)...")
    video_path = local_sys.capture_screen_video(duration=3.0, fps=10, show_preview=True)
    
    if video_path and os.path.exists(video_path):
        logger.info(f"Success: Screen recording saved to: {video_path}")
        preview_media(video_path)
    else:
        logger.error("Failed to capture screen recording")

def test_plot_data(local_sys: LocalSystem) -> None:
    """Test data plotting functionality"""
    logger.info("Testing Data Plotting")
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    logger.info("Creating and saving plot...")
    save_path = os.path.join(local_sys.paths['plots'], "test_plot.png")
    result = local_sys.plot_data(x, y, title="Test Plot", xlabel="X", ylabel="Y", save_path=save_path)
    
    if result and os.path.exists(save_path):
        logger.info(f"Success: Plot saved to: {save_path}")
        preview_media(save_path)
    else:
        logger.error("Failed to save plot")

def test_datetime(local_sys: LocalSystem) -> None:
    """Test datetime functionality"""
    logger.info("Testing DateTime Functions")
    
    # Test instance method
    instance_time = local_sys.get_datetime()
    logger.info(f"Instance time: {instance_time}")
    
    # Test class method
    class_time = LocalSystem.get_datetime()
    logger.info(f"Class time: {class_time}")
    
    # Test custom format
    custom_time = LocalSystem.get_datetime("%Y-%m-%d")
    logger.info(f"Custom format time: {custom_time}")

def test_system_info(local_sys: LocalSystem) -> None:
    """Test system information retrieval"""
    logger.info("Testing System Info")
    
    sys_info = local_sys.get_system_info()
    if sys_info:
        logger.info("System Information:")
        for key, value in sys_info.items():
            logger.info(f"{key}: {value}")
    else:
        logger.error("Failed to retrieve system information")

def main():
    """Run all LocalSystem API tests."""
    logger.info("\n🚀 Starting LocalSystem API Tests")
    logger.info("===============================")
    
    try:
        # Initialize LocalSystem
        local_sys = LocalSystem()
        
        # Run non-intrusive tests first
        test_datetime(local_sys)
        test_system_info(local_sys)
        test_plot_data(local_sys)
        
        # Ask for permission before camera/screen tests
        if input("\nDo you want to run camera and screen capture tests? (y/n): ").lower() == 'y':
            test_webcam_capture(local_sys)
            test_webcam_video(local_sys)
            test_screenshot(local_sys)
            test_screen_recording(local_sys)
        
        logger.info("\n✨ All Tests Complete")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 