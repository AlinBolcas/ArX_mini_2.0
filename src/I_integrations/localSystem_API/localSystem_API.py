import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pyautogui
import os
from typing import Union, Tuple, Optional
import time

class LocalSystem:
    """
    A utility class that provides various system-level functionalities without requiring external API calls.
    Includes camera operations, screenshot capture, data visualization, and system information.
    """
    
    def __init__(self):
        self.camera = None
        # Update default paths to use output directory with appropriate subdirectories
        self.output_base = "output"
        self.paths = {
            'images': os.path.join(self.output_base, "images"),
            'videos': os.path.join(self.output_base, "videos"),
            'plots': os.path.join(self.output_base, "plots"),
            'screenshots': os.path.join(self.output_base, "screenshots")
        }
        # Create all necessary directories
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

    def capture_webcam(self, save_path: Optional[str] = None, show_preview: bool = True) -> Union[str, None]:
        """
        Captures an image from the default webcam.
        
        Args:
            save_path (str, optional): Path to save the captured image
            show_preview (bool): Whether to show preview window during capture
            
        Returns:
            str: Path to saved image if successful, None if failed
        """
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
            
            # Show preview window until space is pressed
            if show_preview:
                print("Press SPACE to capture or ESC to cancel...")
                while True:
                    ret, frame = self.camera.read()
                    if not ret:
                        break
                        
                    cv2.imshow('Webcam Preview', frame)
                    key = cv2.waitKey(1)
                    if key == 32:  # SPACE
                        cv2.destroyWindow('Webcam Preview')
                        break
                    elif key == 27:  # ESC
                        cv2.destroyWindow('Webcam Preview')
                        return None
            else:
                ret, frame = self.camera.read()
                if not ret:
                    print("[ERROR] Failed to capture image from webcam")
                    return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['images'], f"webcam_{timestamp}.jpg")
            
            cv2.imwrite(filename, frame)
            print(f"[SUCCESS] Webcam capture saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Webcam capture failed: {str(e)}")
            return None
        
    def release_webcam(self):
        """Safely releases the webcam resource"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def capture_screenshot(self, region: Optional[Tuple[int, int, int, int]] = None, 
                         save_path: Optional[str] = None) -> Union[str, None]:
        """
        Captures a screenshot of the entire screen or specified region.
        
        Args:
            region (tuple, optional): Region to capture (left, top, width, height)
            save_path (str, optional): Path to save the screenshot
            
        Returns:
            str: Path to saved screenshot if successful, None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['screenshots'], f"screenshot_{timestamp}.png")
            
            screenshot = pyautogui.screenshot(region=region)
            screenshot.save(filename)
            print(f"[SUCCESS] Screenshot saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Screenshot capture failed: {str(e)}")
            return None

    def plot_data(self, x: np.ndarray, y: np.ndarray, 
                 title: str = "Data Visualization",
                 xlabel: str = "X", ylabel: str = "Y",
                 save_path: Optional[str] = None) -> Union[str, None]:
        """
        Creates a plot using matplotlib and optionally saves it.
        
        Args:
            x (np.ndarray): X-axis data
            y (np.ndarray): Y-axis data
            title (str): Plot title
            xlabel (str): X-axis label
            ylabel (str): Y-axis label
            save_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to saved plot if successful, None if failed
        """
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(x, y)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            
            # Always save, generate default path if none provided
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['plots'], f"plot_{timestamp}.png")
            plt.savefig(filename)
            plt.close()
            print(f"[SUCCESS] Plot saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Plot creation failed: {str(e)}")
            return None

    @classmethod
    def get_datetime(cls, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Returns the current date and time in specified format.
        Can be used both as instance method and class method.
        
        Args:
            format_str (str): DateTime format string
            
        Returns:
            str: Formatted current date and time
        """
        return datetime.now().strftime(format_str)

    def capture_webcam_video(self, duration: float = 10.0, fps: int = 30,
                            save_path: Optional[str] = None,
                            show_preview: bool = True) -> Union[str, None]:
        """
        Captures video from webcam for specified duration.
        
        Args:
            duration (float): Recording duration in seconds
            fps (int): Frames per second
            save_path (str, optional): Path to save video file
            show_preview (bool): Whether to show preview during recording
            
        Returns:
            str: Path to saved video if successful, None if failed
        """
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
            
            frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['videos'], f"webcam_video_{timestamp}.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            
            start_time = time.time()
            frames_captured = 0
            
            while (time.time() - start_time) < duration:
                ret, frame = self.camera.read()
                if not ret:
                    break
                    
                out.write(frame)
                frames_captured += 1
                
                if show_preview:
                    cv2.imshow('Recording...', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                        cv2.destroyWindow('Recording...')
                        out.release()
                        return None
                
                elapsed = int(time.time() - start_time)
                print(f"\rRecording: {elapsed}/{int(duration)} seconds", end="")
            
            if show_preview:
                cv2.destroyWindow('Recording...')
            out.release()
            print(f"\n[SUCCESS] Webcam video saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Webcam video capture failed: {str(e)}")
            return None

    def capture_screen_video(self, duration: float = 10.0, fps: int = 30,
                           region: Optional[Tuple[int, int, int, int]] = None,
                           save_path: Optional[str] = None,
                           show_preview: bool = True) -> Union[str, None]:
        """
        Records screen activity for specified duration.
        
        Args:
            duration (float): Recording duration in seconds
            fps (int): Frames per second
            region (tuple, optional): Region to capture (left, top, width, height)
            save_path (str, optional): Path to save video file
            show_preview (bool): Whether to show recording preview
            
        Returns:
            str: Path to saved video if successful, None if failed
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['videos'], f"screen_recording_{timestamp}.mp4")
            
            first_screenshot = pyautogui.screenshot(region=region)
            frame_width, frame_height = first_screenshot.size
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))
            
            start_time = time.time()
            
            while (time.time() - start_time) < duration:
                screenshot = pyautogui.screenshot(region=region)
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                out.write(frame)
                
                if show_preview:
                    cv2.imshow('Screen Recording...', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                        cv2.destroyWindow('Screen Recording...')
                        out.release()
                        return None
                
                elapsed = int(time.time() - start_time)
                print(f"\rRecording: {elapsed}/{int(duration)} seconds", end="")
                
                # Sleep to maintain FPS
                time.sleep(max(0, 1/fps - (time.time() - start_time) % (1/fps)))
            
            if show_preview:
                cv2.destroyWindow('Screen Recording...')
            out.release()
            print(f"\n[SUCCESS] Screen recording saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Screen recording failed: {str(e)}")
            return None

    def get_system_info(self) -> dict:
        """
        Returns basic system information.
        
        Returns:
            dict: System information including OS, CPU, memory, etc.
        """
        import platform
        import psutil
        
        try:
            info = {
                "os": platform.system(),
                "os_version": platform.version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available,
                "disk_usage": psutil.disk_usage('/').percent,
                "python_version": platform.python_version(),
                "timestamp": self.get_datetime()
            }
            return info
            
        except Exception as e:
            print(f"[ERROR] Failed to get system info: {str(e)}")
            return {}
        
    def __del__(self):
        """Cleanup method to ensure webcam is released"""
        self.release_webcam()
