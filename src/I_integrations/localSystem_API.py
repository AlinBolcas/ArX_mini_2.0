import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pyautogui
import os
from typing import Union, Tuple, Optional
import time
import platform
import psutil
import sys
import subprocess

class LocalSystem:
    """
    A utility class that provides various system-level functionalities without requiring external API calls.
    Includes camera operations, screenshot capture, data visualization, and system information.
    """
    
    def __init__(self):
        self.camera = None
        # Update default paths to use data/output directory with appropriate subdirectories
        self.output_base = os.path.join("data", "output")
        self.paths = {
            'images': os.path.join(self.output_base, "images"),
            'videos': os.path.join(self.output_base, "videos"),
            'plots': os.path.join(self.output_base, "plots"),
            'screenshots': os.path.join(self.output_base, "screenshots")
        }
        # Create all necessary directories
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        print(f"LocalSystem initialized. Output directory: ./{self.output_base}")

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
                if not self.camera.isOpened():
                    print("[ERROR] Could not open webcam.")
                    self.camera = None
                    return None

            # Show preview window until space is pressed
            if show_preview:
                print("Press SPACE to capture or ESC to cancel...")
                while True:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("[ERROR] Failed to read frame from webcam during preview.")
                        cv2.destroyAllWindows()
                        return None

                    cv2.imshow('Webcam Preview', frame)
                    key = cv2.waitKey(1)
                    if key == 32:  # SPACE
                        cv2.destroyWindow('Webcam Preview')
                        break
                    elif key == 27:  # ESC
                        print("Webcam capture cancelled by user.")
                        cv2.destroyWindow('Webcam Preview')
                        return None
                ret, frame = self.camera.read()
                if not ret:
                    print("[ERROR] Failed to capture final image from webcam after preview.")
                    return None

            else:
                ret, frame = self.camera.read()
                if not ret:
                    print("[ERROR] Failed to capture image from webcam (no preview).")
                    return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['images'], f"webcam_{timestamp}.jpg")
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            success = cv2.imwrite(filename, frame)

            if success:
                print(f"[SUCCESS] Webcam capture saved to: {filename}")
                return filename
            else:
                print(f"[ERROR] Failed to save webcam image to {filename}")
                return None
            
        except Exception as e:
            print(f"[ERROR] Webcam capture failed: {str(e)}")
            cv2.destroyAllWindows()
            return None
        
    def release_webcam(self):
        """Safely releases the webcam resource"""
        if self.camera is not None:
            if self.camera.isOpened():
                self.camera.release()
                print("Webcam resource released.")
            self.camera = None
            cv2.destroyAllWindows()

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
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)

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
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['plots'], f"plot_{timestamp}.png")
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            plt.savefig(filename)
            plt.close()
            print(f"[SUCCESS] Plot saved to: {filename}")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Plot creation failed: {str(e)}")
            plt.close()
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
        out = None
        try:
            if self.camera is None:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    print("[ERROR] Could not open webcam.")
                    self.camera = None
                    return None

            frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['videos'], f"webcam_video_{timestamp}.mp4")
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, float(fps), (frame_width, frame_height))

            if not out.isOpened():
                 print(f"[ERROR] Failed to open video writer for {filename}")
                 return None

            print(f"Starting webcam video recording for {duration} seconds...")
            start_time = time.time()
            frames_captured = 0

            while (time.time() - start_time) < duration:
                ret, frame = self.camera.read()
                if not ret:
                    print("[WARNING] Dropped frame during webcam video capture.")
                    continue

                out.write(frame)
                frames_captured += 1

                if show_preview:
                    remaining_time = max(0, duration - (time.time() - start_time))
                    preview_text = f"Recording... {remaining_time:.1f}s left (ESC to stop)"
                    cv2.putText(frame, preview_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Recording...', frame)

                    if cv2.waitKey(1) & 0xFF == 27:
                        print("\nWebcam recording stopped by user.")
                        break

            if show_preview:
                cv2.destroyWindow('Recording...')

            out.release()
            out = None

            print(f"\n[SUCCESS] Webcam video saved to: {filename} ({frames_captured} frames captured)")
            return filename
            
        except Exception as e:
            print(f"[ERROR] Webcam video capture failed: {str(e)}")
            if out is not None and out.isOpened():
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
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
            region (tuple, optional): Region to capture (left, top, width, height). Full screen if None.
            save_path (str, optional): Path to save video file
            show_preview (bool): Whether to show recording preview (can impact performance)
            
        Returns:
            str: Path to saved video if successful, None if failed
        """
        out = None
        preview_window_name = 'Screen Recording...'
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_path or os.path.join(self.paths['videos'], f"screen_recording_{timestamp}.mp4")
            
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            if region:
                monitor_region = {'left': region[0], 'top': region[1], 'width': region[2], 'height': region[3]}
                frame_width, frame_height = region[2], region[3]
            else:
                screen_size = pyautogui.size()
                monitor_region = {'left': 0, 'top': 0, 'width': screen_size.width, 'height': screen_size.height}
                frame_width, frame_height = screen_size.width, screen_size.height

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, float(fps), (frame_width, frame_height))

            if not out.isOpened():
                 print(f"[ERROR] Failed to open video writer for {filename}")
                 return None

            print(f"Starting screen recording for {duration} seconds...")
            start_time = time.time()
            frame_interval = 1.0 / fps
            last_frame_time = start_time
            frames_captured = 0

            while (time.time() - start_time) < duration:
                capture_start_time = time.time()

                try:
                    screenshot = pyautogui.screenshot(region=region)
                    frame_np = np.array(screenshot)
                    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                except Exception as scr_e:
                    print(f"[WARNING] Failed to capture screenshot frame: {scr_e}")
                    time.sleep(frame_interval / 2)
                    continue

                out.write(frame_bgr)
                frames_captured += 1

                if show_preview:
                    try:
                        remaining_time = max(0, duration - (time.time() - start_time))
                        preview_text = f"Rec: {remaining_time:.1f}s (ESC stop)"

                        preview_frame = frame_bgr.copy()

                        cv2.putText(preview_frame, preview_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(preview_window_name, preview_frame)

                        if cv2.waitKey(1) & 0xFF == 27:
                            print("\nScreen recording stopped by user.")
                            break
                    except Exception as cv_e:
                         print(f"[WARNING] Error during preview update: {cv_e}")

                current_time = time.time()
                elapsed_since_last = current_time - last_frame_time
                sleep_time = frame_interval - elapsed_since_last
                if sleep_time > 0:
                    time.sleep(sleep_time)
                last_frame_time = time.time()

            if show_preview:
                cv2.destroyWindow(preview_window_name)

            out.release()
            out = None

            print(f"\n[SUCCESS] Screen recording saved to: {filename} ({frames_captured} frames captured)")
            return filename

        except Exception as e:
            print(f"[ERROR] Screen recording failed: {str(e)}")
            if out is not None and out.isOpened():
                out.release()
            if show_preview:
                cv2.destroyAllWindows()
            return None

    def get_system_info(self) -> dict:
        """
        Returns basic system information.
        
        Returns:
            dict: System information including OS, CPU, memory, etc.
        """
        try:
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            info = {
                "os": platform.system(),
                "os_version": platform.release(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_physical_cores": psutil.cpu_count(logical=False),
                "cpu_total_cores": psutil.cpu_count(logical=True),
                "cpu_usage_percent": psutil.cpu_percent(interval=0.5),
                "memory_total_gb": round(mem.total / (1024**3), 2),
                "memory_available_gb": round(mem.available / (1024**3), 2),
                "memory_used_percent": mem.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_used_percent": disk.percent,
                "timestamp": self.get_datetime()
            }
            return info
            
        except Exception as e:
            print(f"[ERROR] Failed to get system info: {str(e)}")
            return {}
        
    def __del__(self):
        """Cleanup method to ensure webcam is released when object is destroyed"""
        print("LocalSystem object cleanup: Releasing webcam if necessary.")
        self.release_webcam()

def preview_media(file_path: str) -> None:
    """Safely preview media files using default system applications."""
    if not os.path.exists(file_path):
        print(f"[ERROR] Preview failed: File not found at {file_path}")
        return

    try:
        print(f"Attempting to open: {file_path}")
        if sys.platform == "darwin":
            subprocess.run(["open", file_path], check=True)
        elif sys.platform == "win32":
            os.startfile(os.path.normpath(file_path))
        else:
            subprocess.run(["xdg-open", file_path], check=True)
        print(f"Opened {file_path} with default application.")
    except FileNotFoundError:
         print(f"[ERROR] Could not find command to open file on this system ({sys.platform}).")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to open file with system command: {e}")
    except Exception as e:
        print(f"[ERROR] Preview failed due to unexpected error: {str(e)}")

def test_webcam_capture(local_sys: LocalSystem) -> None:
    """Test webcam image capture functionality"""
    print("\n--- Testing Webcam Image Capture ---")
    image_path = None
    try:
        print("Capturing single image (press SPACE to capture, ESC to cancel)...")
        image_path = local_sys.capture_webcam(show_preview=True)

        if image_path and os.path.exists(image_path):
            print(f"[OK] Success: Captured and saved image to: {image_path}")
            preview_media(image_path)
        elif image_path is None:
             print("[INFO] Webcam capture cancelled or failed.")
        else:
             print(f"[FAIL] File not found after reported save: {image_path}")
    except Exception as e:
         print(f"[FAIL] Exception during webcam capture test: {e}")
    finally:
        local_sys.release_webcam()
        print("Webcam released after test.")

def test_webcam_video(local_sys: LocalSystem) -> None:
    """Test webcam video capture functionality"""
    print("\n--- Testing Webcam Video Capture ---")
    video_path = None
    try:
        print("Recording 3-second video (press ESC to cancel)...")
        video_path = local_sys.capture_webcam_video(duration=3.0, fps=20, show_preview=True)

        if video_path and os.path.exists(video_path):
            print(f"[OK] Success: Captured video saved to: {video_path}")
            preview_media(video_path)
        elif video_path is None:
            print("[INFO] Webcam video capture cancelled or failed.")
        else:
            print(f"[FAIL] File not found after reported save: {video_path}")
    except Exception as e:
         print(f"[FAIL] Exception during webcam video test: {e}")
    finally:
        local_sys.release_webcam()
        print("Webcam released after test.")

def test_screenshot(local_sys: LocalSystem) -> None:
    """Test screenshot functionality"""
    print("\n--- Testing Screenshot Capture ---")

    print("Capturing full screenshot...")
    screenshot_path = local_sys.capture_screenshot()

    if screenshot_path and os.path.exists(screenshot_path):
        print(f"[OK] Success: Full screenshot saved to: {screenshot_path}")
        preview_media(screenshot_path)
    else:
        print("[FAIL] Failed to capture or save full screenshot")

    time.sleep(1)

    print("Capturing region screenshot (500x500 from top-left)...")
    region = (0, 0, 500, 500)
    region_path = local_sys.capture_screenshot(region=region)

    if region_path and os.path.exists(region_path):
        print(f"[OK] Success: Region screenshot saved to: {region_path}")
        preview_media(region_path)
    else:
        print("[FAIL] Failed to capture or save region screenshot")

def test_screen_recording(local_sys: LocalSystem) -> None:
    """Test screen recording functionality"""
    print("\n--- Testing Screen Recording ---")
    video_path = None
    try:
        print("Recording 3-second screen capture (press ESC to cancel)...")
        video_path = local_sys.capture_screen_video(duration=3.0, fps=10, show_preview=True)

        if video_path and os.path.exists(video_path):
            print(f"[OK] Success: Screen recording saved to: {video_path}")
            preview_media(video_path)
        elif video_path is None:
            print("[INFO] Screen recording cancelled or failed.")
        else:
            print(f"[FAIL] File not found after reported save: {video_path}")
    except Exception as e:
        print(f"[FAIL] Exception during screen recording test: {e}")

def test_plot_data(local_sys: LocalSystem) -> None:
    """Test data plotting functionality"""
    print("\n--- Testing Data Plotting ---")

    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/5)

    print("Creating and saving plot...")
    save_path = os.path.join(local_sys.paths['plots'], "test_plot_sin_decay.png")
    result = local_sys.plot_data(x, y, title="Sine Wave Decay Test Plot", xlabel="Time (s)", ylabel="Amplitude", save_path=save_path)

    if result and os.path.exists(save_path):
        print(f"[OK] Success: Plot saved to: {save_path}")
        preview_media(save_path)
    else:
        print(f"[FAIL] Failed to save plot to {save_path}")

def test_datetime(local_sys: LocalSystem) -> None:
    """Test datetime functionality"""
    print("\n--- Testing DateTime Functions ---")

    instance_time = local_sys.get_datetime()
    print(f"Instance time (default format): {instance_time}")

    class_time = LocalSystem.get_datetime(format_str="%a, %d %b %Y %H:%M:%S")
    print(f"Class time (custom format): {class_time}")

    custom_time = LocalSystem.get_datetime("%Y-%m-%d")
    print(f"Just date: {custom_time}")
    print("[OK] DateTime functions seem operational.")

def test_system_info(local_sys: LocalSystem) -> None:
    """Test system information retrieval"""
    print("\n--- Testing System Info ---")

    sys_info = local_sys.get_system_info()
    if sys_info:
        print("[OK] System Information Retrieved:")
        for key, value in sys_info.items():
            print(f"  - {key.replace('_', ' ').title()}: {value}")
    else:
        print("[FAIL] Failed to retrieve system information")

if __name__ == "__main__":
    """Run all LocalSystem API tests."""
    print("\n🚀 Starting LocalSystem API Tests")
    print("===============================")

    local_system_instance = None
    try:
        local_system_instance = LocalSystem()

        test_datetime(local_system_instance)
        test_system_info(local_system_instance)
        test_plot_data(local_system_instance)
        test_screenshot(local_system_instance)

        print("\n--- Interactive Tests ---")
        print("The following tests require access to your webcam and screen recording.")
        run_interactive = input("Do you want to proceed? (y/n): ").strip().lower()

        if run_interactive == 'y':
            test_webcam_capture(local_system_instance)
            print("Waiting 2 seconds before next camera test...")
            time.sleep(2)
            test_webcam_video(local_system_instance)
            print("Waiting 2 seconds before screen recording test...")
            time.sleep(2)
            test_screen_recording(local_system_instance)
        else:
            print("Skipping interactive webcam and screen recording tests.")

        print("\n=======================")
        print("✨ All Tests Complete ✨")
        print("=======================")

    except Exception as e:
        print(f"\n🚨 Test suite failed with an unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if local_system_instance:
            print("Final cleanup: Releasing webcam.")
            local_system_instance.release_webcam()