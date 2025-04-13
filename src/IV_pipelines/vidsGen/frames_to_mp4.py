#!/usr/bin/env python3
"""
Frames to Video Converter
--------------
A GUI application for converting frame sequences to high-quality MP4 videos.

Features:
- Select a folder of image frames
- Configure frames per second (FPS)
- Set quality level for the output video
- Progress tracking
- Success/failure notifications
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Optional
import customtkinter as ctk
from pathlib import Path
import glob

# Import video and image utilities
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Go up vidsGen, IV_pipelines, src
sys.path.insert(0, str(project_root))

try:
    from src.VI_utils.image_utils import image_to_video
except ImportError as e:
    print(f"ERROR: Could not import image utilities. Error: {e}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Import Error", f"Could not import image utilities. Check console for details. Error: {e}")
        root.destroy()
    except ImportError:
        pass
    sys.exit(1)

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ConversionJob:
    """Class to track a single frames-to-video conversion job"""
    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self.folder_name = os.path.basename(input_folder)
        self.output_path = os.path.join(os.path.dirname(input_folder), f"{self.folder_name}.mp4")
        self.status = "Pending"
        self.success = False
        self.frame_count = 0
        self.fps = 8  # Default FPS
        self.crf = 18  # Default CRF (18 is visually lossless)
        self.preset = "slow"  # Default preset

class FramesToVideoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Frames to Video Converter")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Variables
        self.conversion_running = False
        self.jobs: List[ConversionJob] = []
        self.current_job_index = 0
        
        # Create the main layout
        self.create_ui()
        print("App initialized, UI created")

    def create_ui(self):
        """Create the user interface"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)  # Files list should expand
        
        # === Folder Selection Section ===
        select_frame = ctk.CTkFrame(main_frame)
        select_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        select_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(select_frame, text="Frames to Video Converter", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(5, 10), sticky="w")
        
        # Folder selection buttons
        ctk.CTkButton(
            select_frame,
            text="Select Frames Folder",
            command=self.select_folder,
            width=200,
            height=40,
            font=("Helvetica", 14)
        ).grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        
        ctk.CTkButton(
            select_frame,
            text="Clear Selection",
            command=self.clear_selection,
            width=200,
            height=40,
            fg_color="#D35B58",
            hover_color="#C84744",
            font=("Helvetica", 14)
        ).grid(row=1, column=1, padx=(5, 10), pady=5, sticky="w")
        
        # Settings frame
        settings_frame = ctk.CTkFrame(select_frame)
        settings_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        settings_frame.grid_columnconfigure(1, weight=1)
        
        # FPS setting
        ctk.CTkLabel(settings_frame, text="Frames Per Second (FPS):").grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        self.fps_var = tk.StringVar(value="8")
        fps_entry = ctk.CTkEntry(settings_frame, textvariable=self.fps_var, width=100)
        fps_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Quality setting
        ctk.CTkLabel(settings_frame, text="Video Quality:").grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        self.quality_var = tk.StringVar(value="High (CRF 18)")
        quality_options = ["High (CRF 18)", "Medium (CRF 23)", "Standard (CRF 28)"]
        quality_menu = ctk.CTkOptionMenu(settings_frame, variable=self.quality_var, values=quality_options, width=200)
        quality_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # Encoding speed setting
        ctk.CTkLabel(settings_frame, text="Encoding Speed:").grid(row=2, column=0, padx=(10, 5), pady=5, sticky="w")
        self.preset_var = tk.StringVar(value="slow")
        preset_options = ["ultrafast", "fast", "medium", "slow", "veryslow"]
        preset_menu = ctk.CTkOptionMenu(settings_frame, variable=self.preset_var, values=preset_options, width=200)
        preset_menu.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        # === Folders List Section ===
        folders_frame = ctk.CTkFrame(main_frame)
        folders_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        folders_frame.grid_columnconfigure(0, weight=1)
        folders_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(folders_frame, text="Selected Folders", font=("Helvetica", 14, "bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Scrollable folder list
        folder_list_container = ctk.CTkScrollableFrame(folders_frame)
        folder_list_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        folder_list_container.grid_columnconfigure(0, weight=1)
        
        self.folders_list_frame = folder_list_container
        
        # === Progress Section ===
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        progress_frame.grid_columnconfigure(0, weight=1)
        
        # Progress bar
        ctk.CTkLabel(progress_frame, text="Progress:").grid(row=0, column=0, padx=10, pady=(10, 0), sticky="w")
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")
        self.progress_bar.set(0)
        
        # Current task label
        self.task_label = ctk.CTkLabel(progress_frame, text="Ready to convert")
        self.task_label.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="w")
        
        # Status message
        self.status_frame = ctk.CTkFrame(progress_frame, corner_radius=5)
        self.status_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_message = ctk.CTkLabel(
            self.status_frame,
            text="Select a folder of image frames to convert to video",
            font=("Helvetica", 14),
            height=30
        )
        self.status_message.grid(row=0, column=0, padx=10, pady=10)
        
        # Convert button
        self.convert_button = ctk.CTkButton(
            progress_frame,
            text="Start Conversion",
            command=self.start_conversion,
            width=300,
            height=50,
            font=("Helvetica", 16, "bold"),
            fg_color="#28A745",
            hover_color="#218838"
        )
        self.convert_button.grid(row=4, column=0, padx=10, pady=15)
        
        # Initialize empty folder list
        self.update_folder_list()

    def select_folder(self):
        """Open folder dialog to select a directory of image frames"""
        if self.conversion_running:
            return
            
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Image Frames"
        )
        
        if not folder_path:
            return
        
        # Check if folder contains image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        if not image_files:
            messagebox.showwarning("No Images Found", f"No image files found in {folder_path}")
            return
            
        # Create a job for this folder
        if not any(job.input_folder == folder_path for job in self.jobs):
            job = ConversionJob(folder_path)
            job.frame_count = len(image_files)
            self.jobs.append(job)
            print(f"Added folder: {folder_path} with {job.frame_count} frames")
        
        self.update_folder_list()

    def clear_selection(self):
        """Clear the selected folders"""
        if self.conversion_running:
            return
            
        self.jobs = []
        self.update_folder_list()
        print("Cleared folder selection")

    def update_folder_list(self):
        """Update the folders list display"""
        # Clear existing widgets
        for widget in self.folders_list_frame.winfo_children():
            widget.destroy()
            
        if not self.jobs:
            empty_label = ctk.CTkLabel(
                self.folders_list_frame,
                text="No folders selected",
                font=("Helvetica", 12),
                text_color="gray"
            )
            empty_label.grid(row=0, column=0, pady=20)
            return
            
        # Add header row
        header_frame = ctk.CTkFrame(self.folders_list_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkLabel(header_frame, text="Folder Name", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(header_frame, text="Frames", font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(header_frame, text="Status", font=("Helvetica", 12, "bold")).grid(row=0, column=2, padx=5, pady=5)
        
        # Add folder rows
        for i, job in enumerate(self.jobs):
            row_frame = ctk.CTkFrame(self.folders_list_frame)
            row_frame.grid(row=i+1, column=0, sticky="ew", pady=2)
            row_frame.grid_columnconfigure((0, 1, 2), weight=1)
            
            # Folder name - truncate if too long
            folder_name = job.folder_name
            if len(folder_name) > 40:
                folder_name = folder_name[:37] + "..."
                
            ctk.CTkLabel(row_frame, text=folder_name, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            
            # Frame count
            ctk.CTkLabel(row_frame, text=str(job.frame_count)).grid(row=0, column=1, padx=5, pady=5)
            
            # Status
            status_label = ctk.CTkLabel(row_frame, text=job.status)
            if job.status == "Success":
                status_label.configure(text_color="#28A745")
            elif job.status == "Failed":
                status_label.configure(text_color="#DC3545")
                
            status_label.grid(row=0, column=2, padx=5, pady=5)

    def update_progress(self, progress: float, task_description: str):
        """Update the progress bar and task description"""
        self.progress_bar.set(progress)
        self.task_label.configure(text=task_description)
        self.update_idletasks()

    def update_status(self, message: str, is_success: bool = None):
        """Update the status message with optional success/error styling"""
        self.status_message.configure(text=message)
        
        if is_success is True:
            self.status_frame.configure(fg_color="#28A745")
            self.status_message.configure(text_color="white")
        elif is_success is False:
            self.status_frame.configure(fg_color="#DC3545")
            self.status_message.configure(text_color="white")
        else:
            self.status_frame.configure(fg_color="transparent")
            self.status_message.configure(
                text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]
            )
            
        self.update_idletasks()
        print(f"Status update: {message}")

    def get_crf_from_quality(self) -> int:
        """Convert quality string to CRF value"""
        quality = self.quality_var.get()
        if "High" in quality:
            return 18
        elif "Medium" in quality:
            return 23
        else:
            return 28

    def start_conversion(self):
        """Start the conversion process"""
        if self.conversion_running:
            print("Conversion already running")
            return
            
        if not self.jobs:
            messagebox.showinfo("No Folders", "Please select at least one folder of image frames.")
            print("No folders selected")
            return
            
        try:
            fps = int(self.fps_var.get())
            if fps < 1:
                raise ValueError("FPS must be at least 1")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid FPS value: {str(e)}")
            return
            
        # Get quality settings
        crf = self.get_crf_from_quality()
        preset = self.preset_var.get()
        
        # Update job settings
        for job in self.jobs:
            job.fps = fps
            job.crf = crf
            job.preset = preset
        
        print("Starting conversion process")
        print(f"Settings: FPS={fps}, CRF={crf}, Preset={preset}")
        
        # Disable UI controls
        self.conversion_running = True
        self.convert_button.configure(
            text="Converting...",
            state="disabled",
            fg_color="#6C757D"
        )
        
        # Start conversion thread
        threading.Thread(target=self.conversion_thread, daemon=True).start()

    def find_image_files(self, folder_path: str) -> List[str]:
        """Find all image files in a folder, sorted by name"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
        
        # Sort files naturally (assuming they have frame numbers)
        image_files.sort()
        return image_files

    def conversion_thread(self):
        """Run the conversion process in a separate thread"""
        try:
            print("Conversion thread started")
            total_jobs = len(self.jobs)
            completed_jobs = 0
            
            for i, job in enumerate(self.jobs):
                self.current_job_index = i
                folder_name = job.folder_name
                
                # Update progress
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, f"Converting {folder_name}")
                print(f"Processing job {i+1}/{total_jobs}: {folder_name}")
                
                # Find all image files in the folder
                image_files = self.find_image_files(job.input_folder)
                
                if not image_files:
                    job.status = "Failed"
                    job.success = False
                    self.update_status(f"No image files found in {folder_name}", False)
                    self.update_folder_list()
                    continue
                
                job.frame_count = len(image_files)
                print(f"Found {job.frame_count} image files")
                
                # Converting frames to video
                self.update_status(f"Converting {job.frame_count} frames to video...")
                
                # Use the image_to_video function from image_utils.py
                success = image_to_video(
                    frame_paths=image_files,
                    output_path=job.output_path,
                    fps=job.fps,
                    crf=job.crf,
                    preset=job.preset
                )
                
                # Update job status
                job.success = success
                job.status = "Success" if success else "Failed"
                
                if success:
                    self.update_status(f"Successfully created video: {os.path.basename(job.output_path)}", True)
                    print(f"Video created: {job.output_path}")
                else:
                    self.update_status(f"Failed to create video from {folder_name}", False)
                    print(f"Failed to create video: {job.output_path}")
                
                # Update UI
                self.update_folder_list()
                
                # Mark job as completed
                completed_jobs += 1
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, "Completed")
            
            # All jobs completed
            if all(job.success for job in self.jobs):
                self.update_status("All videos created successfully!", True)
            elif any(job.success for job in self.jobs):
                self.update_status("Some videos created with errors.", None)
            else:
                self.update_status("All conversions failed!", False)
                
            print("Conversion thread finished")
                
        except Exception as e:
            error_msg = f"Error during conversion: {str(e)}"
            self.update_status(error_msg, False)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-enable UI controls
            self.conversion_running = False
            self.convert_button.configure(
                text="Start Conversion",
                state="normal",
                fg_color="#28A745",
                hover_color="#218838"
            )
            print("UI controls re-enabled")

def main():
    print("Starting Frames to Video Converter App")
    app = FramesToVideoApp()
    app.mainloop()

if __name__ == "__main__":
    main()
