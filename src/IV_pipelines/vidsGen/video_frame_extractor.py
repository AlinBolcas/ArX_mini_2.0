#!/usr/bin/env python3
"""
Video Frame Extractor
--------------
A GUI application for extracting frames from video files.

Features:
- Select MP4 files for frame extraction
- Choose output directory
- Configure frame extraction rate
- Configure rotation angle
- Progress tracking
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Dict, Optional
import customtkinter as ctk
from pathlib import Path

# Import video utilities
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2]  # Go up vidsGen, IV_pipelines, src
sys.path.insert(0, str(project_root))

try:
    from src.VI_utils.video_utils import video_to_frames, get_video_info
except ImportError as e:
    print(f"ERROR: Could not import video utilities. Error: {e}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Import Error", f"Could not import video utilities. Check console for details. Error: {e}")
        root.destroy()
    except ImportError:
        pass
    sys.exit(1)

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class ExtractionJob:
    """Class to track a single file extraction job"""
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.output_folder = None
        self.status = "Pending"
        self.success = False
        self.frames_extracted = 0
        self.frame_rate = 1  # Extract every frame by default
        self.rotation_angle = 0  # No rotation by default

class VideoFrameExtractorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor")
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Variables
        self.extraction_running = False
        self.jobs: List[ExtractionJob] = []
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
        
        # === File Selection Section ===
        select_frame = ctk.CTkFrame(main_frame)
        select_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        select_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(select_frame, text="Video Frame Extractor", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(5, 10), sticky="w")
        
        # File selection buttons
        ctk.CTkButton(
            select_frame,
            text="Select Video Files",
            command=self.select_files,
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
        
        # Frame rate setting
        ctk.CTkLabel(settings_frame, text="Extract every N frames:").grid(row=0, column=0, padx=(10, 5), pady=5, sticky="w")
        self.frame_rate_var = tk.StringVar(value="1")
        frame_rate_entry = ctk.CTkEntry(settings_frame, textvariable=self.frame_rate_var, width=100)
        frame_rate_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Rotation angle setting
        ctk.CTkLabel(settings_frame, text="Rotation angle:").grid(row=1, column=0, padx=(10, 5), pady=5, sticky="w")
        self.rotation_var = tk.StringVar(value="0")
        rotation_options = ["0", "90", "180", "270"]
        rotation_menu = ctk.CTkOptionMenu(settings_frame, variable=self.rotation_var, values=rotation_options, width=100)
        rotation_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        # === Files List Section ===
        files_frame = ctk.CTkFrame(main_frame)
        files_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        files_frame.grid_columnconfigure(0, weight=1)
        files_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(files_frame, text="Selected Files", font=("Helvetica", 14, "bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Scrollable file list
        file_list_container = ctk.CTkScrollableFrame(files_frame)
        file_list_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        file_list_container.grid_columnconfigure(0, weight=1)
        
        self.files_list_frame = file_list_container
        
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
        self.task_label = ctk.CTkLabel(progress_frame, text="Ready to extract frames")
        self.task_label.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="w")
        
        # Status message
        self.status_frame = ctk.CTkFrame(progress_frame, corner_radius=5)
        self.status_frame.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.status_frame.grid_columnconfigure(0, weight=1)
        
        self.status_message = ctk.CTkLabel(
            self.status_frame,
            text="Select video files to extract frames",
            font=("Helvetica", 14),
            height=30
        )
        self.status_message.grid(row=0, column=0, padx=10, pady=10)
        
        # Extract button
        self.extract_button = ctk.CTkButton(
            progress_frame,
            text="Start Extraction",
            command=self.start_extraction,
            width=300,
            height=50,
            font=("Helvetica", 16, "bold"),
            fg_color="#28A745",
            hover_color="#218838"
        )
        self.extract_button.grid(row=4, column=0, padx=10, pady=15)
        
        # Initialize empty file list
        self.update_file_list()

    def select_files(self):
        """Open file dialog to select video files"""
        if self.extraction_running:
            return
            
        filetypes = [("Video files", "*.mp4 *.avi *.mkv"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=filetypes
        )
        
        if not files:
            return
            
        # Create job objects for each selected file
        for file_path in files:
            if not any(job.input_path == file_path for job in self.jobs):
                job = ExtractionJob(file_path)
                self.jobs.append(job)
                print(f"Added file: {file_path}")
        
        self.update_file_list()

    def clear_selection(self):
        """Clear the selected files"""
        if self.extraction_running:
            return
            
        self.jobs = []
        self.update_file_list()
        print("Cleared file selection")

    def update_file_list(self):
        """Update the files list display"""
        # Clear existing widgets
        for widget in self.files_list_frame.winfo_children():
            widget.destroy()
            
        if not self.jobs:
            empty_label = ctk.CTkLabel(
                self.files_list_frame,
                text="No files selected",
                font=("Helvetica", 12),
                text_color="gray"
            )
            empty_label.grid(row=0, column=0, pady=20)
            return
            
        # Add header row
        header_frame = ctk.CTkFrame(self.files_list_frame)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        header_frame.grid_columnconfigure((0, 1), weight=1)
        
        ctk.CTkLabel(header_frame, text="File Name", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(header_frame, text="Status", font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
        
        # Add file rows
        for i, job in enumerate(self.jobs):
            row_frame = ctk.CTkFrame(self.files_list_frame)
            row_frame.grid(row=i+1, column=0, sticky="ew", pady=2)
            row_frame.grid_columnconfigure((0, 1), weight=1)
            
            # File name - truncate if too long
            file_name = os.path.basename(job.input_path)
            if len(file_name) > 40:
                file_name = file_name[:37] + "..."
                
            ctk.CTkLabel(row_frame, text=file_name, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            
            # Status and frame count
            status_text = job.status
            if job.frames_extracted > 0:
                status_text = f"{job.status} ({job.frames_extracted} frames)"
                
            status_label = ctk.CTkLabel(row_frame, text=status_text)
            if job.status == "Success":
                status_label.configure(text_color="#28A745")
            elif job.status == "Failed":
                status_label.configure(text_color="#DC3545")
                
            status_label.grid(row=0, column=1, padx=5, pady=5)

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

    def start_extraction(self):
        """Start the frame extraction process"""
        if self.extraction_running:
            print("Extraction already running")
            return
            
        if not self.jobs:
            messagebox.showinfo("No Files", "Please select at least one video file.")
            print("No files selected")
            return
            
        try:
            frame_rate = int(self.frame_rate_var.get())
            if frame_rate < 1:
                raise ValueError("Frame rate must be at least 1")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid frame rate: {str(e)}")
            return
            
        try:
            rotation_angle = int(self.rotation_var.get())
            if rotation_angle not in [0, 90, 180, 270]:
                raise ValueError("Invalid rotation angle")
        except ValueError as e:
            messagebox.showerror("Invalid Input", "Invalid rotation angle")
            return
        
        print("Starting extraction process")
        
        # Update job settings
        for job in self.jobs:
            job.frame_rate = frame_rate
            job.rotation_angle = rotation_angle
        
        # Disable UI controls
        self.extraction_running = True
        self.extract_button.configure(
            text="Extracting...",
            state="disabled",
            fg_color="#6C757D"
        )
        
        # Start extraction thread
        threading.Thread(target=self.extraction_thread, daemon=True).start()

    def extraction_thread(self):
        """Run the frame extraction process in a separate thread"""
        try:
            print("Extraction thread started")
            total_jobs = len(self.jobs)
            completed_jobs = 0
            
            for i, job in enumerate(self.jobs):
                self.current_job_index = i
                file_name = os.path.basename(job.input_path)
                
                # Create output folder next to the video file
                video_dir = os.path.dirname(job.input_path)
                video_name = os.path.splitext(file_name)[0]
                job.output_folder = os.path.join(video_dir, f"{video_name}_frames")
                os.makedirs(job.output_folder, exist_ok=True)
                
                # Update progress
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, f"Extracting frames from {file_name}")
                print(f"Processing job {i+1}/{total_jobs}: {file_name}")
                print(f"Output folder: {job.output_folder}")
                
                # Extract frames
                self.update_status(f"Extracting frames from {file_name}...")
                success, frames_count = video_to_frames(
                    input_path=job.input_path,
                    output_folder=job.output_folder,
                    frame_rate=job.frame_rate,
                    rotation_angle=job.rotation_angle
                )
                
                # Update job status
                job.success = success
                job.frames_extracted = frames_count
                job.status = "Success" if success else "Failed"
                print(f"Frame extraction {'successful' if success else 'failed'}: {frames_count} frames")
                
                # Update UI
                self.update_file_list()
                
                # Mark job as completed
                completed_jobs += 1
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, "Completed")
            
            # All jobs completed
            if all(job.success for job in self.jobs):
                self.update_status("All frames extracted successfully!", True)
            elif any(job.success for job in self.jobs):
                self.update_status("Some extractions completed with errors.", None)
            else:
                self.update_status("All extractions failed!", False)
                
            print("Extraction thread finished")
                
        except Exception as e:
            error_msg = f"Error during extraction: {str(e)}"
            self.update_status(error_msg, False)
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-enable UI controls
            self.extraction_running = False
            self.extract_button.configure(
                text="Start Extraction",
                state="normal",
                fg_color="#28A745",
                hover_color="#218838"
            )
            print("UI controls re-enabled")

def main():
    print("Starting Video Frame Extractor App")
    app = VideoFrameExtractorApp()
    app.mainloop()

if __name__ == "__main__":
    main() 