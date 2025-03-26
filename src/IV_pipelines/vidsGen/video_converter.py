#!/usr/bin/env python3
"""
Video Converter
--------------
A simplified GUI application for batch converting MOV files to MP4 and optimized GIFs.

Features:
- Select single or multiple MOV files for conversion
- Convert MOV to MP4 with high quality settings
- Convert videos to optimized GIFs (under 5MB)
- Progress tracking
- Success/failure notifications
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import List, Tuple, Dict, Optional
import customtkinter as ctk
import math
from pathlib import Path

# Import video utilities
# Calculate the project root directory (3 levels up from the script's directory)
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] # Go up vidsGen, IV_pipelines, src
sys.path.insert(0, str(project_root))

# Now try the import assuming 'src' is directly under project_root
try:
    from src.VI_utils.video_utils import convert_mov_to_mp4, video_to_gif, get_video_info
except ImportError as e:
    print(f"ERROR: Could not import video utilities. Make sure video_utils.py is accessible at src/VI_utils/video_utils.py relative to the project root ({project_root}). Error: {e}")
    # Optionally, show an error message box if tkinter is available
    try:
        # Ensure tkinter and messagebox are imported if needed for the error popup
        import tkinter as tk
        from tkinter import messagebox
        # Need to initialize Tk root temporarily for messagebox if app hasn't started
        root = tk.Tk()
        root.withdraw() # Hide the root window
        messagebox.showerror("Import Error", f"Could not import video utilities. Check console for details. Error: {e}")
        root.destroy()
    except ImportError:
        pass # Tkinter not available or failed to import
    sys.exit(1)

# Set appearance mode and default color theme
ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue", "green", "dark-blue"

class ConversionJob:
    """Class to track a single file conversion job"""
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.base_path = os.path.splitext(input_path)[0]
        self.mp4_path = f"{self.base_path}.mp4"
        self.gif_path = f"{self.base_path}.gif"
        self.hifi_gif_path = f"{self.base_path}_hifi.gif"
        self.mp4_status = "Pending"
        self.gif_status = "Pending"
        self.hifi_gif_status = "Pending"
        self.mp4_success = False
        self.gif_success = False
        self.hifi_gif_success = False
        self.gif_attempts = 0
        self.hifi_gif_attempts = 0
        self.gif_size_mb = 0
        self.hifi_gif_size_mb = 0
        self.mp4_size_mb = 0

class VideoConverterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Converter")
        # Increase window size for better visibility
        self.geometry("900x700")
        self.minsize(800, 600)
        
        # Variables
        self.conversion_running = False
        self.jobs: List[ConversionJob] = []
        self.current_job_index = 0
        
        # Fixed conversion settings
        self.MAX_GIF_SIZE_MB = 5.0
        self.MAX_HIFI_GIF_SIZE_MB = 25.0
        
        # New variable for MP4 size limit
        self.mp4_size_limit = ctk.BooleanVar(value=False)
        
        # Create the main layout
        self.create_ui()
        
        # Debug print for initialization
        print("App initialized, UI created")

    def create_ui(self):
        """Create a simplified user interface"""
        # Use a single main frame with grid layout for better space management
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)  # Files list should expand
        
        # === File Selection Section ===
        select_frame = ctk.CTkFrame(main_frame)
        select_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        select_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(select_frame, text="Video Files Selection", font=("Helvetica", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=(5, 10), sticky="w")
        
        info_text = "Converting MOV → MP4 (high quality) and creating two GIFs:"
        ctk.CTkLabel(select_frame, text=info_text, font=("Helvetica", 12)).grid(row=1, column=0, columnspan=2, pady=(0, 5), sticky="w")
        
        gif_details = "• Optimized GIF (<5MB) and High-Fidelity GIF (<25MB)"
        ctk.CTkLabel(select_frame, text=gif_details, font=("Helvetica", 12)).grid(row=2, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        ctk.CTkCheckBox(
            select_frame,
            text="Limit MP4s to 25MB",
            variable=self.mp4_size_limit,
            onvalue=True,
            offvalue=False
        ).grid(row=3, column=0, columnspan=2, pady=(0, 10), sticky="w")
        
        ctk.CTkButton(
            select_frame, 
            text="Select MOV Files", 
            command=self.select_files,
            width=200,
            height=40,
            font=("Helvetica", 14)
        ).grid(row=4, column=0, padx=(10, 5), pady=5, sticky="w")
        
        ctk.CTkButton(
            select_frame, 
            text="Clear Selection", 
            command=self.clear_selection,
            width=200,
            height=40,
            fg_color="#D35B58",
            hover_color="#C84744",
            font=("Helvetica", 14)
        ).grid(row=4, column=1, padx=(5, 10), pady=5, sticky="w")
        
        # === Files List Section ===
        files_frame = ctk.CTkFrame(main_frame)
        files_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        files_frame.grid_columnconfigure(0, weight=1)
        files_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(files_frame, text="Selected Files", font=("Helvetica", 14, "bold")).grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        # Improved scrollable file list
        file_list_container = ctk.CTkScrollableFrame(files_frame)
        file_list_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        file_list_container.grid_columnconfigure(0, weight=1)
        
        # Store the container for file listings
        self.files_list_frame = file_list_container
        
        # === Progress Section ===
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        progress_frame.grid_columnconfigure(0, weight=1)
        
        # Simplified progress UI - only one progress bar
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
            text="Select MOV files to convert", 
            font=("Helvetica", 14),
            height=30
        )
        self.status_message.grid(row=0, column=0, padx=10, pady=10)
        
        # Convert button - made larger and more prominent
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
        
        # Initialize empty file list
        self.update_file_list()
    
    def select_files(self):
        """Open file dialog to select MOV files"""
        if self.conversion_running:
            return
            
        filetypes = [("MOV files", "*.mov"), ("All Video Files", "*.mov *.mp4 *.avi *.mkv"), ("All files", "*.*")]
        files = filedialog.askopenfilenames(
            title="Select MOV Files",
            filetypes=filetypes
        )
        
        if not files:
            return
            
        # Create job objects for each selected file
        for file_path in files:
            # Only add files that are not already in the job list
            if not any(job.input_path == file_path for job in self.jobs):
                self.jobs.append(ConversionJob(file_path))
                print(f"Added file: {file_path}")
        
        self.update_file_list()
    
    def clear_selection(self):
        """Clear the selected files"""
        if self.conversion_running:
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
        header_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)
        
        ctk.CTkLabel(header_frame, text="File Name", font=("Helvetica", 12, "bold")).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkLabel(header_frame, text="MP4", font=("Helvetica", 12, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ctk.CTkLabel(header_frame, text="Optimized GIF", font=("Helvetica", 12, "bold")).grid(row=0, column=2, padx=5, pady=5)
        ctk.CTkLabel(header_frame, text="Hi-Fi GIF", font=("Helvetica", 12, "bold")).grid(row=0, column=3, padx=5, pady=5)
        
        # Add file rows
        for i, job in enumerate(self.jobs):
            row_frame = ctk.CTkFrame(self.files_list_frame)
            row_frame.grid(row=i+1, column=0, sticky="ew", pady=2)
            row_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
            
            # File name - truncate if too long
            file_name = os.path.basename(job.input_path)
            if len(file_name) > 40:
                file_name = file_name[:37] + "..."
                
            ctk.CTkLabel(row_frame, text=file_name, anchor="w").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            
            # MP4 Status
            if self.mp4_size_limit.get() and job.mp4_size_mb > 0:
                mp4_text = f"{job.mp4_status} ({job.mp4_size_mb:.1f}MB)"
            else:
                mp4_text = job.mp4_status
                
            mp4_label = ctk.CTkLabel(row_frame, text=mp4_text)
            if job.mp4_status == "Success":
                mp4_label.configure(text_color="#28A745")
            elif job.mp4_status == "Failed":
                mp4_label.configure(text_color="#DC3545")
                
            mp4_label.grid(row=0, column=1, padx=5, pady=5)
            
            # Optimized GIF status and size
            if job.gif_size_mb > 0:
                gif_text = f"{job.gif_status} ({job.gif_size_mb:.1f}MB)"
            else:
                gif_text = job.gif_status
                
            gif_label = ctk.CTkLabel(row_frame, text=gif_text)
            if job.gif_status == "Success":
                gif_label.configure(text_color="#28A745")
            elif job.gif_status == "Failed":
                gif_label.configure(text_color="#DC3545")
                
            gif_label.grid(row=0, column=2, padx=5, pady=5)
            
            # High-Fidelity GIF status and size
            if job.hifi_gif_size_mb > 0:
                hifi_text = f"{job.hifi_gif_status} ({job.hifi_gif_size_mb:.1f}MB)"
            else:
                hifi_text = job.hifi_gif_status
                
            hifi_label = ctk.CTkLabel(row_frame, text=hifi_text)
            if job.hifi_gif_status == "Success":
                hifi_label.configure(text_color="#28A745")
            elif job.hifi_gif_status == "Failed":
                hifi_label.configure(text_color="#DC3545")
                
            hifi_label.grid(row=0, column=3, padx=5, pady=5)
    
    def update_progress(self, progress: float, task_description: str):
        """Update the progress bar and task description"""
        self.progress_bar.set(progress)
        self.task_label.configure(text=task_description)
        
        # Force UI update
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
            # Use system default colors
            self.status_frame.configure(fg_color="transparent")
            self.status_message.configure(
                text_color=ctk.ThemeManager.theme["CTkLabel"]["text_color"]
            )
            
        # Force UI update
        self.update_idletasks()
        
        # Debug print for status updates
        print(f"Status update: {message}")
    
    def start_conversion(self):
        """Start the conversion process in a separate thread"""
        if self.conversion_running:
            print("Conversion already running, ignoring button click")
            return
            
        if not self.jobs:
            messagebox.showinfo("No Files", "Please select at least one MOV file to convert.")
            print("No files selected, conversion canceled")
            return
        
        print("Starting conversion process")
        
        # Disable UI controls during conversion
        self.conversion_running = True
        self.convert_button.configure(
            text="Converting...", 
            state="disabled",
            fg_color="#6C757D"
        )
        
        # Start conversion thread
        threading.Thread(target=self.conversion_thread, daemon=True).start()
    
    def conversion_thread(self):
        """Run the conversion process in a separate thread"""
        try:
            print("Conversion thread started")
            total_jobs = len(self.jobs)
            completed_jobs = 0
            
            for i, job in enumerate(self.jobs):
                self.current_job_index = i
                file_name = os.path.basename(job.input_path)
                
                # Update progress for this job
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, f"Converting {file_name}")
                print(f"Processing job {i+1}/{total_jobs}: {file_name}")
                
                # Convert to MP4
                self.update_status(f"Converting {file_name} to MP4...")
                mp4_success = self.convert_to_mp4(job)
                
                # Update job status
                job.mp4_success = mp4_success
                job.mp4_status = "Success" if mp4_success else "Failed"
                print(f"MP4 conversion {'successful' if mp4_success else 'failed'}")
                
                # Update UI with job status
                self.update_file_list()
                
                if mp4_success:
                    # Create high-fidelity GIF first
                    self.update_status(f"Creating high-fidelity GIF for {file_name}...")
                    hifi_success = self.create_hifi_gif(job)
                    
                    # Update job status
                    job.hifi_gif_success = hifi_success
                    job.hifi_gif_status = "Success" if hifi_success else "Failed"
                    print(f"High-fidelity GIF creation {'successful' if hifi_success else 'failed'}")
                    
                    # Create optimized GIF
                    self.update_status(f"Creating optimized GIF for {file_name}...")
                    gif_success = self.create_optimized_gif(job)
                    
                    # Update job status
                    job.gif_success = gif_success
                    job.gif_status = "Success" if gif_success else "Failed"
                    print(f"Optimized GIF creation {'successful' if gif_success else 'failed'}")
                    
                    # Update UI with job status
                    self.update_file_list()
                else:
                    # Skip GIF conversions if MP4 failed
                    job.gif_status = "Skipped"
                    job.hifi_gif_status = "Skipped"
                    self.update_file_list()
                
                # Mark job as completed
                completed_jobs += 1
                progress = completed_jobs / total_jobs if total_jobs > 0 else 0
                self.update_progress(progress, "Completed")
            
            # All jobs completed
            if all(job.mp4_success and job.gif_success and job.hifi_gif_success for job in self.jobs):
                self.update_status("All conversions completed successfully!", True)
            elif any(job.mp4_success or job.gif_success or job.hifi_gif_success for job in self.jobs):
                self.update_status("Some conversions completed with warnings or errors.", None)
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
    
    def convert_to_mp4(self, job: ConversionJob) -> bool:
        """Convert MOV to MP4 with high quality and preserve audio"""
        try:
            max_attempts = 3 if self.mp4_size_limit.get() else 1
            current_attempt = 0
            qualities = ["high", "medium", "low"]  # Quality presets
            
            while current_attempt < max_attempts:
                current_attempt += 1
                quality = qualities[current_attempt-1]
                
                result = convert_mov_to_mp4(
                    input_path=job.input_path,
                    output_path=job.mp4_path,
                    quality=quality,
                    preserve_audio=True
                )
                
                if result and os.path.exists(job.mp4_path):
                    job.mp4_size_mb = os.path.getsize(job.mp4_path) / (1024 * 1024)
                    
                    # Check size if limit enabled
                    if not self.mp4_size_limit.get():
                        return True
                    elif job.mp4_size_mb <= 25:
                        return True
                    elif current_attempt == max_attempts:
                        print(f"MP4 size {job.mp4_size_mb:.2f}MB exceeds 25MB limit")
                        return False
                
            return False
        except Exception as e:
            error_msg = f"Error converting {job.input_path} to MP4: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Error converting to MP4: {str(e)}", False)
            return False
    
    def create_hifi_gif(self, job: ConversionJob) -> bool:
        """Create a high-fidelity GIF with original resolution (max 25MB)"""
        try:
            print(f"Creating high-fidelity GIF: {job.hifi_gif_path}")
            max_attempts = 3
            current_attempt = 0
            
            # Modified settings:
            resize_factor = 90.0  # Start at 90% instead of 100%
            fps = 8  # Reduced from 10
            quality = 100
            max_colors = 256
            
            success = False
            
            # Check if source file exists
            if not os.path.exists(job.mp4_path):
                print(f"ERROR: MP4 file does not exist: {job.mp4_path}")
                self.update_status(f"Error: MP4 file not found", False)
                return False
            
            while current_attempt < max_attempts and not success:
                current_attempt += 1
                job.hifi_gif_attempts = current_attempt
                
                print(f"Hi-Fi GIF attempt {current_attempt}/{max_attempts}")
                print(f"Settings: FPS={fps}, Quality={quality}, Resize={resize_factor}%, Colors={max_colors}")
                
                # Update progress
                self.update_progress(
                    0.4,
                    f"Creating hi-fi GIF (attempt {current_attempt}/{max_attempts})"
                )
                
                # In retry attempts:
                if current_attempt == 2:
                    fps = 6
                    resize_factor = 80
                elif current_attempt == 3:
                    fps = 4
                    resize_factor = 70
                
                # Calculate frame sampling based on FPS
                video_info = get_video_info(job.mp4_path)
                original_fps = video_info.get('fps', 30)
                sample_every = max(1, int(original_fps / fps))
                print(f"Original FPS: {original_fps}, Sampling every {sample_every} frame(s)")
                
                # Create high-fidelity GIF
                result = video_to_gif(
                    input_path=job.mp4_path,
                    output_path=job.hifi_gif_path,
                    fps=fps,
                    resize_factor=resize_factor,
                    quality=quality, 
                    optimize_size=True,
                    sample_every=sample_every,
                    max_colors=max_colors
                )
                
                if result:
                    # Check if output file exists
                    if not os.path.exists(job.hifi_gif_path):
                        print(f"ERROR: Hi-Fi GIF not created: {job.hifi_gif_path}")
                        continue
                        
                    # Check file size
                    gif_size = os.path.getsize(job.hifi_gif_path) / (1024 * 1024)  # MB
                    job.hifi_gif_size_mb = gif_size
                    print(f"Hi-Fi GIF created: {job.hifi_gif_path} ({gif_size:.2f} MB)")
                    
                    if gif_size <= self.MAX_HIFI_GIF_SIZE_MB:
                        # Success - file is under the size limit
                        success = True
                        print(f"Hi-Fi GIF is under size limit of {self.MAX_HIFI_GIF_SIZE_MB} MB")
                        break
                    elif current_attempt >= max_attempts:
                        # Final attempt, accept whatever size we got
                        success = True
                        print(f"WARNING: Hi-Fi GIF exceeds size limit ({gif_size:.2f}MB > {self.MAX_HIFI_GIF_SIZE_MB}MB)")
                        self.update_status(f"Warning: Hi-Fi GIF is {gif_size:.2f}MB, exceeding the {self.MAX_HIFI_GIF_SIZE_MB}MB target.")
                else:
                    # Conversion failed
                    print("Hi-Fi GIF creation failed")
                    break
            
            return success
            
        except Exception as e:
            error_msg = f"Error creating hi-fi GIF: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Error creating hi-fi GIF: {str(e)}", False)
            return False
    
    def create_optimized_gif(self, job: ConversionJob) -> bool:
        """Create an optimized GIF under 5MB using progressive optimization"""
        try:
            print(f"Creating optimized GIF: {job.gif_path}")
            max_attempts = 5
            current_attempt = 0
            
            # More aggressive starting point:
            resize_factor = 40.0  # Reduced from 50%
            fps = 4  # Reduced from 5
            quality = 100
            max_colors = 256
            
            success = False
            
            # Check if source file exists
            if not os.path.exists(job.mp4_path):
                print(f"ERROR: MP4 file does not exist: {job.mp4_path}")
                self.update_status(f"Error: MP4 file not found", False)
                return False
            
            while current_attempt < max_attempts and not success:
                current_attempt += 1
                job.gif_attempts = current_attempt
                
                print(f"Optimized GIF attempt {current_attempt}/{max_attempts}")
                print(f"Settings: FPS={fps}, Quality={quality}, Resize={resize_factor}%, Colors={max_colors}")
                
                # Update progress
                self.update_progress(
                    0.7,
                    f"Creating optimized GIF (attempt {current_attempt}/{max_attempts})"
                )
                
                # In retry attempts:
                if current_attempt == 2:
                    fps = 3
                    resize_factor = 30
                elif current_attempt == 3:
                    fps = 2
                    resize_factor = 25
                elif current_attempt == 4:
                    fps = 1
                    resize_factor = 20
                
                # Calculate frame sampling based on FPS
                video_info = get_video_info(job.mp4_path)
                original_fps = video_info.get('fps', 30)
                sample_every = max(1, int(original_fps / fps))
                print(f"Original FPS: {original_fps}, Sampling every {sample_every} frame(s)")
                
                # Create optimized GIF
                result = video_to_gif(
                    input_path=job.mp4_path,
                    output_path=job.gif_path,
                    fps=fps,
                    resize_factor=resize_factor,
                    quality=quality,
                    optimize_size=True,
                    sample_every=sample_every,
                    max_colors=max_colors
                )
                
                if result:
                    # Check if output file exists
                    if not os.path.exists(job.gif_path):
                        print(f"ERROR: Optimized GIF not created: {job.gif_path}")
                        continue
                        
                    # Check file size
                    gif_size = os.path.getsize(job.gif_path) / (1024 * 1024)  # MB
                    job.gif_size_mb = gif_size
                    print(f"Optimized GIF created: {job.gif_path} ({gif_size:.2f} MB)")
                    
                    if gif_size <= self.MAX_GIF_SIZE_MB:
                        # Success - file is under the size limit
                        success = True
                        print(f"Optimized GIF is under size limit of {self.MAX_GIF_SIZE_MB} MB")
                        break
                    elif current_attempt >= max_attempts:
                        # Final attempt, accept whatever size we got
                        success = True
                        print(f"WARNING: Optimized GIF exceeds size limit ({gif_size:.2f}MB > {self.MAX_GIF_SIZE_MB}MB)")
                        self.update_status(f"Warning: GIF is {gif_size:.2f}MB, exceeding the {self.MAX_GIF_SIZE_MB}MB target.")
                else:
                    # Conversion failed
                    print("Optimized GIF creation failed")
                    break
            
            return success
            
        except Exception as e:
            error_msg = f"Error creating optimized GIF: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            self.update_status(f"Error creating optimized GIF: {str(e)}", False)
            return False

def main():
    print("Starting Video Converter App")
    app = VideoConverterApp()
    app.mainloop()

if __name__ == "__main__":
    main() 