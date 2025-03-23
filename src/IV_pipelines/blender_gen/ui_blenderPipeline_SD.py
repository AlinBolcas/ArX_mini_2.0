# ui_blenderPipeline_SD.py

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import threading
import os
from pathlib import Path
import sys
import random
import re
import shutil
from PIL import Image

project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

from modules.text_gen.textGen_revamp import TextGen
from modules.threed_gen.threedGen_SD import ThreeDGenSD
from modules.image_gen.imageGen_SD_full import ImageGenerator
from modules.blender_gen.blenderGen import BlenderGen
from modules.arvolve.gifs.gif_from_sequence import gifGen
from modules.utils.utils import quick_look

class BlenderPipelineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blender Pipeline UI")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.text_gen = TextGen(model_provider="OpenAI")
        self.threed_gen = ThreeDGenSD()
        self.image_gen = ImageGenerator()

        # Input frame
        self.input_frame = ctk.CTkFrame(root)
        self.input_frame.pack(pady=10, padx=10, fill="x")

        # Image source selection
        self.image_source_var = tk.StringVar(value="generate")
        self.image_source_label = ctk.CTkLabel(self.input_frame, text="Image Source:")
        self.image_source_label.pack(anchor="w", padx=5, pady=5)
        self.image_source_generate = ctk.CTkRadioButton(self.input_frame, text="Generate Image", variable=self.image_source_var, value="generate", command=self.toggle_input_mode)
        self.image_source_generate.pack(anchor="w", padx=5)
        self.image_source_browse = ctk.CTkRadioButton(self.input_frame, text="Browse Image", variable=self.image_source_var, value="browse", command=self.toggle_input_mode)
        self.image_source_browse.pack(anchor="w", padx=5)

        # Input for image generation
        self.prompt_frame = ctk.CTkFrame(self.input_frame)
        self.prompt_frame.pack(pady=5, fill="x")
        self.input_label = ctk.CTkLabel(self.prompt_frame, text="Prompt:")
        self.input_label.pack(side="left", padx=(0, 10))
        self.input_entry = ctk.CTkEntry(self.prompt_frame, width=300)
        self.input_entry.pack(side="left", padx=(0, 10))
        self.auto_prompt_button = ctk.CTkButton(self.prompt_frame, text="Idea", command=self.auto_generate_prompt)
        self.auto_prompt_button.pack(side="left")

        # Input for image browsing
        self.browse_frame = ctk.CTkFrame(self.input_frame)
        self.browse_button = ctk.CTkButton(self.browse_frame, text="Browse Image", command=self.browse_image)
        self.browse_button.pack(side="left", padx=(0, 10))
        self.image_path_label = ctk.CTkLabel(self.browse_frame, text="No image selected")
        self.image_path_label.pack(side="left")

        # Options
        option_row_frame = ctk.CTkFrame(self.input_frame)
        option_row_frame.pack(pady=5, fill="x")
        self.texture_bool_var = tk.BooleanVar(value=True)  # Set to True by default
        self.texture_bool_checkbox = ctk.CTkCheckBox(option_row_frame, text="Use TEXTURES:", variable=self.texture_bool_var)
        self.texture_bool_checkbox.pack(side="left", padx=(0, 10))

        self.send_button = ctk.CTkButton(self.input_frame, text="Gen", command=self.start_pipeline)
        self.send_button.pack(pady=5)

        # Progress frame
        self.progress_frame = ctk.CTkFrame(root)
        self.progress_frame.pack(pady=10, padx=10, fill="x")

        self.progress_text = tk.Text(self.progress_frame, height=10, state="disabled", bg="#2e2e2e", fg="#f0f0f0")
        self.progress_text.pack(pady=5, padx=5)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.progress_frame)
        self.progress_bar.pack(pady=5, padx=5, fill="x")
        self.progress_bar.set(0)

        # Initialize UI state
        self.toggle_input_mode()

    def toggle_input_mode(self):
        if self.image_source_var.get() == "generate":
            self.prompt_frame.pack(pady=5, fill="x")
            self.browse_frame.pack_forget()
        else:
            self.prompt_frame.pack_forget()
            self.browse_frame.pack(pady=5, fill="x")

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.image_path_label.configure(text=file_path)
        else:
            self.image_path_label.configure(text="No image selected")

    def log_progress(self, message):
        self.progress_text.config(state="normal")
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.config(state="disabled")
        self.progress_text.see(tk.END)

    def start_pipeline(self):
        user_input = self.input_entry.get() if self.image_source_var.get() == "generate" else self.image_path_label.cget("text")
        
        if not user_input:
            # Use tkinter's messagebox instead of CTkMessagebox
            tk.messagebox.showwarning("Input Error", "Please enter a prompt or select an image.")
            return

        # Start the pipeline in a separate thread
        threading.Thread(target=self.run_pipeline, args=(user_input,), daemon=True).start()

    def auto_generate_prompt(self):
        random_topic = self.text_gen.promptGen("Generate a random creative prompt for 3D model generation")
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, random_topic)

    def generate_title(self, prompt):
        # Take the first 5 words of the prompt
        words = prompt.split()[:5]
        base_title = "_".join(words)
        
        # Clean the title
        base_title = re.sub(r'[^\w\s-]', '', base_title.lower())
        base_title = re.sub(r'[-\s]+', '_', base_title)
        
        # Check if the folder already exists and add numbering
        output_dir = project_root / "output" / "threedModels"
        counter = 1
        while True:
            title = f"{base_title}_{counter:03d}"
            if not (output_dir / title).exists():
                return title
            counter += 1

    def resize_image(self, image_path, max_size=1024):
        with Image.open(image_path) as img:
            img.thumbnail((max_size, max_size))
            img.save(image_path)

    def run_pipeline(self, user_input):
        try:
            if self.image_source_var.get() == "generate":
                self.root.after(0, self.log_progress, "Generating prompt...")
                prompt = self.text_gen.promptGen(user_input)
                self.root.after(0, self.log_progress, f"Prompt generated: {prompt}")
                
                title = self.generate_title(prompt)
                save_path = project_root / "output" / "threedModels" / title
                save_path.mkdir(parents=True, exist_ok=True)
                self.root.after(0, self.log_progress, f"Title generated: {title}")
                self.root.after(0, self.log_progress, f"Created output folder: {save_path}")

                self.root.after(0, self.progress_bar.set, 0.25)

                self.root.after(0, self.log_progress, "Generating concept image...")
                image_prompt = self.text_gen.promptGen(f"Create a detailed image prompt based on: {prompt}")
                try:
                    image_path = self.image_gen.generate_image(image_prompt, save_path=save_path / f"{title}_concept.jpg")
                    self.root.after(0, self.log_progress, f"Concept image generated and saved at: {image_path}")
                    quick_look(image_path)
                except Exception as e:
                    self.root.after(0, self.log_progress, f"Error generating concept image: {e}")
                    return
                
                input_for_3d = image_path  # Use the generated image path for 3D generation
            else:
                original_image_path = user_input
                title = self.generate_title(Path(original_image_path).stem)
                save_path = project_root / "output" / "threedModels" / title
                save_path.mkdir(parents=True, exist_ok=True)
                image_path = save_path / f"{title}_input.jpg"
                self.root.after(0, self.log_progress, f"Title generated: {title}")
                self.root.after(0, self.log_progress, f"Created output folder: {save_path}")
                
                # Copy and resize the browsed image to the working folder
                shutil.copy2(original_image_path, image_path)
                self.resize_image(image_path)
                self.root.after(0, self.log_progress, f"Input image copied and resized to: {image_path}")
                
                input_for_3d = str(image_path)  # Use the image path for 3D generation

            self.root.after(0, self.progress_bar.set, 0.5)

            texture_bool = self.texture_bool_var.get()

            self.root.after(0, self.log_progress, "Generating 3D model...")
            try:
                obj_path = self.threed_gen.generate_3d_model(input_for_3d, filename=title, save_dir=save_path)
                self.root.after(0, self.log_progress, f"3D MODEL generated and saved at: {obj_path}")
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error generating 3D model: {e}")
                return  # Stop the pipeline if 3D model generation fails

            self.root.after(0, self.progress_bar.set, 0.75)

            self.root.after(0, self.log_progress, "RENDERING with Blender script...")
            try:
                template_path = str("/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_realTime_01.blend")
                mtl_name = "grey_procedural_MAT"
                output_path = save_path / "renders"
                output_path.mkdir(exist_ok=True)
                height = 1.6

                blender_automation = BlenderGen()
                blender_automation.render_model(template_path, obj_path, mtl_name, texture_bool, str(output_path), height)
                self.root.after(0, self.log_progress, "Blender script completed successfully.")
                
                self.root.after(0, self.log_progress, f"Blender scene saved in the renders folder")
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error running Blender script: {e}")

            self.root.after(0, self.log_progress, "Creating GIF from rendered images...")
            try:
                self.create_gif_from_renders(output_path, save_path, title)
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error creating GIF: {e}")
                return

            self.root.after(0, self.progress_bar.set, 1.0)
        except Exception as e:
            self.root.after(0, self.log_progress, f"An unexpected error occurred: {e}")

    def create_gif_from_renders(self, render_dir, save_dir, title):
        images = []
        for file_name in sorted(os.listdir(render_dir)):
            if file_name.endswith('.png'):
                file_path = os.path.join(render_dir, file_name)
                images.append(file_path)

        gif_path = save_dir / f"{title}.gif"
        gifGen(images, str(gif_path), duration=0.18, resize_factor=100)
        self.root.after(0, self.log_progress, f"GIF created and saved at: {gif_path}")

        quick_look(gif_path)

if __name__ == "__main__":
    root = ctk.CTk()
    app = BlenderPipelineUI(root)
    root.mainloop()