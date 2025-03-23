# ui_blenderPipeline.py

import customtkinter as ctk
import tkinter as tk
import threading
import os
import json
from pathlib import Path
import sys
import subprocess
import torch
import random
from PIL import Image
from tkinter import filedialog


# apple futuristic humanoid droid character, full-body, slick elegant and minimalistic design, highly detailed, innovative organic shapes, 8k

# EXAMPLE PROMPTS:
# complex abstract figurative sculpture symbolising the peak of human evolution and consciousness expansion
# the beginning of everything abstract figurative sculpture
# 

# TO DO:
# ADD AUTO PROMPT BUTTON - to write an Idea
# REFINE PROMPT GENERATION
# IMPLEMENT REFINEMENT TRIPO & other API features
 
#  revamp the ui in stages: theme > batch ideas > image > 3d render
#  add straight input to 3d for specific requests

user_prompt = """
TASK:
Generate a list of {n} topics for 3D model generation. The topics should be reflecting the user themes:
'{input}'

SUBJECTS:
The topics should focus on figurative, sculptural, and design elements.
The subjects should include abstract figurative sculptures, modern abstract sculptures, and character-based themes.
Incorporate mythical beings, personification of emotions, and archetypal characters. 
Use inspiration from famous artists who have been deceased for more than 50 years, ensuring the relevance of their styles to the pieces. 
Include descriptive adjectives and words to create vivid, imaginative prompts. The subjects should be relatively simple to model and refine in 3D, avoiding the need for high accuracy. 
The list should invoke a variety of themes and creative ideas that can be illustrated effectively by an AI model. 
Ensure the list is varied, imaginative, and suitable for 3D modeling. Avoid highly detailed subjects and focus on concepts that can be captured well in abstract or simplified forms.

FORMATTING INSTRUCTIONS:
Write only the list and nothing else. 
Don't number them, just add a new line for each new element. No bullet points, no commas.

FEW EXAMPLES: (don't be constrained by these)
Abstract figurative sculpture inspired by the works of Constantin Brâncuși
Dynamic sculpture of a human form in motion, reminiscent of Boccioni
Geometric abstraction of a human torso in the style of Mondrian
Surrealist sculpture of a dreamer inspired by Dalí
Minimalist figurative sculpture evoking the simplicity of Giacometti
Abstract representation of a thinker, inspired by Rodin
Organic abstract sculpture inspired by the forms of Arp
Figurative sculpture of a reclining figure, reminiscent of Moore
Modern abstract sculpture symbolizing inner strength, inspired by Hepworth
Surrealist abstract sculpture symbolizing the subconscious, inspired by Ernst
Geometric abstraction of a human figure in the style of Malevich
Organic figurative sculpture inspired by the works of Calder
Minimalist abstract representation of an archetype
"""

system_prompt = """
You are an expert in generating creative and diverse topics for 3D model generation, focusing on figurative, sculptural, and design elements. 
"""

def convert_to_python_list(raw_text):
    lines = raw_text.strip().split('\n')
    formatted_list = [line.strip() for line in lines if line.strip()]
    return formatted_list

project_root = Path(__file__).resolve().parents[2]
print("Project root:", project_root, "\n", "File root: ", Path(__file__).resolve())
sys.path.append(str(project_root))

# Add this line to explicitly add the threeD_gen directory to sys.path
sys.path.append(str(project_root / "modules/threeD_gen"))

from modules.text_gen.textGen import TextGen
from modules.threeD_gen.threedGen_API import ThreeDGen as threedGen_API
from modules.threeD_gen.threedGen import ThreedGen as threedGen
from modules.blender_gen.blenderGen import BlenderGen
from modules.image_gen.imageGen import ImageGen
from modules.arvolve.gifs.gif_from_sequence import gifGen
from modules.utils.utils import quick_look

class BlenderPipelineUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Blender Pipeline UI")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        
        self.topics = [
        "Abstract white humanoid figure composed of spheres and cylinders",
        "Minimalist white sculpture representing a warrior using pyramidal forms",
        "Modern abstract sculpture of a thinker using cube and sphere elements",
        "White abstract figure inspired by ancient deities, formed from cylinders and spheres",
        "Geometric abstraction of a reclining figure using only white cubes",
        "Abstract white guardian figure composed of pyramids and spheres",
        "Surrealist white humanoid sculpture formed from intersecting cubes and cylinders",
        "Modern white abstract sculpture of a dancer using flowing cylindrical shapes",
        "Minimalist white sculpture of an embracing couple using spherical and cuboid elements",
        "Abstract representation of strength using white pyramids and cylinders",
        "White abstract figure representing wisdom, composed of intersecting spheres and cubes",
        "Dynamic white sculpture of a runner using cylindrical and spherical forms",
        "Geometric white abstraction of a meditative figure using simple shapes",
        "Surrealist white guardian figure formed from interlocking pyramids and spheres",
        "Minimalist white sculpture of a serene figure using only primary shapes",
        "Abstract white humanoid figure representing courage, formed from cubes and cylinders",
        "Modern white sculpture of a contemplative figure using pyramidal elements",
        "Geometric white abstraction of a reclining figure using spheres and cylinders",
        "Minimalist white figure representing joy, composed of intersecting cubes and spheres",
        "White abstract figure symbolizing peace, using cylindrical and spherical shapes",
        "Dynamic white sculpture of a leaping figure using cuboid elements",
        "Geometric abstraction of a thinker using only white cubes and cylinders",
        "Surrealist white sculpture of a mystical being using spherical forms",
        "Modern white abstract figure representing love, composed of intersecting shapes",
        "Minimalist white sculpture of a dreaming figure using simple geometric forms",
        "Abstract white figure symbolizing hope, using pyramids and cylinders",
        "White abstract representation of an angelic figure using spheres and cubes",
        "Dynamic white sculpture of a swimmer using cylindrical forms",
        "Geometric white abstraction of a guardian figure using pyramidal elements",
        "Minimalist white figure representing freedom, composed of intersecting shapes",
        "Abstract white sculpture of a mythical hero using spheres and cubes",
        "Modern white abstract figure symbolizing balance using primary shapes",
        "Surrealist white sculpture of an ethereal being using cylindrical forms",
        "Minimalist white figure representing strength, using geometric shapes",
        "Abstract white representation of a meditative figure using pyramids and spheres",
        "Dynamic white sculpture of a climber using cylindrical elements",
        "Geometric abstraction of a peaceful figure using only white cubes and spheres",
        "Surrealist white humanoid figure formed from intersecting pyramids and cylinders",
        "Modern white abstract figure representing inner peace using primary shapes",
        "Minimalist white sculpture of an embracing couple using spherical forms",
        "Abstract white figure symbolizing resilience using cubes and cylinders",
        "White abstract representation of a guardian angel using pyramids and spheres",
        "Dynamic white sculpture of an athlete using cylindrical forms",
        "Geometric white abstraction of a reclining figure using primary shapes",
        "Minimalist white figure representing unity using intersecting spheres and cubes",
        "Abstract white sculpture of a mythical creature using cylinders and pyramids",
        "Modern white abstract figure symbolizing wisdom using primary shapes",
        "Surrealist white sculpture of a dreamer using cylindrical elements",
        "Minimalist white figure representing courage using cubes and spheres",
        "Abstract white humanoid figure representing joy using pyramids and cylinders",
        "White abstract sculpture of a contemplative figure using primary shapes",
        "Dynamic white figure representing hope using cylindrical forms",
        "Geometric abstraction of a peaceful figure using spheres and cubes",
        "Surrealist white guardian figure using intersecting pyramids and cylinders",
        "Modern white abstract figure symbolizing love using primary shapes",
        "Minimalist white sculpture of a serene figure using geometric forms",
        "Abstract white representation of a mythical hero using spheres and cubes",
        "White abstract figure symbolizing strength using cylindrical elements",
        "Dynamic white sculpture of a dancer using primary shapes",
        "Geometric white abstraction of a guardian angel using spheres and pyramids",
        "Minimalist white figure representing inner peace using intersecting shapes",
        "Abstract white sculpture of an embracing couple using cubes and cylinders",
        "Modern white abstract figure symbolizing resilience using primary shapes",
        "Surrealist white sculpture of an ethereal being using cylindrical forms",
        "Minimalist white figure representing unity using geometric shapes",
        "Abstract white representation of a peaceful figure using spheres and cubes",
        "Dynamic white sculpture of a swimmer using primary shapes",
        "Geometric white abstraction of a mythical creature using pyramids and cylinders",
        "Surrealist white humanoid figure using intersecting spheres and cubes",
        "Modern white abstract figure symbolizing balance using primary shapes",
        "Minimalist white sculpture of a guardian figure using geometric forms",
        "Abstract white representation of a contemplative figure using cubes and spheres",
        "White abstract figure symbolizing hope using cylindrical elements",
        "Dynamic white sculpture of a leaping figure using primary shapes",
        "Geometric white abstraction of an angelic figure using spheres and pyramids",
        "Minimalist white figure representing strength using intersecting shapes",
        "Abstract white sculpture of a serene figure using primary shapes",
        "Modern white abstract figure symbolizing courage using geometric forms",
        "Surrealist white sculpture of a dreamer using spheres and cubes",
        "Minimalist white figure representing love using cylindrical elements",
        "Abstract white representation of a mythical hero using primary shapes",
        "White abstract figure symbolizing inner peace using cubes and spheres",
        "Dynamic white sculpture of a climber using geometric forms",
        "Geometric white abstraction of a guardian angel using primary shapes",
        "Surrealist white humanoid figure using intersecting cylinders and spheres",
        "Modern white abstract figure symbolizing joy using primary shapes",
        "Minimalist white sculpture of an embracing couple using geometric forms",
        "Abstract white representation of a peaceful figure using cubes and cylinders",
        "White abstract figure symbolizing hope using primary shapes",
        "Dynamic white sculpture of an athlete using geometric forms",
        "Geometric white abstraction of a mythical creature using primary shapes",
        "Surrealist white sculpture of an ethereal being using cubes and spheres"
        ]

        # Input frame
        self.input_frame = ctk.CTkFrame(root)
        self.input_frame.pack(pady=10, padx=10, fill="x")

        input_row_frame = ctk.CTkFrame(self.input_frame)
        input_row_frame.pack(pady=5, fill="x")

        self.input_label = ctk.CTkLabel(input_row_frame, text="Input:")
        self.input_label.pack(side="left", padx=(0, 10))

        self.input_entry = ctk.CTkEntry(input_row_frame, width=300)
        self.input_entry.pack(side="left", padx=(0, 10))

        self.auto_prompt_button = ctk.CTkButton(input_row_frame, text="Idea", command=self.auto_generate_prompt)
        self.auto_prompt_button.pack(side="left")

        option_row_frame = ctk.CTkFrame(self.input_frame)
        option_row_frame.pack(pady=5, fill="x")

        self.use_triposr_var = tk.BooleanVar()
        self.use_triposr_checkbox = ctk.CTkCheckBox(option_row_frame, text="Gen CONCEPT IMAGE:", variable=self.use_triposr_var)
        self.use_triposr_checkbox.pack(side="left", padx=(0, 10))

        self.texture_bool_var = tk.BooleanVar()
        self.texture_bool_checkbox = ctk.CTkCheckBox(option_row_frame, text="Use TEXTURES:", variable=self.texture_bool_var)
        self.texture_bool_checkbox.pack(side="left", padx=(0, 10))

        self.image_gen_model_var = tk.StringVar(value="Dalle")
        self.image_gen_toggle = ctk.CTkSegmentedButton(option_row_frame, values=["Dalle", "SD_API"], variable=self.image_gen_model_var)
        self.image_gen_toggle.pack(side="left", padx=(0, 10))

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

    def log_progress(self, message):
        self.progress_text.config(state="normal")
        self.progress_text.insert(tk.END, message + "\n")
        self.progress_text.config(state="disabled")
        self.progress_text.see(tk.END)

    def start_pipeline(self):
        user_input = self.input_entry.get()
        if not user_input:
            ctk.CTkMessagebox.show_warning("Input Error", "Please enter a topic/request.")
            return

        self.progress_bar.set(0)
        threading.Thread(target=self.run_pipeline, args=(user_input,)).start()
        
    def auto_generate_prompt(self):
        random_topic = random.choice(self.topics)
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, random_topic)

    def run_pipeline(self, user_input):
        self.root.after(0, self.log_progress, "Generating prompt...")
        textGen = TextGen(model_provider="OpenAI", model="gpt-4o", system_prompt="Generate a creative prompt for 3D model generation", max_tokens=300, temperature=0.7)
        prompt = textGen.promptGen(user_input)
        self.root.after(0, self.log_progress, f"Prompt generated: {prompt}")
        
        title = textGen.titleGen(prompt)
        save_path = os.path.join("output/threedModels", title)
        os.makedirs(save_path, exist_ok=True)
        self.root.after(0, self.log_progress, f"Title generated: {title}")
        self.root.after(0, self.log_progress, f"Created output folder: {save_path}")

        self.root.after(0, self.progress_bar.set, 0.33)

        use_triposr = self.use_triposr_var.get()
        texture_bool = self.texture_bool_var.get()
        image_gen_model = self.image_gen_model_var.get()

        self.root.after(0, self.log_progress, "Generating 3D model...")
        if use_triposr:
            threed_gen = threedGen_API()
            imageGen = ImageGen()
            image_filename = f"{title}.jpg"
            image_path = os.path.join(save_path, image_filename)
            prompt_with_background = prompt + "fully framed in shot, on a dark-grey empty background, studio lighting"

            try:
                image_path = imageGen.imageGen_fullPipeline(prompt_with_background, provider=image_gen_model, save_path=image_path)
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error generating image: {e}")
                return

            if not os.path.exists(image_path):
                self.root.after(0, self.log_progress, f"Image generation failed, path does not exist: {image_path}")
                return

            self.root.after(0, self.log_progress, f"CONCEPT IMAGE generated and saved at: {image_path}")
            quick_look(image_path)
            
            try:
                obj_path = os.path.join(save_path, f"{title}.glb")
                threed_gen.threedGen(prompt=prompt, image_path=image_path, save_path=obj_path)
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error generating 3D model: {e}")
                return

            if not obj_path or not os.path.exists(obj_path):
                self.root.after(0, self.log_progress, f"3D model generation failed, path does not exist: {obj_path}")
                return

            self.root.after(0, self.log_progress, f"3D MODEL generated and saved at: {obj_path}")
            
        else:
            threed_gen = threedGen_API()
            obj_path = os.path.join(save_path, f"{title}.glb")
            try:
                threed_gen.threedGen(prompt=prompt, save_path=obj_path)
            except Exception as e:
                self.root.after(0, self.log_progress, f"Error generating 3D model: {e}")
                return

        self.root.after(0, self.log_progress, "RENDERING with Blender script...")
        try:
            template_path = str("/Users/arvolve/3D_Projects/PROJECTS/BlenderGen_Automation_TESTS/templates/blenderGen_quick_01.blend")
            mtl_name = "grey_procedural_MAT"
            output_path = str(project_root / f"output/threedModels/{title}/renders")
            os.makedirs(output_path, exist_ok=True)
            height = 1.6

            blender_automation = BlenderGen()
            blender_automation.render_model(template_path, obj_path, mtl_name, texture_bool, output_path, height)
            self.root.after(0, self.log_progress, "Blender script completed successfully.")
        except Exception as e:
            self.root.after(0, self.log_progress, f"Error running Blender script: {e}")

        self.root.after(0, self.progress_bar.set, 0.66)

        self.root.after(0, self.log_progress, "Creating GIF from rendered images...")
        try:
            self.create_gif_from_renders(output_path, save_path, title)
        except Exception as e:
            self.root.after(0, self.log_progress, f"Error creating GIF: {e}")
            return

        self.root.after(0, self.progress_bar.set, 1.0)

    def create_gif_from_renders(self, render_dir, save_dir, title):
        images = []
        for file_name in sorted(os.listdir(render_dir)):
            if file_name.endswith('.png'):
                file_path = os.path.join(render_dir, file_name)
                images.append(file_path)

        gif_path = os.path.join(save_dir, f"{title}.gif")
        gifGen(images, gif_path, duration=0.18, resize_factor=100)
        self.root.after(0, self.log_progress, f"GIF created and saved at: {gif_path}")

        quick_look(gif_path)

if __name__ == "__main__":
    root = ctk.CTk()
    app = BlenderPipelineUI(root)
    root.mainloop()