import os
import sys
import time
import uuid
import asyncio
import tempfile
import replicate
import base64
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from tqdm import tqdm

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
# Navigate up to the project root (3 levels up from src/IV_pipelines/assetsGen/)
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our API wrappers
from src.I_integrations.tripo_API import TripoAPI
from src.I_integrations.replicate_API import ReplicateAPI

# Import utility modules
from src.VI_utils.utils import quick_look

class ThreedGen:
    """
    ThreedGen: A simplified tool for generating 3D models from images
    
    Features:
    1. Generate 3D models from images using TripoAPI
    2. Generate 3D models from images using Replicate's Hunyuan3D
    3. Generate 3D models from images using Replicate's Trellis
    """
    
    def __init__(
        self,
        output_dir: str = "threed_models",
    ):
        # Initialize API clients
        self.tripo = TripoAPI()
        self.replicate = ReplicateAPI()
        
        # Settings
        self.output_dir = output_dir
        
        # Set up output dirs
        self._setup_output_dirs()
        
        # Project state storage
        self.project_id = self._generate_project_id()
        self.state = {
            "project_id": self.project_id,
            "input_image": "",
            "image_url": "",
            "output_models": [],
            "prompt": ""
        }
    
    def _generate_project_id(self) -> str:
        """Generate a unique project ID combining timestamp and UUID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def _setup_output_dirs(self):
        """Set up output directory structure"""
        # Base output directory 
        self.output_base_dir = Path("data/output").resolve()
        self.full_output_dir = self.output_base_dir / self.output_dir
        
        # Create base directory if it doesn't exist
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            
        # Create assets directory if it doesn't exist
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"Output will be stored in {self.full_output_dir}")
    
    def _create_project_dir(self):
        """Create a unique directory for this project"""
        self.project_dir = self.full_output_dir / self.project_id
        if not self.project_dir.exists():
            self.project_dir.mkdir(parents=True, exist_ok=True)
        return self.project_dir
    
    def set_input_image(self, image_path: str):
        """Set the input image for 3D model generation"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.state["input_image"] = str(Path(image_path).resolve())  # Use absolute path
        print(f"Input image set to: {self.state['input_image']}")
        return self.state["input_image"]
    
    def set_prompt(self, prompt: str):
        """Set the prompt for 3D model generation"""
        self.state["prompt"] = prompt
        print(f"Prompt set to: {prompt}")
        return prompt
    
    async def generate_3d_tripo(self, texture_quality: str = "detailed", pbr: bool = True) -> str:
        """Generate 3D model using TripoAPI"""
        if not self.state["input_image"]:
            raise ValueError("Input image not set. Call set_input_image() first.")
            
        input_image = self.state["input_image"]
        prompt = self.state["prompt"] or "3D model of the object in the image"
        
        # Create output path for 3D model
        project_dir = self._create_project_dir()
        model_filename = f"tripo_model_{Path(input_image).stem}_{self.project_id}.glb"
        model_path = str(project_dir / model_filename)
        
        print(f"\n=== GENERATING 3D MODEL USING TRIPO ===")
        print(f"Prompt: {prompt}")
        print(f"Input image: {input_image}")
        
        try:
            # Generate the model
            start_time = time.time()
            generated_model = await self.tripo.generate_threed(
                prompt=prompt,
                image_path=input_image,
                output_path=model_path,
                texture_quality=texture_quality,
                pbr=pbr,
                auto_size=True
            )
            
            if not generated_model:
                print("❌ Failed to generate 3D model with TripoAPI")
                return None
                
            generation_time = time.time() - start_time
            print(f"✅ Generated 3D model: {generated_model}")
            print(f"⏱️ Generation took: {generation_time:.2f}s")
            
            # Store the output model path
            self.state["output_models"].append({
                "method": "tripo",
                "path": generated_model,
                "time": generation_time
            })
            
            # Preview the model
            await self.preview_model(generated_model)
            
            return generated_model
            
        except Exception as e:
            print(f"❌ Error generating 3D model with TripoAPI: {e}")
            return None
    
    async def generate_3d_hunyuan(self, remove_background: bool = True) -> str:
        """Generate 3D model using Replicate's Hunyuan3D"""
        if not self.state["input_image"]:
            raise ValueError("Input image not set. Call set_input_image() first.")
            
        input_image = self.state["input_image"]
        
        # Create output path for 3D model
        project_dir = self._create_project_dir()
        model_filename = f"hunyuan_model_{Path(input_image).stem}_{self.project_id}.glb"
        model_path = str(project_dir / model_filename)
        
        print(f"\n=== GENERATING 3D MODEL USING HUNYUAN3D ===")
        print(f"Input image: {input_image}")
        
        try:
            # Generate the model
            start_time = time.time()
            
            # Use Hunyuan3D to generate the model
            print("Generating 3D model...")
            generated_model_url = self.replicate.generate_threed(
                image_url=input_image,  # Pass the file path directly
                model="hunyuan3d",
                steps=50,
                guidance_scale=5.5,
                octree_resolution=256,
                remove_background=remove_background
            )
            
            if not generated_model_url:
                print("❌ Failed to generate 3D model with Hunyuan3D")
                return None
            
            # Download the model
            print("Downloading model...")
            downloaded_path = self.replicate.download_file(
                generated_model_url,
                output_dir=str(project_dir),
                filename=model_filename
            )
            
            if not downloaded_path:
                print("❌ Failed to download model")
                return None
                
            generation_time = time.time() - start_time
            print(f"✅ Generated 3D model: {downloaded_path}")
            print(f"⏱️ Generation took: {generation_time:.2f}s")
            
            # Store the output model path
            self.state["output_models"].append({
                "method": "hunyuan3d",
                "path": downloaded_path,
                "time": generation_time
            })
            
            # Preview the model
            await self.preview_model(downloaded_path)
            
            return downloaded_path
            
        except Exception as e:
            print(f"❌ Error generating 3D model with Hunyuan3D: {e}")
            return None
    
    async def generate_3d_trellis(self, enable_pbr: bool = True) -> str:
        """Generate 3D model using Replicate's Trellis"""
        if not self.state["input_image"]:
            raise ValueError("Input image not set. Call set_input_image() first.")
            
        input_image = self.state["input_image"]
        
        # Create output path for 3D model
        project_dir = self._create_project_dir()
        model_filename = f"trellis_model_{Path(input_image).stem}_{self.project_id}.glb"
        model_path = str(project_dir / model_filename)
        
        print(f"\n=== GENERATING 3D MODEL USING TRELLIS ===")
        print(f"Input image: {input_image}")
        
        try:
            # Generate the model
            start_time = time.time()
            
            # Use Trellis to generate the model
            print("Generating 3D model...")
            generated_model_url = self.replicate.generate_threed(
                image_url=input_image,  # Pass the file path directly
                model="trellis",
                texture_size=1024,
                mesh_simplify=0.9,
                generate_color=True,
                generate_normal=True,
                randomize_seed=True,
                ss_sampling_steps=38,
                slat_sampling_steps=12,
                ss_guidance_strength=7.5,
                slat_guidance_strength=3
            )
            
            if not generated_model_url:
                print("❌ Failed to generate 3D model with Trellis")
                return None
            
            # Download the model
            print("Downloading model...")
            downloaded_path = self.replicate.download_file(
                generated_model_url,
                output_dir=str(project_dir),
                filename=model_filename
            )
            
            if not downloaded_path:
                print("❌ Failed to download model")
                return None
                
            generation_time = time.time() - start_time
            print(f"✅ Generated 3D model: {downloaded_path}")
            print(f"⏱️ Generation took: {generation_time:.2f}s")
            
            # Store the output model path
            self.state["output_models"].append({
                "method": "trellis",
                "path": downloaded_path,
                "time": generation_time
            })
            
            # Preview the model
            await self.preview_model(downloaded_path)
            
            return downloaded_path
            
        except Exception as e:
            print(f"❌ Error generating 3D model with Trellis: {e}")
            return None
    
    async def preview_model(self, model_path: str):
        """Preview a 3D model file"""
        if not model_path or not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
            
        try:
            print(f"\n🔍 Previewing 3D model: {model_path}")
            quick_look(model_path)
            return True
        except Exception as e:
            print(f"Failed to preview model: {e}")
            return False
            
    def print_results(self):
        """Print a summary of generated models"""
        if not self.state["output_models"]:
            print("\n❌ No models were generated")
            return
            
        print("\n=== GENERATED 3D MODELS ===")
        print(f"Input image: {self.state['input_image']}")
        print(f"Prompt: {self.state['prompt'] or 'No prompt provided'}")
        print("\nModels:")
        
        for i, model in enumerate(self.state["output_models"]):
            print(f"\n{i+1}. Method: {model['method']}")
            print(f"   Path: {model['path']}")
            print(f"   Generation time: {model['time']:.2f} seconds")


async def main():
    """Main function for interactive 3D model generation"""
    print("\n===== 3D MODEL GENERATION TOOL =====")
    generator = None
    
    try:
        # Initialize the generator
        generator = ThreedGen(output_dir="threed_models")
        
        # Get input image path or use default
        default_image = str(project_root / "data/output/src_for_sculptures/headless.png")
        
        if os.path.exists(default_image):
            print(f"\nDefault image: {default_image}")
            use_default = input("Use the default image? (y/n, default: y): ").lower().strip() in ['y', 'yes', '']
            
            if use_default:
                image_path = default_image
            else:
                image_path = input("Enter the path to the input image: ").strip()
        else:
            print("Default image not found.")
            image_path = input("Enter the path to the input image: ").strip()
        
        # Set the input image
        generator.set_input_image(image_path)
        
        # Get prompt for 3D generation
        prompt = input("\nEnter a prompt describing the object (optional): ").strip()
        if prompt:
            generator.set_prompt(prompt)
        
        # Let user choose 3D generation method(s)
        print("\nAvailable 3D generation methods:")
        print("1. TripoAPI - Fast, good for simple objects")
        print("2. Hunyuan3D - Good for organic shapes and sculptures")
        print("3. Trellis - Best for detailed textures and materials")
        print("4. All methods (compare results)")
        
        choice = input("\nChoose a method (1-4): ").strip()
        
        if choice == "1":
            await generator.generate_3d_tripo()
        elif choice == "2":
            await generator.generate_3d_hunyuan()
        elif choice == "3":
            await generator.generate_3d_trellis()
        elif choice == "4":
            print("\nGenerating models with all methods for comparison...")
            tasks = [
                generator.generate_3d_tripo(),
                generator.generate_3d_hunyuan(),
                generator.generate_3d_trellis()
            ]
            # Run sequentially to avoid API rate limits and resource contention
            for task in tasks:
                await task
        else:
            print("Invalid choice. Exiting.")
            return
        
        # Print results summary
        generator.print_results()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if generator:
            print("\n=== GENERATION COMPLETE ===")
            if generator.state["output_models"]:
                print(f"All models saved to: {generator.project_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 