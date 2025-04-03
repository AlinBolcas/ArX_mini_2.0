import os
import json
import requests
import time
import logging
from pathlib import Path
from typing import Optional, Literal, Union, Dict, Tuple
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv

# Import utility using direct path instead of FileFinder
from src.VI_utils.utils import quick_look

# Load environment variables
load_dotenv()

class MeshyAPI:
    """
    Wrapper for Meshy's API endpoints including:
    - Text to 3D model generation (preview and refine stages)
    - Model status checking and downloading
    """
    
    def __init__(self):
        """Initialize the API wrapper with API key from environment"""
        self.api_key = os.getenv("MESHY_API_KEY")
        if not self.api_key:
            raise ValueError("MESHY_API_KEY not found in environment variables")
            
        # Base URLs for different API versions
        self.base_url_v1 = "https://api.meshy.ai/openapi/v1"  # For image-to-3D and balance
        self.base_url_v2 = "https://api.meshy.ai/openapi/v2"  # For text-to-3D
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, data: dict, version: int = 2) -> dict:
        """Make API request to Meshy"""
        base_url = self.base_url_v2 if version == 2 else self.base_url_v1
        url = f"{base_url}/{endpoint}"
        
        try:
            self.logger.info(f"Making request to Meshy endpoint: {endpoint}")
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Error response: {e.response.text}")
            raise
            
    def _check_task_status(self, task_id: str, version: int = 2) -> Dict:
        """Check the status of a generation task"""
        base_url = self.base_url_v2 if version == 2 else self.base_url_v1
        endpoint = "text-to-3d" if version == 2 else "image-to-3d"
        url = f"{base_url}/{endpoint}/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Status check failed: {str(e)}")
            raise
            
    def _download_model(self, url: str, output_path: str) -> str:
        """Download the generated model file"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            return output_path
            
        except Exception as e:
            self.logger.error(f"Model download failed: {str(e)}")
            raise

    async def generate_preview(
        self,
        prompt: str,
        art_style: Literal["realistic", "sculpture"] = "realistic",
        seed: Optional[int] = None,
        ai_model: str = "meshy-4",
        topology: Literal["quad", "triangle"] = "triangle",
        target_polycount: Optional[int] = None,
        should_remesh: bool = True,
        symmetry_mode: Literal["off", "auto", "on"] = "auto",
    ) -> str:
        """
        Generate a preview (mesh-only) 3D model from text
        
        Returns:
            str: Task ID for the preview generation
        """
        data = {
            "mode": "preview",
            "prompt": prompt,
            "art_style": art_style,
            "ai_model": ai_model,
            "topology": topology,
            "should_remesh": should_remesh,
            "symmetry_mode": symmetry_mode
        }
        
        # Add optional parameters if provided
        if seed is not None:
            data["seed"] = seed
        if target_polycount is not None:
            data["target_polycount"] = target_polycount
            
        response = self._make_request("text-to-3d", data, version=2)
        return response.get("result")

    async def generate_refine(
        self,
        preview_task_id: str,
        enable_pbr: bool = False
    ) -> str:
        """
        Generate a refined (textured) 3D model from a preview task
        
        Returns:
            str: Task ID for the refine generation
        """
        data = {
            "mode": "refine",
            "preview_task_id": preview_task_id,
            "enable_pbr": enable_pbr
        }
        
        response = self._make_request("text-to-3d", data, version=2)
        return response.get("result")

    def _check_balance(self) -> int:
        """Check current credit balance"""
        url = f"{self.base_url_v1}/balance"  # Balance endpoint is v1
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            balance = response.json().get('balance', 0)
            self.logger.info(f"💰 Current balance: {balance} credits")
            return balance
            
        except Exception as e:
            self.logger.error(f"Failed to check balance: {str(e)}")
            return 0

    async def generate_from_image(
        self,
        image_url: str = None,
        image_path: str = None,
        output_path: Optional[str] = None,
        ai_model: str = "meshy-4",
        topology: Literal["quad", "triangle"] = "triangle",
        target_polycount: Optional[int] = None,
        should_remesh: bool = True,
        should_texture: bool = True,
        enable_pbr: bool = False,
        symmetry_mode: Literal["off", "auto", "on"] = "auto",
    ) -> str:
        """
        Generate 3D model from image input
        
        Args:
            image_url: Direct URL to image or base64 data URI
            image_path: Local path to image file (will be converted to base64)
            output_path: Where to save the generated model
            ai_model: AI model to use (meshy-4 for hard surface)
            topology: Mesh topology type
            target_polycount: Target polygon count
            should_remesh: Whether to remesh the model
            should_texture: Whether to generate textures
            enable_pbr: Whether to generate PBR maps
            symmetry_mode: Symmetry behavior control
            
        Returns:
            str: Path to the downloaded model file
        """
        try:
            # Check initial balance
            initial_balance = self._check_balance()
            
            # Handle image input
            if image_path and not image_url:
                # Convert local image to base64
                with open(image_path, 'rb') as f:
                    import base64
                    image_data = base64.b64encode(f.read()).decode()
                    file_ext = Path(image_path).suffix[1:].lower()
                    image_url = f"data:image/{file_ext};base64,{image_data}"
            elif not image_url:
                raise ValueError("Either image_url or image_path must be provided")

            # Prepare request data
            data = {
                "image_url": image_url,
                "ai_model": ai_model,
                "topology": topology,
                "should_remesh": should_remesh,
                "should_texture": should_texture,
                "enable_pbr": enable_pbr,
                "symmetry_mode": symmetry_mode
            }
            
            # Add optional target_polycount if specified
            if target_polycount is not None:
                data["target_polycount"] = target_polycount

            # Start generation
            self.logger.info("Starting image-to-3D generation...")
            response = self._make_request("image-to-3d", data, version=1)
            task_id = response.get("result")
            
            if not task_id:
                raise Exception("Failed to get task ID")

            # Wait for completion
            with tqdm(total=100, desc="Generating 3D Model", unit="%") as pbar:
                last_progress = 0
                while True:
                    # Use the v1 endpoint for status check
                    status = self._check_image_task_status(task_id)
                    progress = status.get('progress', 0)
                    
                    # Update progress bar
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        
                    if status.get('status') == "FAILED":
                        error = status.get('task_error', {}).get('message', 'Unknown error')
                        raise Exception(f"Generation failed: {error}")
                        
                    elif status.get('status') == "SUCCEEDED":
                        # Get model URL and store it
                        model_url = status.get('model_urls', {}).get('glb')
                        if not model_url:
                            raise Exception("No GLB model URL in response")
                            
                        # If no output path provided, create one
                        if not output_path:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"meshy_img2model_{timestamp}.glb"
                            output_path = str(Path.cwd() / "output" / "threed" / filename)
                            
                        # Download the model
                        self.logger.info("Downloading model...")
                        saved_path = self._download_model(model_url, output_path)
                        self.logger.info(f"✨ Model saved to: {saved_path}")
                        
                        # Return both the URL and saved path
                        return model_url
                        
                    await asyncio.sleep(2)
                    
        except Exception as e:
            self.logger.error(f"Image-to-3D generation failed: {str(e)}")
            raise

    def _check_image_task_status(self, task_id: str) -> Dict:
        """Check the status of an image-to-3D task"""
        url = f"{self.base_url_v1}/image-to-3d/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Status check failed: {str(e)}")
            raise

    async def generate_draft(
        self,
        prompt: str,
        art_style: Literal["realistic", "sculpture"] = "sculpture",
        output_path: Optional[str] = None,
        preview: bool = False,
        **kwargs
    ) -> Tuple[str, str]:
        """
        Generate draft 3D model from text prompt
        
        Args:
            prompt: Text description of desired model
            art_style: Art style to use (default: sculpture)
            output_path: Optional path to save output model
            preview: Whether to show quicklook preview
        """
        try:
            start_time = time.time()
            # Check initial balance
            initial_balance = self._check_balance()
            
            # Generate preview
            self.logger.info("Starting preview generation...")
            preview_id = await self.generate_preview(
                prompt=prompt,
                art_style=art_style,
                seed=None,
                topology="triangle",
                target_polycount=None,
                should_remesh=True,
                symmetry_mode="auto"
            )
            
            if not preview_id:
                raise Exception("Failed to get preview task ID")
                
            # Wait for preview completion
            with tqdm(total=100, desc="Generating Draft", unit="%") as pbar:
                last_progress = 0
                while True:
                    status = self._check_task_status(preview_id)
                    progress = status.get('progress', 0)
                    
                    # Update progress bar
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        
                    if status.get('status') == "FAILED":
                        error = status.get('task_error', {}).get('message', 'Unknown error')
                        raise Exception(f"Draft generation failed: {error}")
                        
                    elif status.get('status') == "SUCCEEDED":
                        # Get model URL
                        model_url = status.get('model_urls', {}).get('glb')
                        if not model_url:
                            raise Exception("No GLB model URL in response")
                            
                        # If no output path provided, create one
                        if not output_path:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"meshy_draft_{timestamp}.glb"
                            output_path = str(Path.cwd() / "output" / "threed" / "drafts" / filename)
                            
                        # Download the model
                        self.logger.info("Downloading draft model...")
                        saved_path = self._download_model(model_url, output_path)
                        self.logger.info(f"✨ Draft saved to: {saved_path}")
                        
                        # Check final balance and calculate cost
                        final_balance = self._check_balance()
                        credits_used = initial_balance - final_balance
                        print(f"💰 Credits used: {credits_used} (Balance: {final_balance})")
                        
                        # Progress tracking only
                        generation_time = time.time() - start_time
                        print(f"⏱️ Draft generation took: {generation_time:.2f}s")
                        
                        if saved_path and Path(saved_path).exists():
                            if preview:
                                await self.preview_model(saved_path)
                            return preview_id, saved_path
                        
                    await asyncio.sleep(2)
                    
            print(f"{printColoured('✗ Draft generation failed', 'red')}")
            return None, None
            
        except Exception as e:
            print(f"Error generating Meshy draft: {str(e)}")
            return None, None

    async def refine_draft(
        self,
        preview_id: str,
        output_path: Optional[str] = None,
        enable_pbr: bool = True,
        preview: bool = True
    ) -> str:
        """
        Refine a draft model with textures
        
        Args:
            preview_id: ID from draft generation
            output_path: Optional path to save output model
            enable_pbr: Enable PBR textures
            preview: Whether to show quicklook preview
        """
        try:
            start_time = time.time()
            # Check initial balance
            initial_balance = self._check_balance()
            
            if not output_path:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = str(Path("output/threedModels") / f"meshy_refined_{timestamp}.glb")

            # Start refinement
            self.logger.info("Starting model refinement...")
            
            # Generate refine
            self.logger.info("Starting refine generation...")
            refine_id = await self.generate_refine(
                preview_task_id=preview_id,
                enable_pbr=enable_pbr
            )
            
            if not refine_id:
                raise Exception("Failed to get refine task ID")
                
            # Wait for refine completion
            with tqdm(total=100, desc="Generating Textures", unit="%") as pbar:
                last_progress = 0
                while True:
                    status = self._check_task_status(refine_id)
                    progress = status.get('progress', 0)
                    
                    # Update progress bar
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        
                    if status.get('status') == "FAILED":
                        error = status.get('task_error', {}).get('message', 'Unknown error')
                        raise Exception(f"Refine generation failed: {error}")
                        
                    elif status.get('status') == "SUCCEEDED":
                        # Get model URL
                        model_url = status.get('model_urls', {}).get('glb')
                        if not model_url:
                            raise Exception("No GLB model URL in response")
                            
                        # Download the model
                        self.logger.info("Downloading refined model...")
                        saved_path = self._download_model(model_url, output_path)
                        self.logger.info(f"✨ Refined model saved to: {saved_path}")
                        
                        # Check final balance and calculate cost
                        final_balance = self._check_balance()
                        credits_used = initial_balance - final_balance
                        print(f"💰 Credits used: {credits_used} (Balance: {final_balance})")
                        
                        if saved_path and Path(saved_path).exists():
                            if preview:
                                await self.preview_model(saved_path)
                            generation_time = time.time() - start_time
                            print(f"⏱️ Refinement took: {generation_time:.2f}s")
                            return saved_path
                        
                    await asyncio.sleep(2)
                    
            print(f"{printColoured('✗ Refinement failed', 'red')}")
            return None
            
        except Exception as e:
            print(f"Error refining draft: {str(e)}")
            return None

    async def regenerate_texture(
        self,
        model_url: str,
        object_prompt: str,
        style_prompt: str,
        output_path: Optional[str] = None,
        enable_original_uv: bool = True,
        enable_pbr: bool = True,
        resolution: Literal["1024", "2048", "4096"] = "2048",
        negative_prompt: Optional[str] = None,
        art_style: Literal["realistic", "fake-3d-cartoon", "japanese-anime", 
                          "cartoon-line-art", "realistic-hand-drawn", 
                          "fake-3d-hand-drawn", "oriental-comic-ink"] = "realistic"
    ) -> str:
        """
        Generate new textures for an existing 3D model using text prompts
        
        Args:
            model_url: URL to the model file (glb/fbx/obj/stl/gltf)
            object_prompt: Description of what the object is
            style_prompt: Description of desired texture style
            enable_original_uv: Use original UV mapping
            enable_pbr: Generate PBR maps
            resolution: Texture resolution
            negative_prompt: What to avoid in textures
            art_style: Style preset to use
            
        Returns:
            str: Path to the retextured model file
        """
        try:
            # Check initial balance
            initial_balance = self._check_balance()
            
            # Prepare request data
            data = {
                "model_url": model_url,
                "object_prompt": object_prompt,
                "style_prompt": style_prompt,
                "enable_original_uv": enable_original_uv,
                "enable_pbr": enable_pbr,
                "resolution": resolution,
                "art_style": art_style
            }
            
            if negative_prompt:
                data["negative_prompt"] = negative_prompt
                
            # Start texture generation
            self.logger.info("Starting texture generation...")
            response = self._make_request("text-to-texture", data, version=1)
            task_id = response.get("result")
            
            if not task_id:
                raise Exception("Failed to get task ID")
                
            # Wait for completion
            with tqdm(total=100, desc="Generating Textures", unit="%") as pbar:
                last_progress = 0
                while True:
                    status = self._check_texture_task_status(task_id)
                    progress = status.get('progress', 0)
                    
                    # Update progress bar
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                        
                    if status.get('status') == "FAILED":
                        error = status.get('task_error', {}).get('message', 'Unknown error')
                        raise Exception(f"Texture generation failed: {error}")
                        
                    elif status.get('status') == "SUCCEEDED":
                        # Get model URL
                        model_url = status.get('model_urls', {}).get('glb')
                        if not model_url:
                            raise Exception("No GLB model URL in response")
                            
                        # If no output path provided, create one
                        if not output_path:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"meshy_retextured_{timestamp}.glb"
                            output_path = str(Path.cwd() / "output" / "threed" / filename)
                            
                        # Download the model
                        self.logger.info("Downloading retextured model...")
                        saved_path = self._download_model(model_url, output_path)
                        self.logger.info(f"✨ Retextured model saved to: {saved_path}")
                        
                        # Check final balance and calculate cost
                        final_balance = self._check_balance()
                        credits_used = initial_balance - final_balance
                        print(f"💰 Credits used: {credits_used} (Balance: {final_balance})")
                        
                        return model_url
                        
                    await asyncio.sleep(2)
                    
        except Exception as e:
            self.logger.error(f"Texture generation failed: {str(e)}")
            raise
            
    def _check_texture_task_status(self, task_id: str) -> Dict:
        """Check the status of a text-to-texture task"""
        url = f"{self.base_url_v1}/text-to-texture/{task_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Status check failed: {str(e)}")
            raise

    async def preview_model(self, model_path: str):
        """Helper to preview 3D models with quicklook."""
        if model_path and Path(model_path).exists():
            print("\n🔍 Opening preview with quicklook...")
            try:
                quick_look(model_path)
                print("✨ Quicklook preview opened successfully")
            except Exception as e:
                print(f"Failed to open quicklook preview: {str(e)}")

    # Remove duplicate preview methods
    async def preview_draft(self, result_path: str):
        """Alias for preview_model for backwards compatibility."""
        await self.preview_model(result_path)

    async def preview_refined(self, result_path: str):
        """Alias for preview_model for backwards compatibility."""
        await self.preview_model(result_path)
