import os, json
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import time
import logging
from pathlib import Path
from typing import Optional, Literal, Union, Dict
import asyncio
from tqdm import tqdm

# Load the .env file
load_dotenv()

class TripoAPI:
    """
    Wrapper for Tripo3D's API endpoints including:
    - Text to 3D model generation
    - Image to 3D model generation
    - Model status checking and downloading
    """
    
    def __init__(self):
        """Initialize the API wrapper with API key from environment"""
        self.api_key = os.getenv("TRIPO_API_KEY")
        if not self.api_key:
            raise ValueError("TRIPO_API_KEY not found in environment variables")
            
        self.base_url = "https://api.tripo3d.ai/v2/openapi"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _make_request(self, endpoint: str, data: dict) -> dict:
        """Make API request to Tripo3D"""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Reduced logs:  
            self.logger.info(f"Posting to Tripo3D endpoint: {endpoint}")
            response = requests.post(url, headers=self.headers, json=data)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                self.logger.error(f"Error response: {e.response.text}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {str(e)}")
            raise
            
    def _check_task_status(self, task_id: str) -> Dict:
        """Check the status of a generation task"""
        url = f"{self.base_url}/task/{task_id}"
        
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

    def _upload_image(self, image_path: str) -> str:
        """Upload image to Tripo3D and get image token"""
        url = f"{self.base_url}/upload"
        
        try:
            # Prepare file upload
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                # Remove Content-Type from headers for multipart upload
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                self.logger.info(f"Uploading image: {image_path}")
                response = requests.post(url, headers=headers, files=files)
                response.raise_for_status()
                
                result = response.json()
                image_token = result.get('data', {}).get('image_token')
                
                if not image_token:
                    raise Exception("No image token received in upload response")
                    
                return image_token
                
        except Exception as e:
            self.logger.error(f"Image upload failed: {str(e)}")
            raise

    async def generate_threed(
        self,
        prompt: Optional[str] = None,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        model_version: str = "v2.0-20240919",
        output_path: Optional[str] = None,
        model_seed: Optional[int] = None,
        texture_seed: Optional[int] = None,
        face_limit: Optional[int] = None,
        texture: bool = True,
        pbr: bool = True,
        texture_quality: Literal["standard", "detailed"] = "detailed",
        texture_alignment: Optional[Literal["original_image", "geometry"]] = None,
        auto_size: bool = True,
        style: Optional[str] = None,
        orientation: Optional[Literal["default", "align_image"]] = None,
        pose: Optional[str] = None,
        negative_prompt: Optional[str] = None
    ) -> str:
        """
        Generate 3D model from text prompt or image
        
        Args:
            prompt: Text prompt for text-to-3D generation
            image_path: Path to input image for image-to-3D generation
            image_url: URL to input image for image-to-3D generation
            model_version: Model version to use
            output_path: Path to save the generated model
            model_seed: Random seed for model generation
            texture_seed: Random seed for texture generation
            face_limit: Maximum number of faces in output model
            texture: Enable texturing
            pbr: Enable PBR materials
            texture_quality: Texture quality level
            texture_alignment: Texture alignment priority
            auto_size: Auto-scale model to real-world dimensions
            style: Optional style preset
            orientation: Model orientation mode
            pose: Optional pose specification (e.g., "T-pose:1:1:1:1:9")
            negative_prompt: Optional negative prompt for text-to-3D
            
        Returns:
            str: Path to saved 3D model file
        """
        # Validate inputs
        if not prompt and not image_path and not image_url:
            raise ValueError("Either prompt, image_path, or image_url must be provided")
            
        # Handle FileOutput objects (from Replicate API responses)
        if image_url is not None:
            # Convert FileOutput or other objects to string
            if hasattr(image_url, 'url'):
                image_url = image_url.url
            elif not isinstance(image_url, str):
                # Try to get string representation
                image_url = str(image_url)
                if not image_url.startswith(('http://', 'https://')):
                    raise ValueError(f"Invalid image URL: {image_url}. Must be a URL string.")
            
            self.logger.info(f"Using image URL: {image_url[:60]}...")
        
        # Prepare request data
        data = {
            "type": "text_to_model" if prompt else "image_to_model",
            "model_version": model_version
        }
        
        # Add parameters based on generation type
        if prompt:
            data.update({
                "prompt": prompt,
            })
            if negative_prompt:
                data["negative_prompt"] = negative_prompt
            if pose:
                data["prompt"] = f"{prompt}, {pose}"
        else:
            # Handle image input
            if image_url:
                data["file"] = {
                    "type": "png",  # Assuming PNG from Fal
                    "url": image_url
                }
            elif image_path:
                if not os.path.exists(image_path):
                    raise ValueError(f"Image not found: {image_path}")
                    
                # Upload image and get token
                image_token = self._upload_image(image_path)
                data["file"] = {
                    "type": Path(image_path).suffix[1:].lower(),
                    "file_token": image_token
                }
        
        # Add common parameters only if they are set
        optional_params = {
            "model_seed": model_seed,
            "texture_seed": texture_seed,
            "face_limit": face_limit,
            "texture": texture,
            "pbr": pbr,
            "texture_quality": texture_quality,
            "auto_size": auto_size,
            "style": style
        }
        
        # Only add parameters that are not None
        data.update({k: v for k, v in optional_params.items() if v is not None})
        
        # Add orientation and texture alignment if provided
        if orientation:
            data["orientation"] = orientation
        if texture_alignment:
            data["texture_alignment"] = texture_alignment
        
        # Start generation with better error handling
        self.logger.info("Starting 3D model generation...")
        try:
            response = self._make_request("task", data)
            
            # Extract task_id from nested data structure
            task_id = response.get('data', {}).get('task_id')
            
            if not task_id:
                self.logger.error(f"No task ID in response: {response}")
                raise Exception("No task ID received in response")
                
            self.logger.info(f"Starting 3D generation...")
            
            # Add progress bar
            with tqdm(total=100, desc="Generating 3D Model", unit="%") as pbar:
                last_progress = 0
                
                while True:
                    status = self._check_task_status(task_id)
                    state = status.get('data', {}).get('status')
                    progress = status.get('data', {}).get('progress', 0)
                    
                    # Update progress bar
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                    
                    if state == "failed":
                        error_msg = status.get('data', {}).get('message', 'Unknown error')
                        raise Exception(f"Generation failed: {error_msg}")
                        
                    elif state == "success":
                        result = status.get('data', {}).get('result', {})
                        model_info = result.get('pbr_model', {})
                        model_url = model_info.get('url') if isinstance(model_info, dict) else model_info
                        
                        if not model_url:
                            raise Exception("No model URL in success response")
                            
                        # If no output path provided, create one
                        if not output_path:
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            filename = f"tripo_{timestamp}.glb"
                            output_path = str(Path.cwd() / "output" / "threed" / filename)
                        
                        # Download the model
                        self.logger.info("Downloading generated model...")
                        saved_path = self._download_model(model_url, output_path)
                        self.logger.info(f"✨ Model saved to: {saved_path}")
                        return saved_path
                        
                    await asyncio.sleep(2)
                    
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            raise

    async def preview_model(self, model_path: str):
        """Helper to preview 3D models with quicklook."""
        if model_path and Path(model_path).exists():
            print("\n🔍 Opening preview with quicklook...")
            try:
                # Platform-specific file opening
                import platform
                import subprocess
                
                if platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", model_path])
                elif platform.system() == "Windows":
                    os.startfile(model_path)
                else:  # Linux
                    subprocess.run(["xdg-open", model_path])
                    
                print("✨ Preview opened successfully")
            except Exception as e:
                print(f"Failed to open preview: {str(e)}")

# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    import platform
    from pathlib import Path
    
    # Try to import ReplicateAPI
    try:
        from replicate_API import ReplicateAPI
        replicate_available = True
    except ImportError:
        print("ReplicateAPI not available. Will only test text-to-3D.")
        replicate_available = False
    
    async def run_demo():
        print("\n===== TRIPO 3D GENERATION DEMO =====\n")
        
        # Initialize API
        tripo = TripoAPI()
        
        # Predefined creative prompt
        creative_prompt = "A futuristic cyberpunk city with neon lights and flying vehicles, highly detailed 3D environment"
        print(f"Using creative prompt: '{creative_prompt}'")
        
        # Create output directory
        output_dir = Path("output/threed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate 3D model directly from text prompt
        print("\n=== Generating 3D Model from Text ===")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        text_model_path = str(output_dir / f"text_model_{timestamp}.glb")
        
        try:
            text_model = await tripo.generate_threed(
                prompt=creative_prompt,
                output_path=text_model_path,
                texture_quality="detailed"
            )
            
            print(f"\n✅ Text-to-3D model saved to: {text_model}")
            
            # Preview the model
            await tripo.preview_model(text_model)
            input("\nPress Enter to continue...")
            
        except Exception as e:
            print(f"\n❌ Text-to-3D generation failed: {e}")
            text_model = None
        
        # Only run image-to-3D if ReplicateAPI is available
        if replicate_available:
            # Step 2: Generate an image first, then convert to 3D
            print("\n=== Generating Image from Text ===")
            replicate = ReplicateAPI()
            
            try:
                # Generate image
                image_url = replicate.generate_image(
                    prompt=creative_prompt,
                    aspect_ratio="1:1",  # Square aspect ratio
                    safety_tolerance=5    # Moderate safety settings
                )
                
                if image_url:
                    print(f"\n✅ Image generated: {image_url}")
                    
                    # Step 3: Generate 3D model from the image
                    print("\n=== Generating 3D Model from Image ===")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    image_model_path = str(output_dir / f"image_model_{timestamp}.glb")
                    
                    try:
                        image_model = await tripo.generate_threed(
                            image_url=image_url,
                            output_path=image_model_path,
                            texture_quality="detailed"
                        )
                        
                        print(f"\n✅ Image-to-3D model saved to: {image_model}")
                        
                        # Preview the model
                        await tripo.preview_model(image_model)
                        
                    except Exception as e:
                        print(f"\n❌ Image-to-3D generation failed: {e}")
                else:
                    print("\n❌ Image generation failed")
            
            except Exception as e:
                print(f"\n❌ Error in image generation: {e}")
        
        print("\n===== DEMO COMPLETE =====")
    
    # Run the async demo
    asyncio.run(run_demo())
