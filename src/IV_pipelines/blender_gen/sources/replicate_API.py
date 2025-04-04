import os
import sys
import time
import replicate
import requests
import tempfile
import subprocess
import concurrent.futures
from typing import Dict, List, Optional, Union, Any, Tuple
from dotenv import load_dotenv
from pathlib import Path

class ReplicateAPI:
    """
    Integrated Replicate API wrapper with image, video, and music generation capabilities.
    Focuses on returning URLs for generated content rather than managing files directly.
    """

    def __init__(self, api_token: Optional[str] = None):
        """Initialize Replicate API with optional API token override"""
        # Load from .env file if exists
        load_dotenv()
        
        # Use provided API token if available, otherwise use environment variable
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN")
        
        if not self.api_token:
            print("Warning: No Replicate API token provided or found in environment")
        
        # Will be used in each method
        self.client = None
        if self.api_token:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def run_model(
        self,
        model_path: str,
        input_data: Dict,
        version: Optional[str] = None
    ) -> Any:
        """
        Run any Replicate model with given inputs and return results.
        
        Args:
            model_path: The model identifier (e.g., 'owner/model-name')
            input_data: Dictionary of input parameters for the model
            version: Optional specific model version
            
        Returns:
            The model's output (often URLs to generated content)
        """
        try:
            # Pre-process any image inputs
            for key, value in input_data.items():
                if isinstance(value, str) and (
                    key in ['image', 'image_path', 'init_image'] or 'image' in key
                ) and not value.startswith(('http://', 'https://')):
                    input_data[key] = self.prepare_image_input(value)
            
            # Set the complete model path with version if provided
            if version:
                model = f"{model_path}:{version}"
            else:
                model = model_path
                
            # Run the model
            output = replicate.run(model, input=input_data)
            
            # Handle different output formats consistently
            if isinstance(output, list) and output:
                # Most media generation models return a list with the first item being the URL
                return output[0]
            
            return output
            
        except Exception as e:
            print(f"Error running model: {e}")
            return None

    def prepare_image_input(self, image_path: str) -> Optional[Union[str, bytes]]:
        """
        Prepare image input for Replicate API.
        
        Args:
            image_path: Path to local image file or URL
            
        Returns:
            URL string for remote files or file object for local files
        """
        try:
            # If already a URL, return as is
            if image_path.startswith(('http://', 'https://')):
                return image_path
                
            # If it's a local file, return file object
            if os.path.exists(image_path):
                return open(image_path, "rb")
                
            raise ValueError(f"Invalid image path: {image_path}")
            
        except Exception as e:
            print(f"Error preparing image input: {e}")
            return None

    def generate_image(
        self,
        prompt: str,
        model: str = "flux-dev",
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "3:2",
        output_format: str = "jpg",
        raw: bool = False,
        safety_tolerance: int = 2,  # Range 0-6, default 2
        image_prompt_strength: float = 0.1,
    ) -> Optional[str]:
        """
        Generate image using various Replicate models.
        
        Args:
            prompt: Text description of the desired image
            model: Model to use (default: "flux-dev")
                Options: "flux-schnell", "flux-pro", "flux-pro-ultra", "flux-dev", "recraft", "imagen-3", "imagen-3-fast"
            negative_prompt: What to avoid (optional)
            aspect_ratio: Image aspect ratio (default: "3:2")
            output_format: Output file format (default: "jpg")
            raw: Whether to use raw mode (default: False)
            safety_tolerance: Safety filter level (range 0-6, default: 2)
            image_prompt_strength: Strength of image prompt (default: 0.1)
            
        Returns:
            URL to the generated image or None if generation failed
        """
        try:
            # Prepare base input data common for most models
            input_data = {
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "output_format": output_format,
            }
            
            # Only add negative_prompt if provided
            if negative_prompt:
                input_data["negative_prompt"] = negative_prompt

            # Determine which model to use
            model_path = None
            version = None
            
            # Flux models family (Black Forest Labs)
            if model in ["flux-schnell", "flux-pro", "flux-pro-ultra", "flux-dev"]:
                if model == "flux-schnell":
                    model_path = "black-forest-labs/flux-schnell"
                elif model == "flux-pro":
                    model_path = "black-forest-labs/flux-1.1-pro"
                elif model == "flux-pro-ultra":
                    model_path = "black-forest-labs/flux-1.1-pro-ultra"
                else:  # Default to flux-dev
                    model_path = "black-forest-labs/flux-dev"
            
            # Recraft model
            elif model == "recraft":
                model_path = "recraft-ai/recraft-v3"
                # Add model-specific parameters but keep aspect ratio consistent
                if "16:9" in aspect_ratio:
                    input_data["width"] = 1024
                    input_data["height"] = 576
                elif "1:1" in aspect_ratio:
                    input_data["width"] = 1024
                    input_data["height"] = 1024
                else:  # Default to 3:2
                    input_data["width"] = 1024
                    input_data["height"] = 683
            
            # Google Imagen models
            elif model in ["imagen-3", "imagen-3-fast"]:
                model_path = "google/imagen-3"
                if model == "imagen-3-fast":
                    input_data["scale"] = 7.5  # Set the guidance scale
                    input_data["steps"] = 30   # Reduced steps for faster generation
            
            # Fall back to flux-dev if model not recognized
            else:
                print(f"Warning: Unrecognized model '{model}'. Using flux-dev instead.")
                model_path = "black-forest-labs/flux-dev"
            
            print(f"Generating image with {model_path}...")
            output = self.run_model(model_path, input_data=input_data, version=version)
            
            print(f"Image generated successfully with {model}: {prompt[:30]}...")
            return output
            
        except Exception as e:
            print(f"Error generating image with {model}: {e}")
            return None

    def generate_video(
        self,
        prompt: str,
        model: str = "wan-i2v-480p",
        image_url: Optional[str] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "16:9",
        duration: int = 5
    ) -> Optional[str]:
        """
        Generate video using various Replicate models.
        
        Args:
            prompt: Text description of the desired video
            model: Model to use (default: "wan-i2v-480p")
                Options: "wan-i2v-720p", "wan-t2v-720p", "wan-i2v-480p", "wan-t2v-480p", "veo2"
            image_url: URL of the source image or local file path (required for image-to-video models)
            seed: Random seed for reproducibility (optional)
            aspect_ratio: Aspect ratio for text-to-video models (default: "16:9")
            duration: Video duration in seconds for veo2 model (default: 5)
            
        Returns:
            URL to the generated video or None if generation failed
        """
        try:
            # Default parameters for WAN models
            default_wan_params = {
                "fast_mode": "Balanced",
                "num_frames": 81,  # Minimum required by model
                "sample_steps": 30,
                "frames_per_second": 16,
                "sample_guide_scale": 5.0
            }
            
            # Process image URL if provided
            if image_url:
                # Check if it's a file path instead of URL
                if os.path.exists(image_url):
                    print("Local file path detected, uploading to Replicate...")
                    try:
                        image_url = replicate.upload(open(image_url, "rb"))
                        print(f"Image uploaded successfully: {image_url}")
                    except Exception as e:
                        print(f"Error uploading image: {e}")
                        if "i2v" in model:
                            raise ValueError(f"Failed to upload image for {model} model which requires an image.")
                # Convert FileOutput object to string URL if needed
                elif hasattr(image_url, 'url'):
                    image_url = image_url.url
                
                # Ensure we have a valid URL
                if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                    raise ValueError(f"Invalid image URL: {image_url}. Must be a URL string.")
                
                print(f"Using image URL for video generation: {image_url[:50]}...")
            
            # Prepare model-specific parameters and validate requirements
            if model == "wan-i2v-720p":
                if not image_url:
                    raise ValueError("Image URL is required for image-to-video models")
                
                model_path = "wavespeedai/wan-2.1-i2v-720p"
                input_data = {
                    "image": image_url,
                    "prompt": prompt,
                    "max_area": "720x1280",
                    "sample_shift": 5,
                    **default_wan_params
                }
                
            elif model == "wan-i2v-480p":
                if not image_url:
                    raise ValueError("Image URL is required for image-to-video models")
                
                model_path = "wavespeedai/wan-2.1-i2v-480p"
                input_data = {
                    "image": image_url,
                    "prompt": prompt,
                    "max_area": "832x480",
                    "sample_shift": 3,
                    **default_wan_params
                }
                
                # Add seed if provided
                if seed is not None:
                    input_data["seed"] = seed
                
            elif model == "wan-t2v-720p":
                model_path = "wavespeedai/wan-2.1-t2v-720p"
                input_data = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "sample_shift": 5,
                    **default_wan_params
                }
            elif model == "wan-t2v-480p":
                model_path = "wavespeedai/wan-2.1-t2v-480p"
                input_data = {
                    "prompt": prompt,
                    "aspect_ratio": aspect_ratio,
                    "sample_shift": 5,
                    **default_wan_params
                }
                
            elif model == "veo2":
                model_path = "google/veo-2"
                input_data = {
                    "prompt": prompt,
                    "duration": duration,
                    "aspect_ratio": aspect_ratio
                }
                
                # Add seed if provided
                if seed is not None:
                    input_data["seed"] = seed
                
            else:
                raise ValueError(f"Unsupported model: {model}. Choose from: wan-i2v-720p, wan-t2v-720p, wan-i2v-480p, wan-t2v-480p, veo2")
            
            # Run the model
            output = self.run_model(model_path, input_data=input_data)
            
            print(f"Video generated successfully using {model}")
            return output
            
        except Exception as e:
            print(f"Error generating video: {type(e).__name__}: {e}")
            return None

    def generate_music(
        self,
        prompt: str,
        duration: int = 8,
        model_version: str = "stereo-large",
        top_k: int = 250,
        top_p: float = 0.0,
        temperature: float = 1.0,
        continuation: bool = False,
        output_format: str = "mp3",
        continuation_start: int = 0,
        multi_band_diffusion: bool = False,
        normalization_strategy: str = "peak",
        classifier_free_guidance: float = 3.0
    ) -> Optional[str]:
        """
        Generate music using Meta's MusicGen model.
        
        Args:
            prompt: Text description of desired music
            duration: Length in seconds (default: 8)
            model_version: Model version to use (default: "stereo-large")
            top_k: Top-k sampling parameter (default: 250)
            top_p: Top-p sampling parameter (default: 0.0)
            temperature: Generation temperature (default: 1.0)
            continuation: Whether to continue from previous (default: False)
            output_format: Output audio format (default: "mp3")
            continuation_start: Start time for continuation (default: 0)
            multi_band_diffusion: Use multi-band diffusion (default: False)
            normalization_strategy: Audio normalization (default: "peak")
            classifier_free_guidance: Guidance scale (default: 3.0)
            
        Returns:
            URL to the generated audio or None if generation failed
        """
        try:
            output = self.run_model(
                "meta/musicgen",
                input_data={
                    "prompt": prompt,
                    "duration": duration,
                    "model_version": model_version,
                    "top_k": top_k,
                    "top_p": top_p,
                    "temperature": temperature,
                    "continuation": continuation,
                    "output_format": output_format,
                    "continuation_start": continuation_start,
                    "multi_band_diffusion": multi_band_diffusion,
                    "normalization_strategy": normalization_strategy,
                    "classifier_free_guidance": classifier_free_guidance
                },
                version="671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
            )
            
            print(f"Music generated successfully: {prompt[:30]}...")
            return output
            
        except Exception as e:
            print(f"Error generating music: {e}")
            return None

    def generate_threed(
        self,
        image_url: str,
        model: str = "hunyuan3d",
        seed: int = 1234,
        steps: int = 50,
        guidance_scale: float = 5.5,
        octree_resolution: int = 256,
        remove_background: bool = True,
        texture_size: int = 1024,
        mesh_simplify: float = 0.9,
        generate_color: bool = True,
        generate_normal: bool = True,
        randomize_seed: bool = False,
        save_gaussian_ply: bool = False,
        ss_sampling_steps: int = 38,
        slat_sampling_steps: int = 12,
        ss_guidance_strength: float = 7.5,
        slat_guidance_strength: float = 3
    ) -> Optional[str]:
        """
        Generate 3D model from an image.
        
        Args:
            image_url: URL of the source image
            model: Model to use (default: "hunyuan3d")
                Options: "hunyuan3d", "trellis"
            seed: Random seed for reproducibility (default: 1234)
            steps: Number of inference steps for Hunyuan3D (default: 50)
            guidance_scale: Guidance scale for Hunyuan3D (default: 5.5)
            octree_resolution: Resolution of the 3D model for Hunyuan3D (default: 256)
            remove_background: Whether to remove image background for Hunyuan3D (default: True)
            
            # Trellis-specific parameters
            texture_size: Size of generated textures for Trellis (default: 1024)
            mesh_simplify: Mesh simplification ratio for Trellis (default: 0.9)
            generate_color: Whether to generate color textures for Trellis (default: True)
            generate_normal: Whether to generate normal maps for Trellis (default: True)
            randomize_seed: Whether to randomize seed for Trellis (default: False)
            save_gaussian_ply: Whether to save Gaussian PLY file for Trellis (default: False)
            ss_sampling_steps: Surface sampling steps for Trellis (default: 38)
            slat_sampling_steps: SLAT sampling steps for Trellis (default: 12)
            ss_guidance_strength: Surface sampling guidance strength for Trellis (default: 7.5)
            slat_guidance_strength: SLAT guidance strength for Trellis (default: 3)
            
        Returns:
            URL to the generated 3D model or None if generation failed
        """
        try:
            # Process image URL
            if hasattr(image_url, 'url'):
                image_url = image_url.url
            
            # Ensure we have a valid URL
            if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid image URL: {image_url}. Must be a URL string.")
            
            # Choose the appropriate model
            if model.lower() == "hunyuan3d":
                print(f"Generating 3D model with Hunyuan3D from image: {image_url[:50]}...")
                
                output = self.run_model(
                    "tencent/hunyuan3d-2",
                    input_data={
                        "seed": seed,
                        "image": image_url,
                        "steps": steps,
                        "guidance_scale": guidance_scale,
                        "octree_resolution": octree_resolution,
                        "remove_background": remove_background
                    },
                    version="b1b9449a1277e10402781c5d41eb30c0a0683504fb23fab591ca9dfc2aabe1cb"
                )
                
                # Handle the specific output format for Hunyuan3D model
                if isinstance(output, dict) and 'mesh' in output:
                    if hasattr(output['mesh'], 'url'):
                        # Extract URL from FileOutput object
                        return output['mesh'].url
                    else:
                        # Try to get the URL as a string if it's directly available
                        return output['mesh']
                
            elif model.lower() == "trellis":
                print(f"Generating 3D model with Trellis from image: {image_url[:50]}...")
                
                # Prepare images as a list even if only one image is provided
                images = [image_url]
                
                output = self.run_model(
                    "firtoz/trellis",
                    input_data={
                        "seed": seed if not randomize_seed else 0,
                        "images": images,
                        "texture_size": texture_size,
                        "mesh_simplify": mesh_simplify,
                        "generate_color": generate_color,
                        "generate_model": True,
                        "randomize_seed": randomize_seed,
                        "generate_normal": generate_normal,
                        "save_gaussian_ply": save_gaussian_ply,
                        "ss_sampling_steps": ss_sampling_steps,
                        "slat_sampling_steps": slat_sampling_steps,
                        "return_no_background": remove_background,
                        "ss_guidance_strength": ss_guidance_strength,
                        "slat_guidance_strength": slat_guidance_strength
                    },
                    version="4876f2a8da1c544772dffa32e8889da4a1bab3a1f5c1937bfcfccb99ae347251"
                )
                
                print("Trellis 3D model generated successfully")
                
                # Extract URL from Trellis output - Handle the FileOutput object correctly
                if output and isinstance(output, dict):
                    if "model_file" in output:
                        # Extract URL from FileOutput object if necessary
                        if hasattr(output["model_file"], "url"):
                            return output["model_file"].url
                        else:
                            return output["model_file"]
                
                # If we have a direct link to the mesh (non-dictionary output)
                if isinstance(output, str) and output.endswith((".glb", ".obj", ".fbx")):
                    return output
                
                print(f"Warning: Unexpected output format from Trellis: {type(output)}")
                # Return the raw output as a last resort - will need to be handled by the caller
                return output
                
            else:
                raise ValueError(f"Unsupported 3D model: {model}. Choose from: hunyuan3d, trellis")
            
            # Fallback to direct output if not in the expected format
            return output
            
        except Exception as e:
            print(f"Error generating 3D model: {type(e).__name__}: {e}")
            return None

    def download_file(self, url: str, output_dir: Optional[str] = None, filename: Optional[str] = None) -> Optional[str]:
        """
        Download a file from a URL to a specific output directory.

        Args:
            url: URL of the file to download
            output_dir: Directory to save the file (defaults to temp directory)
            filename: Optional filename (generated from timestamp if not provided)

        Returns:
            Path to the downloaded file
        """
        try:
            # Handle dictionary output from Trellis
            if isinstance(url, dict):
                print(f"Received dictionary output: {url}")
                # Try to find a model file or any other usable URL in the dictionary
                model_file = None

                # Check for model_file first
                if 'model_file' in url:
                    model_file = url['model_file']
                    if hasattr(model_file, 'url'):
                        url = model_file.url
                    else:
                        url = model_file
                # Otherwise, try to find any key with a FileOutput object that has a URL
                else:
                    for key, value in url.items():
                        if hasattr(value, 'url'):
                            url = value.url
                            print(f"Found URL in '{key}' key: {url}")
                            break
                    else:
                        raise ValueError(f"No valid URL found in dictionary: {url}")

            # Create output directory if provided
            if output_dir:
                # Create base output directory in data folder
                base_dir = Path("data/output")
                if not base_dir.exists():
                    base_dir.mkdir(parents=True, exist_ok=True)

                # Create specific media directory
                media_dir = base_dir / output_dir
                if not media_dir.exists():
                    media_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename if not provided
                if not filename:
                    extension = url.split('.')[-1] if '.' in url.split('/')[-1] else 'tmp'
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = f"{output_dir}_{timestamp}.{extension}"

                output_path = media_dir / filename
            else:
                # Use temp directory if no output directory specified
                extension = url.split('.')[-1] if '.' in url.split('/')[-1] else 'tmp'
                fd, output_path = tempfile.mkstemp(suffix=f'.{extension}')
                os.close(fd)
                output_path = Path(output_path)

            # Download the file
            print(f"Downloading to {output_path}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Downloaded successfully ({os.path.getsize(output_path) / 1024:.1f} KB)")
            return str(output_path)

        except Exception as e:
            print(f"Error downloading file: {e}")
            return None

    def display_media(self, file_path: str, media_type: str = "image"):
        """Display media using appropriate system tools."""
        try:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False

            if sys.platform == "darwin":  # macOS
                if media_type == "image":
                    # Try QuickLook first
                    print("Opening image with QuickLook...")
                    subprocess.run(["qlmanage", "-p", file_path],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
                elif media_type == "video":
                    # For videos, use QuickTime Player instead of QuickLook for better playback
                    print("Opening video with QuickTime Player...")
                    subprocess.run(["open", "-a", "QuickTime Player", file_path])
                elif media_type == "audio":
                    # Use afplay for audio
                    print("Playing audio...")
                    subprocess.run(["afplay", file_path])
                elif media_type == "model" or media_type == "3d":
                    # Open 3D model with default application
                    print("Opening 3D model with default viewer...")
                    subprocess.run(["open", file_path])

            elif sys.platform == "win32":  # Windows
                # Use the default application
                os.startfile(file_path)

            else:  # Linux
                try:
                    subprocess.run(["xdg-open", file_path])
                except:
                    print(f"Could not open file: {file_path}")
                    return False

            return True

        except Exception as e:
            print(f"Error displaying media: {e}")
            return False

    def merge_video_audio(self, video_path: str, audio_path: str, filename: Optional[str] = None) -> Optional[str]:
        """Merge video and audio into a single file using ffmpeg."""
        try:
            # Create output directory in data folder
            output_dir = Path("data/output/videos")
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename if not provided
            if not filename:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"merged_{timestamp}.mp4"

            output_path = output_dir / filename
            
            print(f"Merging video and audio to {output_path}...")
            
            # Use ffmpeg to merge video and audio
            ffmpeg_cmd = [
                "ffmpeg", "-y",  # Overwrite output file if exists
                "-i", video_path,  # Video input
                "-i", audio_path,  # Audio input
                "-map", "0:v",  # Use video from first input
                "-map", "1:a",  # Use audio from second input
                "-c:v", "copy",  # Copy video codec
                "-shortest",  # Make output duration same as shortest input
                str(output_path)
            ]
            
            # Check if ffmpeg is available
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                print("Error: ffmpeg not found. Please install ffmpeg to merge video and audio.")
                return None
            
            # Run ffmpeg command
            process = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if process.returncode != 0:
                print(f"Error merging files: {process.stderr.decode()}")
                return None
            
            print(f"Successfully merged video and audio to {output_path}")
            return str(output_path)
            
        except Exception as e:
            print(f"Error merging video and audio: {e}")
            return None

# Entry point
if __name__ == "__main__":
    def run_test():
        print("\n===== REPLICATE API TEST =====")
        print("Testing different media generation capabilities with predefined prompts.")
        
        # Initialize API
        try:
            api = ReplicateAPI()
        except ValueError as e:
            print(f"Error: {e}")
            print("Make sure to set the REPLICATE_API_TOKEN in your .env file.")
            sys.exit(1)
        
        # Image models to test
        image_models = [
            "flux-schnell", 
            "flux-pro", 
            "flux-pro-ultra", 
            "flux-dev", 
            "recraft", 
            "imagen-3",
            "imagen-3-fast"
        ]
        
        # Display available image models and let the user choose
        print("\nAvailable image models:")
        for i, model in enumerate(image_models):
            print(f"{i+1}. {model}")
        
        # Get model selection from user
        model_selection = input("\nChoose image models to test (comma-separated numbers or 'all'): ").strip().lower()
        
        # Parse the selection
        selected_models = []
        if model_selection == 'all':
            selected_models = image_models
        else:
            try:
                # Split by comma and convert to integers
                indices = [int(idx.strip()) - 1 for idx in model_selection.split(',')]
                # Filter valid indices and get corresponding models
                selected_models = [image_models[idx] for idx in indices if 0 <= idx < len(image_models)]
                
                if not selected_models:
                    print("No valid models selected. Using flux-dev as default.")
                    selected_models = ["flux-dev"]
            except ValueError:
                print("Invalid selection. Using flux-dev as default.")
                selected_models = ["flux-dev"]
        
        print(f"\nSelected image models: {', '.join(selected_models)}")
        
        # Predefined prompts for testing
        test_prompts = {
            # Previous test prompts (commented out)
            # "image": "Photorealistic exterior view of a futuristic Geode habitat in San Francisco, geodesic glass sphere with hexagonal panels, three distinct internal levels visible through transparent exterior, integrated bioluminescent AI technology embedded in framework, sustainable self-sufficient ecosystem, advanced sustainable materials, crisp morning light creating lens flares through structure, architectural visualization, 8K resolution, ultra-detailed",
            
            # "video": "Cinematic aerial shot slowly orbiting around a futuristic Geode habitat structure in San Francisco, transparent geodesic sphere with hexagonal glass panels, blue energy circuits pulsing through framework, three distinct internal levels visible, lush internal gardens, advanced AI systems monitoring environmental controls, morning sunlight glinting off surface, camera smoothly orbiting the structure, photorealistic rendering, epic scale",
            
            # "music": "Futuristic ambient electronic soundscape with hopeful undertones, gentle synthesizer arpeggios representing flowing energy systems within the Geode habitat, subtle orchestral elements conveying grandeur of architectural innovation, medium tempo evolving composition, evokes feelings of technological harmony with nature, soaring sections suggesting vast open interior spaces of the geodesic structure"
            
            # Caiman character prompts (commented out)
            # "image": "Realistic caiman, stylized character with realistic proportions, appealing design, vibrant green color, created by Disney, full-body depiction standing on two legs in a neutral A-pose, VFX turntable setup, studio lighting environment, highly detailed with realistic texturing and shading, cinematic quality, 8K resolution",
            
            # "video": "VFX turntable video of a realistic caiman character, stylized with realistic proportions, appealing design, vibrant green, created by Disney, full-body standing on two legs in a neutral A-pose, studio lighting environment, highly detailed with realistic texturing and shading, smooth camera rotation around the character, cinematic quality, 8K resolution",
            
            # "music": "Orchestral soundtrack with whimsical undertones, representing the playful nature of a Disney-styled caiman character, gentle woodwind melodies with vibrant string sections, medium tempo evolving composition, evokes feelings of adventure and charm, suitable for a character introduction scene, cinematic quality, inspired by classic Disney scores"
            
            # Mongolian-inspired 3D printable gift prompts
            # "image": "Abstract sculptural desk piece inspired by Mongolian culture, featuring elegant flowing lines and a harmonious design, incorporating traditional motifs and symbols, polished stone base with turquoise and coral accents, number 100 subtly incorporated into base pattern, mixed media approach with wood-like and metal textures, elegant 8-inch tall composition suitable for desk display, dramatic lighting highlighting curved contours, product photography with shallow depth of field, ultra-detailed for high-resolution 3D printing",
            
            # "video": "Cinematic product showcase of a Mongolian-inspired desk sculpture, camera slowly orbiting to reveal the harmonious design and traditional motifs, focus pulls highlighting intricate details and craftsmanship, close-up shots of turquoise and coral accents embedded in polished stone base, reveal of subtle 100-day symbolism integrated into design pattern, gentle rotation showing how light interacts with mixed wood and metallic surfaces, transitions demonstrating how different viewing angles reveal different aspects of the design, warm directional lighting creating dramatic shadows, 8K resolution with realistic material rendering",
            
            # "music": "Romantic composition featuring traditional Mongolian instruments with modern accompaniment, capturing the essence of Mongolian culture through a harmonious melody, subtle textures creating depth, medium-slow tempo with gentle rhythm suggesting heartbeats, emotional progression from tender beginning to passionate middle section symbolizing 100 days together, traditional Mongolian scales with modern harmonic structure, warm reverb creating spatial dimension, culminating in intertwining melodic lines representing two lives coming together"
            
            # Mongolian BJL Jewelry Box Designs (commented out)
            # "image": "Hexagonal Mongolian ger-shaped wooden jewelry box with embossed 'BJL' logo on lid, handcrafted from rich cedar wood featuring intricate traditional Mongolian endless knot (Ulzii) patterns carved into sides, turquoise and coral inlay details, silver hardware with aged patina, felt-lined interior, product photography with dramatic side lighting highlighting wood grain and embossed details, ultra-detailed rendering for 3D printing, top view showing the prominent BJL logo surrounded by symmetrical traditional patterns",
            
            # "video": "Cinematic product showcase of a hexagonal ger-shaped Mongolian jewelry box, camera slowly orbiting to reveal the 'BJL' logo embossed prominently on the wooden lid, focus pulls highlighting the intricate endless knot carvings and turquoise inlays along the edges, detailed shots of the traditional patterns flowing around the six sides of the box, gentle rotation showing how light interacts with the polished wood and silver hardware, close-up of the lid opening to reveal the felt-lined interior with specialized compartments for jewelry storage, transitions showing different viewing angles of the embossed BJL logo surrounded by traditional Mongolian motifs, warm directional lighting creating rich shadows that emphasize the craftsmanship, 8K resolution with photorealistic wood grain textures",
            
            # "music": "Serene composition featuring traditional Mongolian morin khuur (horsehead fiddle) and indigenous flutes, creating an authentic cultural atmosphere with gentle rhythmic patterns inspired by horse gaits, meditative melody conveying the elegance of Mongolian jewelry traditions, subtle percussion elements using traditional instruments like the yoochin (hammered dulcimer), layered harmonies representing the combination of traditional craft and modern luxury embodied by the BJL brand, natural acoustic recording with minimal processing, embracing the organic warmth of the instruments and their cultural significance"
            
            # Sonic the Hedgehog character asset prompts (full body version)
            # "image": "Realistic Sonic the Hedgehog, stylized character with realistic proportions, appealing design, vibrant blue fur with detailed quills, full-body depiction standing in a neutral A-pose, VFX turntable setup, studio lighting environment, highly detailed with realistic texturing and shading, red sneakers with white strap, white gloves, emerald green eyes, cinematic quality, 8K resolution",
            
            # "video": "VFX turntable video of a realistic Sonic the Hedgehog character, stylized with realistic proportions, appealing design, vibrant blue fur with detailed quills, full-body standing in a neutral A-pose, studio lighting environment, highly detailed with realistic texturing and shading, smooth camera rotation around the character, red sneakers with white strap, white gloves, emerald green eyes, cinematic quality, 8K resolution",
            
            # "music": "Fast-paced orchestral soundtrack with electronic elements, representing the speedy nature of Sonic the Hedgehog, energetic brass and string sections with modern synthesizer accents, upbeat tempo with driving rhythm, evokes feelings of adventure and excitement, suitable for a character introduction scene, cinematic quality, inspired by classic Sonic game soundtracks with modern production"
            
            # Sonic the Hedgehog facial expression test prompts
            # "image": "Photorealistic mid-shot of Sonic the Hedgehog in 3/4 view showing upper torso and head, stylized character with realistic blue fur texturing, detailed quills framing his face, expressive emerald green eyes with depth and personality, slight smirk showing characteristic confidence, sharp detailed features with subsurface scattering on skin and fur, studio lighting with rim light highlighting blue fur edges, cinematic color grading, ultra-high resolution, 8K detail, inspired by modern VFX character design",
            
            # "video": "Rapid facial expression test of Sonic the Hedgehog's bust in a fixed 3/4 view, quick transitions between diverse expressions including smirk, surprise, anger, joy, determination, confusion, fear, disgust, contempt, and excitement, realistic blue fur with physics simulation, detailed eye movements and brow articulation showing full emotional range, completely stationary camera, studio three-point lighting emphasizing facial contours, rapid-fire expression changes with no dialogue or mouth movement for speech, photorealistic texturing with subsurface scattering, 4K resolution with cinematic depth of field",
            
            # "music": "Relaxing ambient soundtrack with subtle nostalgic Sonic game motifs, gentle synthesizer pads and soft piano melodies, slow tempo at 80 BPM creating a calming atmosphere, familiar Sonic themes reimagined in a soothing arrangement, minimal percussion with occasional soft bell tones, warm major key progression maintaining a peaceful mood throughout, inspired by classic Sonic soundtracks but transformed into a meditative listening experience with modern production techniques"
            
            # Modern Sonic the Hedgehog VFX prompts
            # "image": "Photorealistic Sonic the Hedgehog, full body shot, stylized proportions with longer limbs, fashionable longer quills with subtle blue highlights, modern streetwear outfit with casual jeans and stylish jacket, characteristic red sneakers with custom details, white gloves with fingerless design, emerald green eyes with confident expression, friendly smirk showing personality, VFX turntable setup with dramatic lighting, highly detailed fur simulation with realistic texturing, cinematic quality with depth of field, 8K resolution, modern VFX character design",
            
            # "video": "VFX turntable video of Sonic the Hedgehog, full body 360-degree rotation, stylized proportions with dynamic features, longer styled quills with subtle blue variations, modern streetwear with jacket and casual jeans, customized red sneakers, fingerless gloves, characteristic confident posture, studio lighting with dramatic rim lights highlighting fur edges, ultra-realistic fur and cloth physics, slow smooth camera rotation capturing all details, cinematic color grading, 8K resolution with shallow depth of field focusing on changing expressions showing range of emotions",
            
            # "music": "Dynamic energetic soundtrack blending classic Sonic themes with modern electronic elements, electric guitar riffs with synthesizer accents, medium-fast tempo with driving drum beats, energetic verses with emotional chorus sections, musical progression from playful to heroic, subtle nostalgic Sonic game motifs reimagined with contemporary production, perfect for capturing the spirit of Sonic in his adventures"
            
            # Futuristic Sculpture Design 2042
            # "image": "7 timeless sculptural designs, made in 2042, abstract figurative design, made for 3d printing, all in a line, all very distinct, bright studio environment lighting, photo realistic, 4k, 8k",
            # "image": "adorable skunk, stylized character, appealing design, full-body, moving naturally, VFX asset presentation, advertising, high-quality, photorealistic, hyper-realistic, realistic, 8k",
            
            "image": "black dragon character design, slick, elegant, agile and nimble looking, slim long aerodynamic shape, neutral binding pose, two wings, four legs, full-body, studio environment, highly detailed, hyper-realistic",
            "video": "Cinematic turntable video of a timeless abstract figurative sculpture from 2042, camera slowly rotating 360 degrees around the piece, bright studio lighting highlighting the flowing contours and intricate details, focus pulls revealing the complex interplay of materials and textures, dramatic shadows emphasizing the sculptural form, photorealistic rendering with perfect material properties, 8K resolution with shallow depth of field",
            
            "music": "Ethereal ambient composition with futuristic sound design elements, minimal piano motifs floating over atmospheric synthesizer pads, slow evolving harmonies creating a sense of timelessness, subtle electronic percussion with occasional crystalline bell tones, gradual build in complexity representing the intricate details of the sculpture, modern production techniques creating spatial depth and dimension"
        }
        
        # Storage for generated media
        generated = {
            "images": {},
            "video": None,
            "music": None,
            "merged": None
        }
        
        # Function to generate an image with a specific model
        def generate_image_worker(model_name):
            print(f"\n🖼️ Generating image with model: {model_name}")
            try:
                image_url = api.generate_image(
                    prompt=test_prompts["image"],
                    model=model_name,
                    aspect_ratio="16:9",
                    safety_tolerance=6
                )
                
                if image_url:
                    print(f"✅ Image generated successfully with {model_name}")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    image_path = api.download_file(
                        image_url, 
                        output_dir="test_images", 
                        filename=f"test_{model_name}_{timestamp}.jpg"
                    )
                    
                    return {
                        "model": model_name,
                        "url": image_url,
                        "path": image_path,
                        "success": image_path is not None
                    }
                else:
                    print(f"❌ Image generation failed with {model_name}")
            except Exception as e:
                print(f"❌ Error with {model_name}: {str(e)}")
            
            return {
                "model": model_name,
                "url": None,
                "path": None,
                "success": False
            }
        
        # 1. Generate images with selected models in parallel
        image_results = []
        successful_images = []
        
        print("\n===== GENERATING IMAGES WITH SELECTED MODELS =====")
        print(f"Testing {len(selected_models)} image models in parallel with prompt: '{test_prompts['image'][:50]}...'")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(selected_models), 4)) as executor:
            # Submit tasks (limit to 4 concurrent to avoid rate limiting)
            futures = {executor.submit(generate_image_worker, model): model for model in selected_models}
            
            # Monitor progress
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    result = future.result()
                    image_results.append(result)
                    
                    if result["success"]:
                        generated["images"][model] = result["path"]
                except Exception as e:
                    print(f"❌ Error processing result for {model}: {str(e)}")
        
        # Display results summary for images
        print("\n===== IMAGE GENERATION RESULTS =====")
        successful_images = [r for r in image_results if r["success"]]
        
        print(f"Tested {len(selected_models)} models")
        print(f"Successfully generated: {len(successful_images)}")
        
        for result in successful_images:
            print(f"✅ {result['model']}: {os.path.basename(result['path'])}")
        
        # Display all successful images
        if successful_images:
            print("\n===== DISPLAYING IMAGES =====")
            for result in successful_images:
                print(f"Opening image from {result['model']}...")
                api.display_media(result['path'], "image")
                time.sleep(0.5)  # Short delay between images
        
        # Ask user if they want to generate 3D models after seeing the images
        test_threed = False
        if successful_images:
            test_threed = input("\nGenerate 3D models from images? (y/n): ").lower().strip() == 'y'
            
        # Generate 3D models from selected images if user wants to
        if test_threed:
            print("\n===== GENERATING 3D MODELS =====")
            
            # Display available 3D models and let the user choose
            print("\nAvailable 3D models:")
            print("1. Trellis - Better for objects, detailed textures, faster")
            print("2. Hunyuan3D - Better for detailed geometry, slower")
            print("3. Both models - Try both and compare results")

            # Get 3D model selection from user
            threed_models = []  # Store selected models
            threed_model_choice = input("\nChoose 3D model(s) (1, 2, or 3 for both, default is 1): ").strip()
            if threed_model_choice == "2":
                threed_models = ["hunyuan3d"]
            elif threed_model_choice == "3":
                threed_models = ["trellis", "hunyuan3d"]
            else:
                threed_models = ["trellis"]  # Default to Trellis

            print(f"\nSelected 3D model(s): {', '.join(threed_models)}")
            
            # Ask user which images to use
            print("\nAvailable successful images:")
            for i, result in enumerate(successful_images):
                print(f"{i+1}. {result['model']}")
            
            # Get user's choice for which images to use
            while True:
                try:
                    image_choices_input = input("\nEnter the numbers of the images to use for 3D models (comma-separated): ").strip()
                    image_indices = [int(idx.strip()) - 1 for idx in image_choices_input.split(',')]
                    
                    # Validate if all provided indices are valid
                    valid_choices = [idx for idx in image_indices if 0 <= idx < len(successful_images)]
                    
                    if valid_choices:
                        threed_chosen_results = [successful_images[idx] for idx in valid_choices]
                        break
                    else:
                        print(f"Please enter valid numbers between 1 and {len(successful_images)}")
                except ValueError:
                    print("Please enter valid numbers separated by commas")
            
            # Storage for 3D results
            threed_results = []
            
            # Function to generate a 3D model from an image
            def generate_threed_worker(chosen_result, model_name):
                print(f"\n🧊 Generating 3D model from image: {chosen_result['model']} using {model_name}")
                try:
                    # Get the URL string from the object or directly
                    image_url = chosen_result['url']
                    
                    # If it's a FileOutput object, extract the url attribute
                    if hasattr(image_url, 'url'):
                        image_url = image_url.url
                    
                    # Ensure we have a valid URL string
                    if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                        print(f"Invalid image URL from {chosen_result['model']}, skipping...")
                        return {
                            "image_source": chosen_result['model'],
                            "url": None,
                            "path": None,
                            "success": False
                        }
                    
                    # Generate the 3D model using the selected model
                    print(f"Generating 3D model using {chosen_result['model']} image with {model_name.upper()}...")
                    threed_url = api.generate_threed(
                        image_url=image_url,
                        model=model_name,  # Use the selected 3D model
                        seed=1234,
                        randomize_seed=True,
                        texture_size=1024,
                        mesh_simplify=0.9,
                        generate_color=True,
                        generate_normal=True,
                        ss_sampling_steps=38,
                        slat_sampling_steps=12,
                        ss_guidance_strength=7.5,
                        slat_guidance_strength=3
                    )
                    
                    if threed_url:
                        print(f"✅ 3D model generated successfully from {chosen_result['model']} image")
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        threed_path = api.download_file(
                            threed_url, 
                            output_dir="test_3d_models", 
                            filename=f"test_3d_{chosen_result['model']}_{timestamp}.glb"
                        )
                        
                        if threed_path:
                            print(f"Opening 3D model from {chosen_result['model']}...")
                            api.display_media(threed_path, "model")  # Display the 3D model
                        
                        return {
                            "image_source": chosen_result['model'],
                            "url": threed_url,
                            "path": threed_path,
                            "success": threed_path is not None
                        }
                    else:
                        print(f"❌ 3D model generation failed with {chosen_result['model']} image")
                except Exception as e:
                    print(f"❌ Error generating 3D model from {chosen_result['model']} image: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                return {
                    "image_source": chosen_result['model'],
                    "url": None,
                    "path": None,
                    "success": False
                }
            
            # Generate 3D models with selected images in parallel - updated for multiple model types
            if threed_chosen_results:
                all_tasks = []
                # Create tasks for all combinations of selected images and models
                for result in threed_chosen_results:
                    for model in threed_models:
                        all_tasks.append((result, model))
                
                print(f"\nGenerating {len(all_tasks)} 3D models ({len(threed_chosen_results)} images × {len(threed_models)} models)")
                
                # Use ThreadPoolExecutor to run 3D generation in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(all_tasks), 4)) as executor:
                    # Submit tasks
                    futures = {executor.submit(generate_threed_worker, result, model): (result, model) 
                              for result, model in all_tasks}
                    
                    # Monitor progress
                    for future in concurrent.futures.as_completed(futures):
                        result_img, model_name = futures[future]
                        result = future.result()
                        threed_results.append(result)
                        
                        # Store successful results
                        if result["success"]:
                            if "threed_models" not in generated:
                                generated["threed_models"] = []
                            generated["threed_models"].append(result["path"])
                
                # Display results summary
                print("\n===== 3D MODEL GENERATION RESULTS =====")
                successful_threed = [r for r in threed_results if r["success"]]
                
                print(f"Attempted: {len(threed_chosen_results)} images with {', '.join(threed_models)} models")
                print(f"Successfully generated: {len(successful_threed)}")
                
                for result in successful_threed:
                    print(f"✅ 3D from {result['image_source']} image: {os.path.basename(result['path'])}")
        
        # Ask user if they want to generate videos after seeing the images
        test_video = False
        if successful_images:
            test_video = input("\nGenerate videos from an image? (y/n): ").lower().strip() == 'y'
        
        # 2. Generate video using selected images if user wants to
        if test_video:
            print("\n===== GENERATING VIDEO =====")
            
            # Ask user which images to use
            print("\nAvailable successful images:")
            for i, result in enumerate(successful_images):
                print(f"{i+1}. {result['model']}")
            
            # Get user's choice for which images to use
            while True:
                try:
                    image_choices_input = input("\nEnter the numbers of the images to use for video (comma-separated): ").strip()
                    image_indices = [int(idx.strip()) - 1 for idx in image_choices_input.split(',')]
                    
                    # Validate if all provided indices are valid
                    valid_choices = [idx for idx in image_indices if 0 <= idx < len(successful_images)]
                    
                    if valid_choices:
                        chosen_results = [successful_images[idx] for idx in valid_choices]
                        break
                    else:
                        print(f"Please enter valid numbers between 1 and {len(successful_images)}")
                except ValueError:
                    print("Please enter valid numbers separated by commas")
            
            # Ask user if they want to test high-res 720p (more tokens, slower)
            test_720p = input("Test high-resolution 720p video? (y/n): ").lower().strip() == 'y'
            
            # Select video model based on 720p choice - only ONE model
            video_model = "wan-i2v-720p" if test_720p else "wan-i2v-480p"
            
            # Storage for video results
            video_results = []
            
            # Prepare base parameters for video generation
            base_video_params = {
                "prompt": test_prompts["video"],
                "aspect_ratio": "16:9"
            }
            
            # Function to generate a video with a specific model and image
            def generate_video_worker(model_name, chosen_result):
                print(f"\n🎬 Generating video with model: {model_name} using image from {chosen_result['model']}")
                try:
                    # Get the URL string from the object or directly
                    image_url = chosen_result['url']
                    
                    # If it's a FileOutput object, extract the url attribute
                    if hasattr(image_url, 'url'):
                        image_url = image_url.url
                    
                    # Ensure we have a valid URL string
                    if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                        print(f"Invalid image URL from {chosen_result['model']}, skipping...")
                        return {
                            "model": model_name,
                            "image_source": chosen_result['model'],
                            "url": None,
                            "path": None,
                            "success": False
                        }
                
                    # Clone the base parameters and add model-specific options
                    video_params = base_video_params.copy()
                    video_params["model"] = model_name
                    video_params["image_url"] = image_url
                    
                    # Generate the video
                    print(f"Generating video using {chosen_result['model']} image...")
                    video_url = api.generate_video(**video_params)
                    
                    if video_url:
                        print(f"✅ Video generated successfully with {model_name} using {chosen_result['model']} image")
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        video_path = api.download_file(
                            video_url, 
                            output_dir="test_videos", 
                            filename=f"test_{model_name}_{chosen_result['model']}_{timestamp}.mp4"
                        )
                        
                        return {
                            "model": model_name,
                            "image_source": chosen_result['model'],
                            "url": video_url,
                            "path": video_path,
                            "success": video_path is not None
                        }
                    else:
                        print(f"❌ Video generation failed with {model_name} using {chosen_result['model']} image")
                except Exception as e:
                    print(f"❌ Error with {model_name} using {chosen_result['model']} image: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                return {
                    "model": model_name,
                    "image_source": chosen_result['model'],
                    "url": None,
                    "path": None,
                    "success": False
                }
            
            # Generate videos with selected model in parallel
            if chosen_results:
                print(f"\nGenerating {len(chosen_results)} videos with model {video_model}")
                print(f"Using prompt: '{test_prompts['video'][:50]}...'")
                
                # Use ThreadPoolExecutor to run video generation in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chosen_results), 4)) as executor:
                    # Submit tasks
                    futures = {executor.submit(generate_video_worker, video_model, result): result for result in chosen_results}
                    
                    # Monitor progress
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        video_results.append(result)
                        
                        # Display the video immediately if generated successfully
                        if result["success"]:
                            print(f"Displaying video from {result['model']} using {result['image_source']} image...")
                            api.display_media(result["path"], "video")
                            # Add this video to generated videos list
                            if "videos" not in generated:
                                generated["videos"] = []
                            generated["videos"].append(result["path"])
                
                # Display results summary
                print("\n===== VIDEO GENERATION RESULTS =====")
                successful_videos = [r for r in video_results if r["success"]]
                
                print(f"Tested {len(chosen_results)} image(s) with {video_model}")
                print(f"Successfully generated: {len(successful_videos)}")
                
                for result in successful_videos:
                    print(f"✅ {result['model']} using {result['image_source']} image: {os.path.basename(result['path'])}")
            
            # Update the music generation section to handle multiple videos
            test_music = False
            if generated.get("videos") and len(generated["videos"]) > 0:
                test_music = input("\nGenerate music for videos? (y/n): ").lower().strip() == 'y'
                
                if test_music:
                    # If there are multiple videos, ask which one to use for music
                    selected_video = generated["videos"][0]  # Default to first video
                    if len(generated["videos"]) > 1:
                        print("\nSelect video to add music to:")
                        for i, video_path in enumerate(generated["videos"]):
                            print(f"{i+1}. {os.path.basename(video_path)}")
                        
                        try:
                            video_choice = int(input("\nEnter the number of the video to use: ")) - 1
                            if 0 <= video_choice < len(generated["videos"]):
                                selected_video = generated["videos"][video_choice]
                        except ValueError:
                            print(f"Invalid choice, using first video")
                    
                    generated["video"] = selected_video  # Set this for the music generation logic
        
        # Ask about generating music after seeing the video
        test_music = False
        if generated["video"]:
            test_music = input("\nGenerate music for this video? (y/n): ").lower().strip() == 'y'
        
        # 3. Generate music if selected
        if test_music:
            print("\n===== GENERATING MUSIC =====")
            
            try:
                print(f"Generating music with prompt: '{test_prompts['music'][:50]}...'")
                
                music_url = api.generate_music(
                    prompt=test_prompts["music"],
                    duration=8,
                    model_version="stereo-large"
                )
                
                if music_url:
                    print("✅ Music generated successfully")
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    music_path = api.download_file(
                        music_url, 
                        output_dir="test_music", 
                        filename=f"test_music_{timestamp}.mp3"
                    )
                    
                    if music_path:
                        generated["music"] = music_path
                        print(f"Music saved to: {music_path}")
                        api.display_media(music_path, "audio")
                    else:
                        print("❌ Failed to download music")
                else:
                    print("❌ Music generation failed")
            except Exception as e:
                print(f"❌ Error generating music: {str(e)}")
        
        # 4. Automatically merge video and music if both are available
        if generated["video"] and generated["music"]:
            print("\n===== MERGING VIDEO AND MUSIC =====")
            
            try:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                merged_path = api.merge_video_audio(
                    generated["video"], 
                    generated["music"],
                    filename=f"merged_{timestamp}.mp4"
                )
                
                if merged_path:
                    generated["merged"] = merged_path
                    print(f"✅ Merged file saved to: {merged_path}")
                    api.display_media(merged_path, "video")
                else:
                    print("❌ Failed to merge video and music")
            except Exception as e:
                print(f"❌ Error merging video and music: {str(e)}")
        
        # Summary at exit
        print("\n===== GENERATED MEDIA SUMMARY =====")
        
        if generated["images"]:
            print("\nImages:")
            for model, path in generated["images"].items():
                print(f"  - {model}: {path}")
        
        if generated.get("threed_models"):
            print("\n3D Models:")
            for path in generated["threed_models"]:
                print(f"  - {path}")
                
        if generated["video"]:
            print(f"\nVideo: {generated['video']}")
            
        if generated["music"]:
            print(f"\nMusic: {generated['music']}")
            
        if generated["merged"]:
            print(f"\nMerged: {generated['merged']}")
    
    # Run the test function
    run_test()
    