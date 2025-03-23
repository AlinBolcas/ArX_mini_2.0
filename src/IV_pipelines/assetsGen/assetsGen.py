import os
import json
import time
import sys
import uuid
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import tempfile
import subprocess
import pprint
import asyncio

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
# Navigate up to the project root (3 levels up from src/IV_pipelines/assetsGen/)
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our API wrappers
from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
from src.I_integrations.replicate_API import ReplicateAPI, download_file
from src.I_integrations.tripo_API import TripoAPI

# Import utility modules
from src.VI_utils.utils import quick_look
from src.VI_utils.video_utils import video_to_gif, get_video_info, video_loop

class AssetsGen:
    """
    AssetsGen: Core asset generation functionality for product marketing
    
    Features:
    1. Generate styled product image using Replicate
    2. Generate turntable-style motion video from the image (optional)
    3. Create background music that matches the product's theme (optional)
    4. Generate a 3D model from the image (optional)
    5. Combine assets into a final product video with GIF version
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        replicate_api_key: Optional[str] = None,
        output_dir: str = "assets",
        llm_model: str = "gpt-4o-mini",
        image_gen_model: str = "flux",
        video_gen_model: str = "wan-i2v-480p",
        turntable: bool = True,
        loop: bool = False,
        should_generate_music: bool = True,
        should_generate_video: bool = True,
        should_generate_threed: bool = False
    ):
        # Initialize API clients
        self.llm = OpenAIResponsesAPI(
            api_key=openai_api_key, 
            model=llm_model,
            system_message="You are an AI product marketing assistant"
        )
        self.replicate = ReplicateAPI(api_token=replicate_api_key)
        self.tripo = TripoAPI()
        
        # Settings
        self.output_dir = output_dir
        self.image_gen_model = image_gen_model
        self.video_gen_model = video_gen_model
        self.turntable = turntable
        self.loop = loop
        self.should_generate_music = should_generate_music
        self.should_generate_video = should_generate_video
        self.should_generate_threed = should_generate_threed
        
        # Set up output dirs
        self._setup_output_dirs()
        
        # Project state storage
        self.project_id = self._generate_project_id()
        self.state = {
            "project_id": self.project_id,
            "prompt": "",
            "description": {},
            "image_prompt": "",
            "video_prompt": "",
            "music_prompt": "",
            "model_prompt": "",
            "image_path": "",
            "image_url": "",
            "video_path": "",
            "video_url": "",
            "music_path": "",
            "music_url": "",
            "model_path": "",
            "final_video_path": "",
            "gif_path": "",
            "selected_option": {},
            "conversation_history": []
        }
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self.futures = []
    
    def _generate_project_id(self) -> str:
        """Generate a unique project ID combining timestamp and UUID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{timestamp}_{short_uuid}"
    
    def __del__(self):
        """Cleanup method to properly shut down the executor"""
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources, wait for pending tasks, and shut down the executor"""
        try:
            # Wait for any pending futures to complete
            self.wait_for_futures()
            
            # Shutdown the executor
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)
                print("Thread pool executor shut down.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def _setup_output_dirs(self):
        """Set up output directory structure"""
        # Base output directory - FIX: Using absolute path to avoid path duplication
        self.output_base_dir = Path("data/output").resolve()  # Use resolve to get absolute path
        self.full_output_dir = self.output_base_dir / self.output_dir
        
        # Create base directory if it doesn't exist
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            
        # Create assets directory if it doesn't exist
        if not self.full_output_dir.exists():
            self.full_output_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"Output will be stored in {self.full_output_dir}")
    
    def _create_product_dir(self):
        """Create a unique directory for this product"""
        self.product_dir = self.full_output_dir / self.project_id
        if not self.product_dir.exists():
            self.product_dir.mkdir(parents=True, exist_ok=True)
        return self.product_dir
    
    def wait_for_futures(self):
        """Wait for all pending futures to complete"""
        if not self.futures:
            return
            
        print(f"Waiting for {len(self.futures)} pending tasks to complete...")
        for future in concurrent.futures.as_completed(self.futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in background task: {e}")
                
        self.futures = []
        print("All background tasks completed.")
        
    def generate_prompts(self, product_description: dict, conversation_history: list = None):
        """Generate full prompts using comprehensive guidelines"""
        if conversation_history is None:
            conversation_history = []
            
        schema = {
            "type": "object",
            "properties": {
                "image_prompt": {
                    "type": "string",
                    "description": "Detailed image generation prompt",
                    "additionalProperties": False
                },
                "video_prompt": {
                    "type": "string",
                    "description": f"Comprehensive video generation prompt with {'360° turntable' if self.turntable else 'motion'} details",
                    "additionalProperties": False
                },
                "music_prompt": {
                    "type": "string",
                    "description": "Detailed music generation prompt",
                    "additionalProperties": False
                },
                "model_prompt": {
                    "type": "string", 
                    "description": "Detailed 3D model generation prompt",
                    "additionalProperties": False
                }
            },
            "required": ["image_prompt", "video_prompt", "music_prompt", "model_prompt"],
            "additionalProperties": False
        }

        system_prompt = f"""
        You are a master prompt engineer specializing in multimodal content generation. Your expertise enables you to craft precise, evocative prompts that produce exceptional results across image, video, music, and 3D model generation platforms.

        Follow these expert guidelines for prompt creation:

        **Image Prompt Structure**
        Create a single, focused paragraph with comma-separated elements:
        - Begin with image type (photograph, render, illustration) and main subject
        - Include 3+ specific material descriptors (textures, surfaces, finishes)
        - Reference 2+ artistic influences or style movements
        - Specify lighting conditions, color palette, and mood
        - Add technical parameters: resolution, perspective, focal length
        - Example: "Photorealistic product render of a smartwatch, brushed titanium case with sapphire crystal and matte silicone band, holographic interface glowing in cool blue tones, studio lighting with dramatic rim light accent, minimal white background, 8k resolution, macro lens perspective with shallow depth of field"

        **Video Prompt Architecture**
        Construct a comprehensive motion description:
        - {'Specify precise 360° rotation parameters (speed, direction, timing)' if self.turntable else 'Define exact camera movement paths and timing'}
        - Detail lighting evolution throughout the sequence
        - Include specific duration markers (seconds)
        - Reference focal length, depth of field, and perspective shifts
        - Example: {"Smooth 360° turntable rotation of product with 6-second duration, starting slowly then maintaining steady pace, dramatic lighting transitioning from cool blue to warm amber as rotation progresses, maintaining sharp focus on product details with subtle depth-of-field shift highlighting key features, soft spotlight following the rotation to emphasize materials and textures" if self.turntable else "Cinematic dolly-zoom revealing product details over 6 seconds, starting with wide establishing shot then smoothly pushing in to highlight key features, lighting transitions from soft ambient to focused spotlighting, shallow depth of field gradually deepening to reveal context"}

        **Music Prompt Composition**
        Design a precise audio direction:
        - Specify genre, exact tempo (BPM), and key signature
        - Detail 2-3 primary instrument layers and their roles
        - Chart the emotional progression and intensity arc
        - Include production style and spatial characteristics
        - Example: "Ambient electronic music at 92 BPM in F minor, layered with ethereal synthesizer pads, subtle percussive elements, and occasional piano accents, building from minimal atmospheric introduction to moderately complex midpoint with added bass elements, then gradually reducing to simpler conclusion, spatial reverb creating sense of depth, professionally produced with clean mix and subtle dynamic compression"

        **3D Model Prompt Architecture**
        Create a comprehensive 3D asset description:
        - Begin with model type (character, product, environment) and subject
        - Specify exact materials, textures, and surface properties
        - Detail geometric complexity, proportions, and scale
        - Include lighting environment and presentation context
        - Reference specific 3D style (photorealistic, stylized, low-poly)
        - Example: "Highly detailed 3D model of a futuristic smartwatch, precision-modeled with accurate proportions and ergonomic design, featuring brushed titanium case with subtle anodized finish, sapphire crystal screen with anti-reflective properties, matte silicone band with microperforations for breathability, photorealistic materials with proper subsurface scattering on translucent elements, optimized geometry suitable for product visualization, presented on a subtle shadow-casting surface with neutral studio lighting"
        """

        user_prompt = f"""
        Based on the provided product details:

        Name: {product_description.get('name', 'Product')}
        Tagline: {product_description.get('tagline', '')}
        Description: {product_description.get('description', '')}
        Visual Style: {product_description.get('visual_style', '')}
        Motion Type: {product_description.get('motion_type', '')}
        Music Style: {product_description.get('music_style', '')}
        
        Create four specialized generation prompts that work together as a cohesive suite:

        1. IMAGE PROMPT:
        - Craft a detailed, comma-separated prompt for a still image visualization
        - Include at least 5 material/texture descriptions
        - Reference specific lighting setup and technical parameters
        - Maintain the established visual style while optimizing for image generation
        - Length: 50-100 words in a single paragraph

        2. VIDEO PROMPT:
        - Design a comprehensive motion sequence description
        - Specify {'turntable rotation parameters' if self.turntable else 'camera movement details'} with exact timing
        - Include lighting transitions and focus techniques
        - Ensure the motion complements the product's key features
        - Length: 50-100 words in a single paragraph

        3. MUSIC PROMPT:
        - Create a detailed audio direction with specific tempo (BPM)
        - Include at least 3 instrument/sound elements
        - Specify emotional progression and audio production style
        - Ensure the audio perfectly enhances the visual experience
        - Length: 50-100 words in a single paragraph

        4. 3D MODEL PROMPT:
        - Craft a comprehensive 3D model description
        - Detail exact materials, textures, and surface properties
        - Specify geometric complexity, proportions, and scale
        - Include technical specifications for model quality
        - Ensure the 3D representation matches the product's visual identity
        - Length: 50-100 words in a single paragraph

        Focus on technical precision and creative coherence. Each prompt should be specialized for its medium while maintaining a unified aesthetic direction.
        """
        
        # Format conversation history properly for the API
        formatted_history = []
        for item in conversation_history:
            if isinstance(item, dict) and "user" in item and "assistant" in item:
                formatted_history.append({"role": "user", "content": item["user"]})
                # Check if assistant value is a string or dict/list
                if isinstance(item["assistant"], str):
                    formatted_history.append({"role": "assistant", "content": item["assistant"]})
                else:
                    # Convert complex objects to strings
                    formatted_history.append({"role": "assistant", "content": json.dumps(item["assistant"])})
        
        # Store product description in state for future reference
        self.state["description"] = product_description
        self.state["conversation_history"] = formatted_history
        
        # Generate prompts with robust error handling
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Adjust temperature based on retry attempts (more deterministic on retries)
                temperature = max(0.3, 0.65 - (retry_count * 0.15))
                
                response = self.llm.structured_response(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    output_schema=schema,
                    message_history=formatted_history,
                    temperature=temperature
                )
                
                # Check for valid response structure
                if not isinstance(response, dict):
                    print(f"Invalid response format (attempt {retry_count+1}/{max_retries}), retrying...")
                    retry_count += 1
                    continue
                
                # Check for error response
                if "error" in response:
                    print(f"API error: {response['error']} (attempt {retry_count+1}/{max_retries}), retrying...")
                    retry_count += 1
                    continue
                
                # Validate that all required fields are present and non-empty
                missing_fields = []
                for key in ["image_prompt", "video_prompt", "music_prompt", "model_prompt"]:
                    if key not in response or not response[key] or len(response[key].strip()) < 10:
                        missing_fields.append(key)
                
                if missing_fields:
                    print(f"Missing or empty fields: {', '.join(missing_fields)} (attempt {retry_count+1}/{max_retries}), retrying...")
                    retry_count += 1
                    continue
                
                # If we made it here, the response is valid
                break
                
            except Exception as e:
                print(f"Error generating prompts (attempt {retry_count+1}/{max_retries}): {str(e)}")
                retry_count += 1
                # Short delay before retry
                time.sleep(1)
        
        # If we exhausted retries and still don't have valid prompts, create basic ones
        if retry_count >= max_retries:
            print("Failed to generate prompts after multiple attempts. Creating basic prompts.")
            product_name = product_description.get('name', 'Product')
            response = {
                "image_prompt": f"Photorealistic product render of {product_name} on a clean background with professional lighting, showing detailed materials and textures, high-resolution 8K image with balanced composition and studio lighting setup, product showcased from its best angle with attention to detail and craftsmanship.",
                
                "video_prompt": f"Professional {product_name} showcase with {'smooth 360-degree turntable rotation' if self.turntable else 'elegant camera movement'} lasting 6 seconds, maintaining consistent lighting that highlights product features and materials, gradual motion with subtle lighting transitions to emphasize different aspects of the design.",
                
                "music_prompt": f"Professional background music for {product_name} marketing video at 110 BPM, modern and clean production with subtle rhythmic elements and atmospheric textures, building to a satisfying conclusion that enhances the product presentation.",
                
                "model_prompt": f"Detailed 3D model of {product_name} with accurate scale and proportions, photorealistic materials and textures, optimized geometry with clean topology, suitable for product visualization and marketing purposes."
            }
        
        # Store prompts in state
        self.state["image_prompt"] = response["image_prompt"]
        self.state["video_prompt"] = response["video_prompt"]
        self.state["music_prompt"] = response["music_prompt"]
        self.state["model_prompt"] = response["model_prompt"]
        
        # Save description to markdown
        self._save_description_as_md()
        
        return response
    
    def _save_description_as_md(self):
        """Save project description as Markdown"""
        product_dir = self._create_product_dir()
        md_path = product_dir / f"description_{self.project_id}.md"
        
        md_content = f"""
        # {self.state['description'].get('name', 'Unnamed Product')}
        
        ## Project Overview
        **Tagline**: {self.state['description'].get('tagline', '')}
        
        ### Description
        {self.state['description'].get('description', '')}
        
        ### Visual Style
        {self.state['description'].get('visual_style', '')}
        
        ### Motion Direction
        {self.state['description'].get('motion_type', '')} {'(360° Turntable)' if self.turntable else ''}
        
        ### Music Style
        {self.state['description'].get('music_style', '')}
        
        ## Generation Prompts
        **Image Prompt**  
        {self.state['image_prompt']}
        
        **Video Prompt**  
        {self.state['video_prompt']}
        
        **Music Prompt**  
        {self.state['music_prompt']}
        
        **3D Model Prompt**  
        {self.state['model_prompt']}
        
        ## Assets
        - Image: `{self.state.get('image_path', '')}`
        - Video: `{self.state.get('video_path', '')}`
        - Music: `{self.state.get('music_path', '')}`
        - Final video: `{self.state.get('final_video_path', '')}`
        - GIF version: `{self.state.get('gif_path', '')}`
        - 3D Model: `{self.state.get('model_path', '')}`
        """
        
        with open(md_path, 'w') as f:
            f.write(md_content)
            
        return md_path
        
    def generate_image(self):
        """Generate product image based on image prompt"""
        if not self.state["image_prompt"]:
            print("Creating a basic image prompt...")
            product_name = self.state["description"].get("name", "Product")
            self.state["image_prompt"] = f"Photorealistic render of {product_name} with professional lighting and high-quality materials"
            
        prompt = self.state["image_prompt"]
        print(f"\nGenerating product image...")
        print(f"Image prompt: {prompt[:100]}...")
        
        # Generate the image with 16:9 aspect ratio
        image_output = self.replicate.generate_image(
            prompt=prompt,
            aspect_ratio="16:9"  # Good for marketing display
        )
        
        if not image_output:
            print("Failed to generate image")
            return None
            
        # Store the original image URL
        if hasattr(image_output, 'url'):
            original_image_url = image_output.url
        elif isinstance(image_output, dict) and 'url' in image_output:
            original_image_url = image_output['url']
        else:
            original_image_url = str(image_output)
        
        # Download the image
        product_name = self.state["description"].get("name", "product").replace(" ", "_")
        filename = f"image_{product_name}_{self.project_id}.png"
        
        # FIX: Ensure we're using the correct absolute path string
        output_dir_str = str(self.product_dir.resolve())  # Use resolve() to get absolute path
        
        # FIX: Add retry logic for downloading
        max_retries = 3
        for attempt in range(max_retries):
            try:
                image_path = download_file(
                    url=image_output, 
                    output_dir=output_dir_str,  # Use the resolved path
                    filename=filename
                )
                if image_path:
                    break
                else:
                    print(f"Download attempt {attempt+1}/{max_retries} failed. Retrying...")
                    time.sleep(2)  # Add delay between retries
            except Exception as e:
                print(f"Download error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)  # Add delay between retries
        
        if image_path:
            # FIX: Ensure image_path is an absolute path
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)
            
            print(f"✅ Downloaded product image: {image_path}")
            self.state["image_path"] = image_path
            self.state["image_url"] = original_image_url
            
            # Preview the image
            self.preview_asset(image_path, "image")
            
            return image_path
        else:
            print("Failed to download image after multiple attempts")
            return None
    
    def generate_video(self):
        """Generate video from product image with turntable motion"""
        if not self.state["image_path"] or not self.state["image_url"]:
            print("Image must be generated before creating video")
            return None
        
        if not self.state["video_prompt"]:
            print("Creating a basic video prompt...")
            product_name = self.state["description"].get("name", "Product")
            motion_type = "360-degree turntable rotation" if self.turntable else "cinematic camera movement"
            self.state["video_prompt"] = f"Professional {product_name} showcase with smooth {motion_type}, crisp lighting, and elegant motion"
        
        video_prompt = self.state["video_prompt"]
        print(f"\nGenerating product video...")
        print(f"Video prompt: {video_prompt[:100]}...")
        
        # Generate the video
        video_url = self.replicate.generate_video(
            prompt=video_prompt,
            model=self.video_gen_model,
            image_url=self.state["image_url"]
        )
        
        if not video_url:
            print("Failed to generate video")
            return None
            
        # Download the video
        product_name = self.state["description"].get("name", "product").replace(" ", "_")
        filename = f"video_{product_name}_{self.project_id}.mp4"
        
        # FIX: Ensure we're using the correct absolute path string
        output_dir_str = str(self.product_dir.resolve())
        
        # FIX: Add robust retry logic for video download
        max_retries = 5  # More retries for video which is larger
        for attempt in range(max_retries):
            try:
                video_path = download_file(
                    url=video_url, 
                    output_dir=output_dir_str,
                    filename=filename
                )
                if video_path:
                    break
                else:
                    print(f"Video download attempt {attempt+1}/{max_retries} failed. Retrying in {(attempt+1)*2}s...")
                    time.sleep((attempt+1) * 2)  # Increasing delay between retries
            except Exception as e:
                print(f"Video download error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep((attempt+1) * 2)  # Increasing delay between retries
        
        if video_path:
            # FIX: Ensure video_path is an absolute path
            if not os.path.isabs(video_path):
                video_path = os.path.abspath(video_path)
            
            print(f"✅ Downloaded product video: {video_path}")
            
            # Apply looping if enabled
            if self.loop:
                print("Creating seamless loop...")
                looped_path = Path(video_path).with_name(f"{Path(video_path).stem}_looped.mp4")
                # FIX: Better error handling for looping
                try:
                    if video_loop(str(video_path), str(looped_path)):
                        video_path = str(looped_path)
                        print(f"✅ Created looped version: {video_path}")
                    else:
                        print("Could not create looped version, using original video")
                except Exception as e:
                    print(f"Error creating video loop: {e}")
                    print("Using original video instead")
            
            # Create GIF regardless of looping
            try:
                gif_path = Path(video_path).with_suffix('.gif')
                if video_to_gif(str(video_path), str(gif_path), fps=15):
                    print(f"✅ Created GIF preview: {gif_path}")
                    self.state["gif_path"] = str(gif_path)
                else:
                    print("Could not create GIF version")
            except Exception as e:
                print(f"Error creating GIF: {e}")
            
            self.state["video_path"] = video_path
            self.state["video_url"] = video_url
            
            # Preview the video
            self.preview_asset(video_path, "video")
            
            return video_path
        else:
            print("Failed to download video after multiple attempts")
            return None
    
    def generate_music(self):
        """Generate background music based on the music prompt"""
        if not self.state.get("music_prompt"):
            print("Creating a basic music prompt...")
            product_name = self.state["description"].get("name", "product")
            self.state["music_prompt"] = f"Professional background music for {product_name} advertisement, modern and upbeat"
            
        music_prompt = self.state["music_prompt"]    
        # Use a fixed duration for product videos
        duration_seconds = 5  # Standard turntable video length
            
        print(f"\nGenerating background music ({duration_seconds} seconds)...")
        print(f"Music prompt: {music_prompt[:100]}...")
        
        # Generate music 
        music_url = self.replicate.generate_music(
            prompt=music_prompt,
            duration=duration_seconds,
            model_version="stereo-large"
        )
        
        if not music_url:
            print("Failed to generate music")
            return None
            
        # Download the music
        product_name = self.state["description"].get("name", "product").replace(" ", "_")
        filename = f"music_{product_name}_{self.project_id}.mp3"
        music_path = download_file(
            url=music_url, 
            output_dir=str(self.product_dir),
            filename=filename
        )
        
        if music_path:
            print(f"✅ Downloaded background music: {music_path}")
            # Store music info
            self.state["music_path"] = music_path
            self.state["music_url"] = music_url
            
            # Preview the music
            self.preview_asset(music_path, "audio")
            
            return music_path
        else:
            print("Failed to download music")
            return None
            
    async def generate_threed_model(self):
        """Generate 3D model from product image and prompt"""
        if not self.state["image_path"] or not self.state["image_url"]:
            print("Image must be generated before creating 3D model")
            return None
            
        if not self.state["model_prompt"]:
            print("Creating a basic 3D model prompt...")
            product_name = self.state["description"].get("name", "Product")
            self.state["model_prompt"] = f"Detailed 3D model of {product_name} with accurate materials, textures, and proportions, photorealistic quality"
            
        model_prompt = self.state["model_prompt"]
        print(f"\nGenerating 3D model...")
        print(f"Model prompt: {model_prompt[:100]}...")
        
        # Create output path for 3D model
        product_name = self.state["description"].get("name", "product").replace(" ", "_")
        model_filename = f"model_{product_name}_{self.project_id}.glb"
        model_path = str(self.product_dir.resolve() / model_filename)  # FIX: Use resolve() for absolute path
        
        # FIX: Add retry logic and better error handling for 3D model generation
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Generate the 3D model using Tripo API
                generated_model = await self.tripo.generate_threed(
                    prompt=model_prompt,
                    image_path=self.state["image_path"],
                    output_path=model_path,
                    texture_quality="detailed",
                    pbr=True,
                    auto_size=True
                )
                
                if generated_model:
                    print(f"✅ Generated 3D model: {generated_model}")
                    self.state["model_path"] = generated_model
                    
                    # Preview the 3D model
                    try:
                        await self.tripo.preview_model(generated_model)
                    except Exception as e:
                        print(f"Note: Could not preview 3D model: {e}")
                    
                    return generated_model
                else:
                    print(f"Failed to generate 3D model on attempt {attempt+1}/{max_retries}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {(attempt+1)*3} seconds...")
                        time.sleep((attempt+1) * 3)
            except Exception as e:
                print(f"Error generating 3D model (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {(attempt+1)*3} seconds...")
                    time.sleep((attempt+1) * 3)
        
        print("Failed to generate 3D model after multiple attempts")
        return None
    
    def combine_video_music(self):
        """Combine product video with background music"""
        if not self.state["video_path"]:
            print("Video must be generated before combining with music")
            return {"status": "error", "error": "No video available"}
        
        print("\nCombining video with background music...")
        
        video_path = self.state["video_path"]
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return {"status": "error", "error": "Video file not found"}
            
        # Create output path for final video
        product_name = self.state["description"].get("name", "product").replace(" ", "_")
        final_filename = f"final_{product_name}_{self.project_id}.mp4"
        final_path = self.product_dir / final_filename
        
        # If music is available, add it to the video
        if self.state.get("music_path") and os.path.exists(self.state["music_path"]):
            music_path = self.state["music_path"]
            print(f"Using music: {music_path}")
            
            # Get video duration
            video_info = get_video_info(str(video_path))
            video_duration = video_info.get('duration', 0)
            
            if video_duration > 0:
                print(f"Video duration: {video_duration:.2f} seconds")
                print(f"Trimming music to match video duration")
                
                # Create a temporary trimmed audio file
                temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix='.mp3')
                os.close(temp_audio_fd)
                
                # Trim audio to match video duration
                trim_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(music_path),
                    "-ss", "0",
                    "-t", str(video_duration),
                    "-c:a", "copy",
                    temp_audio_path
                ]
                
                print("Trimming music...")
                subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Add music to video
                audio_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-i", str(temp_audio_path),
                    "-map", "0:v",  # Use video from first input
                    "-map", "1:a",  # Use audio from second input
                    "-c:v", "copy",  # Copy video codec
                    "-c:a", "aac",  # Ensure audio is in a compatible format
                    "-b:a", "192k",
                    str(final_path)
                ]
                
                print("Adding music to video...")
                subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Clean up temporary files
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
            else:
                print("Could not determine video duration, using video without music")
                import shutil
                shutil.copy2(video_path, final_path)
        else:
            # If no music, just copy the video to the final path
            print("No music available. Creating final video without audio.")
            import shutil
            shutil.copy2(video_path, final_path)
        
        # Create a GIF version of the final video
        gif_filename = f"final_{product_name}_{self.project_id}.gif"
        gif_path = self.product_dir / gif_filename
        
        print("Creating GIF version of the video...")
        gif_success = video_to_gif(
            input_path=str(final_path),
            output_path=str(gif_path),
            fps=10,
            resize_factor=75.0,
            quality=85,
            optimize_size=True
        )
        
        if gif_success:
            print(f"✅ Created GIF version: {gif_path}")
            self.state["gif_path"] = str(gif_path)
        
        # Store final video info
        self.state["final_video_path"] = str(final_path)
        print(f"✅ Final product video created: {final_path}")
        
        # Preview the final video
        self.preview_asset(str(final_path), "video")
        
        return {
            "status": "success",
            "final_video": str(final_path),
            "gif": str(gif_path) if gif_success else None
        }
    
    def preview_asset(self, asset_path: str, asset_type: str = "image"):
        """
        Preview an asset using quick_look utility.
        
        Args:
            asset_path: Path to the asset file
            asset_type: Type of asset ("image", "video", or "audio")
        """
        try:
            print(f"Previewing {asset_type}: {asset_path}")
            quick_look(asset_path)
        except Exception as e:
            print(f"Error previewing asset: {e}")
    
    def set_product_description(self, description: dict):
        """Set product description directly from external source"""
        self.state["description"] = description
        # Ensure we have a project directory
        self._create_product_dir()
        return self._save_description_as_md()
    
    def create_description_from_idea(self, idea_text: str):
        """Enrich a user's idea with descriptive details while maintaining the original concept"""
        print(f"\nEnhancing prompt with descriptive details...")
        
        system_prompt = """
        You are a creative prompt enhancer who specializes in adding descriptive details to concept ideas.
        
        Your task is to take a brief concept idea and enrich it with specific descriptive details while
        STRICTLY preserving the original concept and intent.
        
        DO NOT create a paragraph or narrative. Instead, produce a single descriptive prompt with added details:
        - Add specific visual characteristics (colors, materials, textures)
        - Include style references or artistic influences when appropriate
        - Mention lighting, perspective, mood where relevant
        - Suggest technical aspects (level of detail, rendering style)
        
        FORMAT YOUR RESPONSE AS A SINGLE DESCRIPTIVE PROMPT, similar to:
        "Futuristic sports car, metallic blue with silver accents, sleek aerodynamic design, carbon fiber details, 
        dramatic studio lighting, photorealistic rendering, ultra-detailed, cinematic composition"
        
        Keep the original essence intact but enhance with specific, useful details.
        DO NOT add elaborate storytelling or completely change the concept.
        """
        
        user_prompt = f"""
        Enhance the following concept with descriptive details while preserving its core intent:
        
        {idea_text}
        """
        
        # Generate the enhanced description
        try:
            # Use the OpenAIResponsesAPI response method
            enhanced_prompt = self.llm.response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.7
            )
            
            # Clean up the description
            enhanced_prompt = enhanced_prompt.strip() if isinstance(enhanced_prompt, str) else idea_text
            
            print("\n✅ Enhanced prompt with descriptive details")
            print(f"Original: {idea_text}")
            print(f"Enhanced: {enhanced_prompt}")
            
            # Create a simple state object with the description
            # Extract a name from the first part of the idea
            product_name = " ".join(idea_text.split()[:3])
            self.state["description"] = {
                "name": product_name,  # Use first few words as name
                "description": enhanced_prompt,
                # Basic placeholder values for required fields
                "tagline": f"{product_name}",
                "visual_style": "Professional visualization",
                "motion_type": "Dynamic presentation",
                "music_style": "Complementary soundtrack"
            }
            
            return enhanced_prompt
            
        except Exception as e:
            print(f"Error enhancing description: {e}")
            # Fallback to using the original idea
            print("Using original prompt without enhancement")
            product_name = " ".join(idea_text.split()[:3])
            self.state["description"] = {
                "name": product_name,
                "description": idea_text,
                "tagline": f"{product_name}",
                "visual_style": "Professional visualization",
                "motion_type": "Dynamic presentation",
                "music_style": "Complementary soundtrack"
            }
            
            return idea_text


if __name__ == "__main__":
    # Test the core functionality with user input
    async def test_assets_gen():
        print("\n===== ASSET GENERATION TOOL =====")
        generator = None
        
        try:
            # Get user idea for product
            print("\n1. What product or character would you like to create assets for?")
            idea = input("Describe your idea: ")
            
            if not idea:
                print("No idea provided. Using a default example.")
                idea = "realistic green caiman character, standing on two legs, appealing design, full-body, American beer advertisement, highly detailed, lifting a beer bottle and smiling, keep the same camera angle"
                print(f"Using default: {idea}")
                
            # Initialize the generator
            generator = AssetsGen(
                output_dir="assets",
            )
            
            # Ask if user wants to enhance the description
            enhance_prompt = input("\nWould you like to enhance your prompt with descriptive details? (y/n): ").lower().strip() in ['y', 'yes', '']
            
            description = idea
            if enhance_prompt:
                # Generate enhanced description from idea
                print("\n=== ENHANCING PROMPT ===")
                description = generator.create_description_from_idea(idea)
            else:
                print("\nUsing original prompt without enhancement")
                # Set the state with the original description
                product_name = " ".join(idea.split()[:3])
                generator.state["description"] = {
                    "name": product_name,
                    "description": idea,
                    "tagline": f"{product_name}",
                    "visual_style": "Professional visualization",
                    "motion_type": "Dynamic presentation", 
                    "music_style": "Complementary soundtrack"
                }
            
            # Ask which assets to generate
            print("\n2. Which assets would you like to generate?")
            gen_image = input("Generate image? (y/n): ").lower().strip() in ['y', 'yes', '']
            gen_video = input("Generate video? (y/n): ").lower().strip() in ['y', 'yes', '']
            gen_music = input("Generate background music? (y/n): ").lower().strip() in ['y', 'yes', '']
            gen_3d = input("Generate 3D model? (y/n): ").lower().strip() in ['y', 'yes', '']
            
            # Video options
            turntable = False
            loop = False
            if gen_video:
                turntable = input("Use 360° turntable rotation for video? (y/n): ").lower().strip() in ['y', 'yes', '']
                loop = input("Create seamless video loop? (y/n): ").lower().strip() in ['y', 'yes', '']
                
            # Update generator with user choices
            generator.turntable = turntable
            generator.loop = loop
            generator.should_generate_video = gen_video
            generator.should_generate_music = gen_music
            generator.should_generate_threed = gen_3d
            
            # Generate prompts
            print("\n=== GENERATING ASSET PROMPTS ===")
            prompts = generator.generate_prompts(generator.state["description"])
            
            # Always generate image as it's needed for other assets
            print("\n=== GENERATING IMAGE ===")
            image_path = generator.generate_image()
            
            if not image_path:
                print("\n❌ Could not generate image. Cannot proceed with other assets.")
                return
            
            print(f"\n✅ Image generated: {image_path}")
            
            # Start parallel asset generation based on user choices
            futures = []
            
            # Generate music if requested
            music_future = None
            if gen_music:
                print("\n=== GENERATING MUSIC ===")
                music_future = generator.executor.submit(generator.generate_music)
                futures.append(music_future)
                generator.futures.append(music_future)
            
            # Generate video if requested
            video_future = None
            if gen_video:
                print("\n=== GENERATING VIDEO ===")
                video_future = generator.executor.submit(generator.generate_video)
                futures.append(video_future)
                generator.futures.append(video_future)
            
            # Generate 3D model if requested
            model_task = None
            if gen_3d:
                print("\n=== GENERATING 3D MODEL ===")
                model_task = asyncio.create_task(generator.generate_threed_model())
            
            # Wait for assets to complete
            video_path = None
            music_path = None
            
            if video_future:
                video_path = video_future.result()
                if video_path:
                    print(f"\n✅ Video generated: {video_path}")
            
            if music_future:
                music_path = music_future.result()
                if music_path:
                    print(f"\n✅ Music generated: {music_path}")
            
            if model_task:
                model_path = await model_task
                if model_path:
                    print(f"\n✅ 3D model generated: {model_path}")
            
            # Combine video and music if both were generated
            if video_path and music_path:
                print("\n=== COMBINING VIDEO AND MUSIC ===")
                result = generator.combine_video_music()
                if result.get("status") == "success":
                    print(f"\n✅ Final video created: {result.get('final_video')}")
            
            # Save and display final results
            md_path = generator._save_description_as_md()
            
            print("\n\n=== GENERATION COMPLETE ===")
            print(f"All assets saved to: {generator.product_dir}")
            print(f"Project description: {md_path}")
            
            # List generated assets
            assets = []
            if generator.state.get("image_path"):
                assets.append(f"Image: {generator.state.get('image_path')}")
            if generator.state.get("video_path"):
                assets.append(f"Video: {generator.state.get('video_path')}")
            if generator.state.get("music_path"):
                assets.append(f"Music: {generator.state.get('music_path')}")
            if generator.state.get("model_path"):
                assets.append(f"3D Model: {generator.state.get('model_path')}")
            if generator.state.get("final_video_path"):
                assets.append(f"Final Video: {generator.state.get('final_video_path')}")
            
            if assets:
                print("\nGenerated assets:")
                for asset in assets:
                    print(f"✓ {asset}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            # Clean up resources
            if generator:
                generator.cleanup()
    
    # Run the async test
    asyncio.run(test_assets_gen()) 