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
project_root = current_dir.parents[2]  # Go up 3 levels to reach project root
sys.path.insert(0, str(project_root))

# Import our API wrappers
from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
from src.I_integrations.replicate_API import ReplicateAPI, download_file
from src.I_integrations.tripo_API import TripoAPI  # Import the Tripo API

# Import utility modules
from src.VI_utils.utils import quick_look
from src.VI_utils.video_utils import video_optimise, video_to_gif, get_video_info, video_loop
from src.VI_utils.image_utils import images_to_gif

class VidsGenTurntable:
    """
    VidsGenTurntable: AI-powered Product Marketing Asset Generation
    
    Features:
    1. Taking a product idea prompt and refining it with LLM
    2. Generating styled product image using Replicate
    3. Generating turntable-style motion video from the image (optional)
    4. Creating background music that matches the product's theme (optional)
    5. Generating a 3D model from the image (optional)
    6. Combining everything into a final product video with GIF version
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        replicate_api_key: Optional[str] = None,
        output_dir: str = "products",  # Will be saved in data/output/products/
        llm_model: str = "gpt-4o",
        image_gen_model: str = "flux",
        video_gen_model: str = "wan-i2v-480p",
        turntable: bool = True,
        loop: bool = False,
        should_generate_music: bool = True,
        should_generate_video: bool = True,
        should_generate_threed: bool = False  # Add 3D model generation option
    ):
        # Initialize API clients
        self.llm = OpenAIResponsesAPI(
            api_key=openai_api_key, 
            model=llm_model,
            system_message="You are an AI product marketing assistant"
        )
        self.replicate = ReplicateAPI(api_token=replicate_api_key)
        self.tripo = TripoAPI()  # Initialize Tripo API client
        
        # Settings
        self.output_dir = output_dir  # Base subfolder name
        self.image_gen_model = image_gen_model
        self.video_gen_model = video_gen_model
        self.turntable = turntable  # Add turntable parameter
        self.loop = loop  # Store loop setting
        self.should_generate_music = should_generate_music  # Store music generation setting
        self.should_generate_video = should_generate_video  # Store video generation setting
        self.should_generate_threed = should_generate_threed  # Store 3D model generation setting
        
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
            "model_prompt": "",  # Add model prompt to state
            "image_path": "",
            "image_url": "",
            "video_path": "",
            "video_url": "",
            "music_path": "",
            "music_url": "",
            "model_path": "",  # Add model path to state
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
        # Base output directory
        self.output_base_dir = Path("data/output")
        self.full_output_dir = self.output_base_dir / self.output_dir
        
        # Create base directory if it doesn't exist
        if not self.output_base_dir.exists():
            self.output_base_dir.mkdir(parents=True, exist_ok=True)
            
        # Create products directory if it doesn't exist
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
    
    def _generate_compact_schema(self):
        """Schema for compact project options"""
        return {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "additionalProperties": False},
                            "tagline": {"type": "string", "additionalProperties": False},
                            "description": {"type": "string", "additionalProperties": False},
                            "visual_style": {"type": "string", "additionalProperties": False},
                            "motion_type": {"type": "string", "additionalProperties": False},
                            "music_style": {"type": "string", "additionalProperties": False}
                        },
                        "required": ["name", "tagline", "description", "visual_style", "motion_type", "music_style"],
                        "additionalProperties": False
                    },
                    "additionalProperties": False
                }
            },
            "required": ["options"],
            "additionalProperties": False
        }

    def generate_project_options(self, prompt: str):
        """Generate 3 compact project options using product design guidelines"""
        schema = self._generate_compact_schema()
        
        system_prompt = """
        You are a senior product designer and creative director specializing in immersive product presentations. Your expertise lies in crafting compelling visual and multimedia concepts for marketing campaigns.

        When creating product concept options, follow these expert guidelines:

        **Visual Presentation Principles**
        1. Start with the product type and main subject
        2. Emphasize materials, textures, and unique features
        3. Reference specific visual styles and artistic movements
        4. Include technical specifications for lighting and camera

        **Motion Design Elements**
        - Define motion type (turntable/zoom/pan) with specific timing
        - Specify camera angles and viewpoints (eye-level, high angle, etc.)
        - Include lighting transitions and dynamic elements

        **Audio Aesthetic Guide**
        - Select genre and tempo that matches product personality
        - Define emotional tone that enhances product perception
        - Specify key instrumentation and sound design elements

        Create 3 distinct concepts that position the product in different aesthetic contexts. Each concept should be:
        - Detailed enough to inspire a full production pipeline
        - Visually coherent with matched style, motion, and audio elements
        - Distinctive with clear differentiation between options
        """
        
        user_prompt = f"""
        Input concept: {prompt}
        
        Generate 3 distinct marketing concept options, each with a cohesive creative direction. For each option provide:

        1. Name: Create a distinctive, memorable name (3-7 words) combining descriptive and emotional elements.
        
        2. Tagline: Craft a compelling tagline (5-12 words) using active voice that captures the product's key value proposition.
        
        3. Description: Write a concise paragraph (30-50 words) describing the concept's overall theme and marketing approach.
        
        4. Visual Style: Detail a rich visual aesthetic (40-60 words) referencing at least 2 specific artistic influences, color palette, materials, textures, and lighting approach.
        
        5. Motion Type: Define a specific motion technique (30-40 words) including camera movement, timing, transitions, and perspective shifts.
        
        6. Music Style: Specify a musical direction (30-40 words) including genre, tempo (BPM), instrumentation, mood progression, and emotional quality.

        Make each option tonally distinct - for example:
        - Option 1: Modern/minimalist with technical precision
        - Option 2: Dramatic/emotional with artistic flair
        - Option 3: Lifestyle/aspirational with human connection

        For motion types, choose from approaches like slow-zoom, 360-spin, dolly-zoom, parallax shift, or reveal techniques, but add specific details about implementation.
        """
        
        response = self.llm.structured_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            output_schema=schema,
            temperature=0.75
        )
        
        return response.get("options", [])[:3]

    def generate_prompts(self, selected_option: dict, conversation_history: list):
        """Generate full prompts using comprehensive guidelines"""
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
        Based on the selected concept:

        Name: {selected_option['name']}
        Tagline: {selected_option['tagline']}
        Description: {selected_option['description']}
        Visual Style: {selected_option['visual_style']}
        Motion Type: {selected_option['motion_type']}
        Music Style: {selected_option['music_style']}
        
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
        
        # Store selected option in state for fallback needs
        self.state["selected_option"] = selected_option
        self.state["conversation_history"] = formatted_history
        
        # Try to generate prompts, with fallback to hardcoded defaults if needed
        try:
            response = self.llm.structured_response(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                output_schema=schema,
                message_history=formatted_history,
                temperature=0.65
            )
            
            # Handle error response
            if isinstance(response, dict) and "error" in response:
                print(f"Error generating prompts: {response['error']}")
                # Use fallback prompts
                response = self._get_fallback_prompts(selected_option)
            
            # If any key is missing, use fallbacks for those
            for key in ["image_prompt", "video_prompt", "music_prompt", "model_prompt"]:
                if key not in response or not response[key]:
                    prompt_type = key.split("_")[0]
                    response[key] = self._get_fallback_prompt(prompt_type, selected_option)
                    
        except Exception as e:
            print(f"Exception generating prompts: {e}")
            # Use fallback prompts if generation fails
            response = self._get_fallback_prompts(selected_option)
        
        # Store prompts directly without validation
        self.state["image_prompt"] = response["image_prompt"]
        self.state["video_prompt"] = response["video_prompt"]
        self.state["music_prompt"] = response["music_prompt"]
        self.state["model_prompt"] = response["model_prompt"]  # Store model prompt
        
        # Add description fallbacks for the markdown file 
        if "description" not in self.state or not self.state["description"]:
            self.state["description"] = {
                "product_name": selected_option.get("name", "Product"),
                "tagline": selected_option.get("tagline", ""),
                "description": selected_option.get("description", ""),
                "visual_style": selected_option.get("visual_style", ""),
                "motion_direction": selected_option.get("motion_type", ""),
                "music_style": selected_option.get("music_style", "")
            }
        
        return response
    
    def _get_fallback_prompt(self, prompt_type: str, selected_option: dict):
        """Get a reliable fallback prompt"""
        product_name = selected_option.get('name', 'Product')
        visual_style = selected_option.get('visual_style', 'modern and sleek')
        motion_type = selected_option.get('motion_type', 'smooth movement')
        music_style = selected_option.get('music_style', 'contemporary electronic')
        
        fallbacks = {
            "image": f"Photorealistic product render of {product_name}, {visual_style}, with detailed textures, high-quality materials, reflective surfaces, studio lighting with dramatic rim light accent, professional product photography style, 8k resolution, shallow depth of field focusing on key details, neutral background with subtle gradient",
            
            "video": f"Smooth {'360-degree turntable rotation' if self.turntable else 'camera movement'} showing {product_name} from all angles, 6 second duration, starting slowly then maintaining steady pace, cinematic lighting that emphasizes materials and textures, subtle spotlight following the motion to highlight key features, professional product showcase with high production value, clean background with minimal distractions",
            
            "music": f"Professional {music_style} background music at 110 BPM, with layered synthesizer elements, subtle rhythmic patterns, gradual dynamic progression building to midpoint then elegantly resolving, modern and professional mood that enhances the visual experience, clean production with spatial depth, perfect for product demonstration",
            
            "model": f"Highly detailed 3D model of {product_name}, professional-grade asset with photorealistic materials and textures, {visual_style}, precise geometry with optimized topology, proper scale and proportions, PBR materials with accurate reflections and surface properties, includes high-resolution textures for close-up examination, suitable for product visualization in any 3D environment, ready for AR/VR implementation"
        }
        
        return fallbacks[prompt_type]
    
    def _get_fallback_prompts(self, selected_option: dict):
        """Get all fallback prompts as a dictionary"""
        return {
            "image_prompt": self._get_fallback_prompt("image", selected_option),
            "video_prompt": self._get_fallback_prompt("video", selected_option),
            "music_prompt": self._get_fallback_prompt("music", selected_option),
            "model_prompt": self._get_fallback_prompt("model", selected_option)
        }

    def _save_description_as_md(self):
        """Save project description as Markdown"""
        product_dir = self._create_product_dir()
        md_path = product_dir / f"description_{self.project_id}.md"
        
        md_content = f"""
        # {self.state['description'].get('product_name', 'Unnamed Product')}
        
        ## Project Overview
        **Tagline**: {self.state['description'].get('tagline', '')}
        
        ### Description
        {self.state['description'].get('description', '')}
        
        ### Visual Style
        {self.state['description'].get('visual_style', '')}
        
        ### Motion Direction
        {self.state['description'].get('motion_direction', '')} {'(360° Turntable)' if self.turntable else ''}
        
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
            raise ValueError("Image prompt must be generated before creating image")
            
        prompt = self.state["image_prompt"]
        print(f"\nGenerating product image...")
        print(f"Image prompt: {prompt[:100]}...")
        
        try:
            # Generate the image with 16:9 aspect ratio
            image_output = self.replicate.generate_image(
                prompt=prompt,
                aspect_ratio="16:9"  # Good for marketing display
            )
            
            if not image_output:
                print(f"Failed to generate image")
                return None
                
            # Store the original image URL
            if hasattr(image_output, 'url'):
                original_image_url = image_output.url
            elif isinstance(image_output, dict) and 'url' in image_output:
                original_image_url = image_output['url']
            else:
                original_image_url = str(image_output)
            
            # Download the image
            product_name = self.state["description"].get("product_name", "product").replace(" ", "_")
            filename = f"image_{product_name}_{self.project_id}.png"
            image_path = download_file(
                url=image_output, 
                output_dir=str(self.product_dir),
                filename=filename
            )
            
            if image_path:
                print(f"Downloaded product image: {image_path}")
                self.state["image_path"] = image_path
                self.state["image_url"] = original_image_url
                
                # Preview the image
                self.preview_asset(image_path, "image")
                
                return image_path
            else:
                print(f"Failed to download image")
                return None
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
    
    def generate_video(self):
        """Generate video from product image with turntable motion"""
        if not self.state["image_path"] or not self.state["image_url"]:
            raise ValueError("Image must be generated before creating video")
        
        if not self.state["video_prompt"]:
            print("No specific video prompt, using default turntable motion")
            video_prompt = f"Turntable rotation of the product with smooth motion"
        else:
            video_prompt = self.state["video_prompt"]
        
        print(f"\nGenerating product video...")
        print(f"Video prompt: {video_prompt[:100]}...")
        
        try:
            # Generate the video
            video_url = self.replicate.generate_video(
                prompt=video_prompt,
                model=self.video_gen_model,
                image_url=self.state["image_url"]  # Use the URL string
            )
            
            if not video_url:
                print(f"Failed to generate video")
                return None
                
            # Download the video
            product_name = self.state["description"].get("product_name", "product").replace(" ", "_")
            filename = f"video_{product_name}_{self.project_id}.mp4"
            video_path = download_file(
                url=video_url, 
                output_dir=str(self.product_dir),
                filename=filename
            )
            
            if video_path:
                print(f"Downloaded product video: {video_path}")
                
                # Apply looping if enabled
                if self.loop:
                    print("Creating seamless loop...")
                    looped_path = Path(video_path).with_name(f"{Path(video_path).stem}_looped.mp4")
                    if video_loop(str(video_path), str(looped_path)):
                        video_path = str(looped_path)
                        print(f"Created looped version: {video_path}")
                
                # Create GIF regardless of looping
                gif_path = Path(video_path).with_suffix('.gif')
                if video_to_gif(str(video_path), str(gif_path), fps=15):
                    print(f"Created GIF preview: {gif_path}")
                
                self.state["video_path"] = video_path
                self.state["video_url"] = video_url
                
                # Preview the video
                self.preview_asset(video_path, "video")
                
                return video_path
            else:
                print(f"Failed to download video")
                return None
        except Exception as e:
            print(f"Error generating video: {e}")
            return None
    
    def generate_music(self):
        """Generate background music based on the music prompt"""
        if not self.state["music_prompt"]:
            print("No specific music prompt, using default background music")
            music_prompt = f"Professional background music for a product advertisement"
        else:
            music_prompt = self.state["music_prompt"]
            
        # Use a fixed duration for product videos
        duration_seconds = 5  # Standard turntable video length
            
        print(f"\nGenerating background music ({duration_seconds} seconds)...")
        print(f"Music prompt: {music_prompt[:100]}...")
        
        try:
            # Generate music
            music_url = self.replicate.generate_music(
                prompt=music_prompt,
                duration=duration_seconds
            )
            
            if not music_url:
                print("Failed to generate music")
                return None
                
            # Download the music
            product_name = self.state["description"].get("product_name", "product").replace(" ", "_")
            filename = f"music_{product_name}_{self.project_id}.mp3"
            music_path = download_file(
                url=music_url, 
                output_dir=str(self.product_dir),
                filename=filename
            )
            
            if music_path:
                print(f"Downloaded background music: {music_path}")
                # Store music info
                self.state["music_path"] = music_path
                self.state["music_url"] = music_url
                
                # Preview the music
                self.preview_asset(music_path, "audio")
                
                return music_path
            else:
                print("Failed to download music")
                return None
        except Exception as e:
            print(f"Error generating music: {e}")
            return None
    
    def combine_video_music(self):
        """Combine product video with background music"""
        if not self.state["video_path"]:
            raise ValueError("Video must be generated before combining")
        
        print("\nCombining video with background music...")
        
        try:
            video_path = self.state["video_path"]
            
            # Check if video file exists and is valid
            if not os.path.exists(video_path):
                raise ValueError(f"Video file not found: {video_path}")
            elif os.path.getsize(video_path) == 0:
                raise ValueError(f"Video file is empty: {video_path}")
            
            # Create output path for final video
            product_name = self.state["description"].get("product_name", "product").replace(" ", "_")
            final_filename = f"final_{product_name}_{self.project_id}.mp4"
            final_path = self.product_dir / final_filename
            
            # If music is available, add it to the video
            if self.state.get("music_path") and os.path.exists(self.state["music_path"]):
                music_path = self.state["music_path"]
                
                # Get video duration to determine exact trim length
                video_info = get_video_info(str(video_path))
                video_duration = video_info.get('duration', 0)
                
                if video_duration > 0:
                    print(f"Video duration: {video_duration:.2f} seconds")
                    print(f"Trimming music to match video duration: {video_duration:.2f} seconds")
                    
                    # Create a temporary trimmed audio file
                    temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix='.mp3')
                    os.close(temp_audio_fd)
                    
                    # Trim audio to exactly match video duration
                    trim_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(music_path),
                        "-ss", "0",
                        "-t", str(video_duration),  # Exact duration of the video
                        "-c:a", "copy",
                        temp_audio_path
                    ]
                    
                    print("Trimming music...")
                    trim_result = subprocess.run(trim_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if trim_result.returncode != 0:
                        print(f"Error trimming music: {trim_result.stderr}")
                        # Use original music file if trimming fails
                        temp_audio_path = music_path
                    
                    # Add trimmed music to the video
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
                    audio_result = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if audio_result.returncode != 0:
                        print(f"Error adding music: {audio_result.stderr}")
                        # Fallback to using the video without music
                        import shutil
                        shutil.copy2(video_path, final_path)
                    
                    # Clean up temporary files
                    if os.path.exists(temp_audio_path) and trim_result.returncode == 0:
                        os.unlink(temp_audio_path)
                else:
                    print("Could not determine video duration, using video without music")
                    import shutil
                    shutil.copy2(video_path, final_path)
            else:
                # If no music, just copy the video to the final path
                import shutil
                shutil.copy2(video_path, final_path)
                print("No music available. Creating final video without audio.")
            
            # Create a GIF version of the video
            gif_filename = f"final_{product_name}_{self.project_id}.gif"
            gif_path = self.product_dir / gif_filename
            
            print("Creating GIF version of the video...")
            gif_success = video_to_gif(
                input_path=str(final_path),
                output_path=str(gif_path),
                fps=10,  # Good balance for smooth motion
                resize_factor=75.0,  # 75% of original size
                quality=85,
                optimize_size=True,
                max_colors=256
            )
            
            if gif_success:
                print(f"Created GIF version: {gif_path}")
                self.state["gif_path"] = str(gif_path)
            
            # Store final video info
            self.state["final_video_path"] = str(final_path)
            
            print(f"\n✅ Final product video created: {final_path}")
            
            # Preview the final video
            self.preview_asset(str(final_path), "video")
            
            return {
                "status": "success",
                "final_video": str(final_path),
                "gif": str(gif_path) if gif_success else None
            }
            
        except Exception as e:
            print(f"Error combining video and music: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
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
    
    async def generate_threed_model(self):
        """Generate 3D model from product image and prompt"""
        if not self.state["image_path"] or not self.state["image_url"]:
            raise ValueError("Image must be generated before creating 3D model")
            
        if not self.state["model_prompt"]:
            print("No specific model prompt, using default 3D model description")
            model_prompt = f"Detailed 3D model of the product with accurate materials and textures"
        else:
            model_prompt = self.state["model_prompt"]
            
        print(f"\nGenerating 3D model...")
        print(f"Model prompt: {model_prompt[:100]}...")
        
        try:
            # Create output path for 3D model
            product_name = self.state["description"].get("product_name", "product").replace(" ", "_")
            model_filename = f"model_{product_name}_{self.project_id}.glb"
            model_path = str(self.product_dir / model_filename)
            
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
                await self.tripo.preview_model(generated_model)
                
                return generated_model
            else:
                print("❌ Failed to generate 3D model")
                return None
                
        except Exception as e:
            print(f"Error generating 3D model: {e}")
            return None

    async def run_pipeline(self, prompt: str):
        """Updated pipeline flow with optional asset generation"""
        conversation_history = []
        
        # Stage 1: Generate options
        print("\n>>> Generating Project Options")
        options = self.generate_project_options(prompt)
        conversation_history.append({"user": prompt, "assistant": options})
        
        # Present options
        print("\n==== Available Concepts ====")
        for i, opt in enumerate(options, 1):
            print(f"\n{i}. {opt['name']}")
            print(f"   {opt['tagline']}")
            print(f"   Visual Style: {opt['visual_style']}")
            print(f"   Motion: {opt['motion_type']}")
        
        # Get user selection
        choice = int(input("\nSelect concept (1-3): ")) - 1
        selected = options[choice]
        
        # Stage 2: Generate prompts
        print("\n>>> Generating Detailed Prompts")
        prompts = self.generate_prompts(selected, conversation_history)
        conversation_history.append({"user": "Generate prompts", "assistant": prompts})
        
        # Create product directory
        self._create_product_dir()
        
        # Save description
        md_path = self._save_description_as_md()
        print(f"Saved project description to: {md_path}")
        
        # Proceed with generation
        print("\n>>> Starting Asset Generation")
        try:
            # Start music generation in background if enabled
            music_future = None
            if self.should_generate_music:
                print("Starting music generation in background...")
                music_future = self.executor.submit(self.generate_music)
                self.futures.append(music_future)
            
            # Generate product image (always required)
            print("\n>>> Generating Product Image")
            image_path = self.generate_image()
            if not image_path:
                print("❌ Failed to generate product image. Cannot proceed.")
                return False
                
            # Start parallel tasks for video and 3D model if enabled
            video_future = None
            model_task = None
            
            # Generate video if enabled
            if self.should_generate_video:
                print("\n>>> Starting Video Generation")
                video_future = self.executor.submit(self.generate_video)
                self.futures.append(video_future)
            
            # Generate 3D model if enabled
            if self.should_generate_threed:
                print("\n>>> Starting 3D Model Generation")
                # 3D model generation is async, so we create a task
                model_task = asyncio.create_task(self.generate_threed_model())
            
            # Wait for video if it was generated
            video_path = None
            if video_future:
                print("Waiting for video generation to complete...")
                video_path = video_future.result()
                if not video_path:
                    print("⚠️ Video generation did not complete successfully.")
            
            # Wait for music if it was generated
            music_path = None
            if music_future:
                print("Waiting for music generation to complete...")
                music_path = music_future.result()
                if not music_path:
                    print("⚠️ Music generation did not complete successfully.")
            
            # Wait for 3D model if it was generated
            if model_task:
                print("Waiting for 3D model generation to complete...")
                try:
                    model_path = await model_task
                    if model_path:
                        print(f"✅ 3D model saved to: {model_path}")
                    else:
                        print("⚠️ 3D model generation did not complete successfully.")
                except Exception as e:
                    print(f"❌ Error during 3D model generation: {str(e)}")
            
            # Combine video and music if both were generated
            if self.should_generate_video and video_path and self.should_generate_music and music_path:
                print("\n>>> Combining Video and Music")
                result = self.combine_video_music()
                video_success = result.get("status") == "success"
                if not video_success:
                    print("⚠️ Failed to combine video and music.")
            else:
                # If either video or music was disabled, we still consider this step a success
                video_success = True
                
            # Update the markdown with final paths
            updated_md = self._save_description_as_md()
            print(f"Updated project description with all asset paths: {updated_md}")
                
            # Consider overall success based on what was enabled
            success = True
            
            # If any critical enabled feature failed, mark as unsuccessful
            if not image_path:
                success = False  # Image is always required
            if self.should_generate_video and not video_path:
                success = False
            if self.should_generate_music and not music_path:
                success = False
            if self.should_generate_threed and not model_task:
                success = False
                
            return success
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    # Get user preferences
    print("\n===== PRODUCT MARKETING ASSET GENERATOR =====")
    print("Please configure your generation options:\n")
    
    # Core asset options
    should_generate_video = input("Generate video animation? (y/n): ").lower().strip() == 'y'
    should_generate_music = input("Generate background music? (y/n): ").lower().strip() == 'y'
    should_generate_threed = input("Generate 3D model? (y/n): ").lower().strip() == 'y'
    
    # Only ask about video options if video is enabled
    turntable = False
    loop = False
    if should_generate_video:
        turntable = input("Enable 360° turntable rotation? (y/n): ").lower().strip() == 'y'
        loop = input("Enable seamless video looping? (y/n): ").lower().strip() == 'y'
    
    # Initialize with settings
    generator = VidsGenTurntable(
        turntable=turntable, 
        loop=loop,
        should_generate_music=should_generate_music,
        should_generate_video=should_generate_video,
        should_generate_threed=should_generate_threed
    )
    
    # List of product prompts 
    PRODUCT_PROMPTS = [
        "Meta Quest Pro: A premium mixed reality headset with high-resolution displays, advanced eye tracking, and sleek design for immersive experiences",
        
        "Apple Vision Ring: A minimalist smart ring with haptic feedback, gesture controls, and health sensors in brushed titanium and matte black finishes",
        
        "Meta Ray-Ban Smart Glasses: Stylish eyewear with integrated cameras, spatial audio, and voice assistant in a classic frame design",
        
        "Horizon Home Robot: A sleek, white cylindrical home assistant robot with expressive display face, mobility base, and ambient lighting accents",
        
        "Apple AirPods Studio: Premium over-ear headphones with spatial audio, active noise cancellation, and seamless aluminum construction with mesh cushioning"
    ]
    
    # Print all prompts with pretty formatting
    print("\n==== Available Product Prompts ====")
    for i, prompt in enumerate(PRODUCT_PROMPTS, 1):
        print(f"\n{i}. {prompt}")
    
    # Let user select a prompt
    choice = 0
    while choice < 1 or choice > len(PRODUCT_PROMPTS):
        try:
            choice = int(input(f"\nSelect a product prompt (1-{len(PRODUCT_PROMPTS)}): "))
            if choice < 1 or choice > len(PRODUCT_PROMPTS):
                print(f"Please enter a number between 1 and {len(PRODUCT_PROMPTS)}")
        except ValueError:
            print("Please enter a valid number")
    
    selected_prompt = PRODUCT_PROMPTS[choice-1]
    
    print("\n==== Selected Product ====")
    print(f"{selected_prompt}")
    
    # Create a summary of enabled features
    enabled_features = []
    enabled_features.append("Product Image")
    if should_generate_video:
        enabled_features.append("Video Animation" + (" (with turntable)" if turntable else ""))
        if loop:
            enabled_features.append("Video Looping")
    if should_generate_music:
        enabled_features.append("Background Music")
    if should_generate_threed:
        enabled_features.append("3D Model")
    
    print("\n==== Enabled Features ====")
    for feature in enabled_features:
        print(f"✓ {feature}")
    
    print("\nStarting generation pipeline...")
    
    # Run the asset generation pipeline using asyncio
    try:
        success = asyncio.run(generator.run_pipeline(selected_prompt))
    except KeyboardInterrupt:
        print("\n\nGeneration cancelled by user.")
        success = False
    except Exception as e:
        print(f"\n\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        success = False
    finally:
        # Always clean up resources
        generator.cleanup()
    
    # Final message after everything completes
    if success:
        print("\n" + "="*70)
        print("🎉 CONGRATULATIONS! YOUR PRODUCT ASSETS ARE READY!")
        print("="*70)
        print("\nAll the assets have been saved automatically.")
        print("You can find them in the product directory shown above.")
        
        # List what was generated based on user choices
        print("\nGenerated assets:")
        print("✓ Product concept image")
        if should_generate_video:
            print("✓ Product video animation")
        if should_generate_music:
            print("✓ Background music")
        if should_generate_video and should_generate_music:
            print("✓ Final video with music")
            print("✓ GIF version")
        if should_generate_threed:
            print("✓ 3D model (GLB format)")
        
        print("\nThank you for using the Product Marketing Asset Generator!")
    else:
        print("\n" + "="*70)
        print("⚠️ There were some issues during asset creation.")
        print("Please check the error messages above.")
        print("="*70)

if __name__ == "__main__":
    main()
