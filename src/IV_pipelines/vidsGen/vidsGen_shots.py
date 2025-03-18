import os
import json
import time
import sys
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import tempfile
import subprocess
import pprint

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[2]  # Go up 3 levels to reach project root
sys.path.insert(0, str(project_root))

# Import our API wrappers
from src.I_integrations.openai_API import OpenAIAPI
from src.I_integrations.replicate_API import ReplicateAPI, download_file

# Import utility modules
from src.VI_utils.utils import quick_look
from src.VI_utils.video_utils import (
    video_optimise, video_to_gif, get_video_info
)

class VidsGenSimple:
    """
    VidsGenSimple: Streamlined AI Video Generation Pipeline
    
    A simplified version of VidsGen that focuses on:
    1. Taking a single prompt and number of shots
    2. Creating 3 art direction options
    3. Generating images, videos and music in parallel
    4. Combining everything into a final video
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        replicate_api_key: Optional[str] = None,
        output_dir: str = "vidsGen",  # Changed to just the subfolder name, as download_file will add data/output/
        llm_model: str = "gpt-4o",
        image_gen_model: str = "flux",
        video_gen_model: str = "wan-i2v-480p"
    ):
        # Initialize API clients
        self.llm = OpenAIAPI(
            api_key=openai_api_key, 
            model=llm_model,
            system_message="You are an AI video production assistant"
        )
        self.replicate = ReplicateAPI(api_token=replicate_api_key)
        
        # Settings
        self.output_dir = output_dir  # Just store the subfolder name
        self.image_gen_model = image_gen_model
        self.video_gen_model = video_gen_model
        
        # Set up output dirs (now just for reference)
        self._setup_output_dirs()
        
        # Project state storage
        self.project_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.state = {
            "project_id": self.project_id,
            "prompt": "",
            "num_shots": 2,
            "art_directions": [],
            "selected_art_direction": None,
            "scripts": [],
            "selected_script": None,
            "shot_prompts": [],
            "video_direction": "",
            "music_prompt": "",
            "generated_images": [],
            "generated_videos": [],
            "music_url": "",
            "music_path": "",
            "final_video_path": ""
        }
        
        # Thread pool for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.futures = []
    
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
        """Record the output directory paths for reference only"""
        # The download_file function will handle directory creation
        # Just keep track of the full path for reference
        self.output_base_dir = Path("data/output")
        self.full_output_dir = self.output_base_dir / self.output_dir
        print(f"Output will be stored in {self.full_output_dir}")
    
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
    
    def generate_art_directions(self, prompt: str, num_shots: int = 1, num_options: int = 3):
        """Generate art direction options based on the input prompt"""
        self.state["prompt"] = prompt
        self.state["num_shots"] = max(1, num_shots)  # Ensure at least 1 shot
        
        # Craft detailed prompt for art direction generation with explicit JSON format
        prompt_text = f"""
        Based on the following video concept:
        
        "{prompt}"
        
        Generate EXACTLY {num_options} distinct, creative art direction options for an AI-generated video with {num_shots} shots.
        
        You MUST provide EXACTLY {num_options} options - no more, no fewer.
        
        Each art direction MUST include ALL of the following fields (all are required):
        1. A numeric id (starting from 1)
        2. A captivating title that captures the essence of the direction
        3. A detailed description of the visual approach
        4. The overall visual style and aesthetics
        5. The mood and emotional tone
        6. Pacing and rhythm suggestions
        7. Color palette recommendations
        8. Creative inspirations or references
        9. Technical considerations for AI generation
        
        Return your answer in this exact JSON format:
        
        ```json
        {{
          "art_directions": [
            {{
              "id": 1,
              "title": "Title for first direction",
              "description": "Detailed description",
              "visual_style": "Visual style details",
              "mood": "Mood description",
              "pacing": "Pacing description",
              "color_palette": "Color palette details",
              "inspiration": "Creative inspirations",
              "technical_considerations": "Technical notes"
            }},
            {{
              "id": 2,
              "title": "Title for second direction",
              ...
            }},
            {{
              "id": 3,
              "title": "Title for third direction",
              ...
            }}
          ]
        }}
        ```
        
        DO NOT omit any of these fields - all are required for each art direction.
        Make each direction meaningfully different from the others to give real creative options.
        """
        
        # Generate art directions using OpenAI
        print("Generating art direction options...")
        try:
            response = self.llm.chat_completion(
                user_prompt=prompt_text,
                temperature=0.8,
                system_prompt="You are an expert creative director who creates precise, structured art direction options for videos. You always return valid JSON with exactly the requested fields and number of options."
            )
            
            # Parse JSON response
            try:
                # Find JSON in the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    # Try to extract from code blocks
                    import re
                    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
                    if json_blocks:
                        parsed_response = json.loads(json_blocks[0])
                    else:
                        raise ValueError("No JSON found in response")
                        
                # Store generated art directions
                self.state["art_directions"] = parsed_response.get("art_directions", [])
                
                # Verify we got the expected number of options
                if len(self.state["art_directions"]) != num_options:
                    print(f"Warning: Received {len(self.state['art_directions'])} art directions instead of {num_options}")
                    
                # Number each art direction for selection (ensuring they're numbered correctly)
                for i, direction in enumerate(self.state["art_directions"]):
                    direction["id"] = i + 1
                    
                print(f"Generated {len(self.state['art_directions'])} art direction options")
                
                return self.state["art_directions"]
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", response[:500])  # Print first 500 chars of response
                return None
        except Exception as e:
            print(f"Error generating art directions: {e}")
            return None
        
    def select_art_direction(self, selection_id: int):
        """Select an art direction from the generated options"""
        if not self.state["art_directions"]:
            raise ValueError("No art directions available to select from")
        
        # Find the selected art direction by ID
        selected = None
        for direction in self.state["art_directions"]:
            if direction["id"] == selection_id:
                selected = direction
                break
        
        if not selected:
            raise ValueError(f"Art direction with ID {selection_id} not found")
        
        # Store selected art direction
        self.state["selected_art_direction"] = selected
        print(f"Selected art direction: {selected['title']}")
        
        return selected
    
    def generate_script_and_prompts(self):
        """Generate script, image prompts, video and music directions in a streamlined process"""
        if not self.state["selected_art_direction"]:
            raise ValueError("An art direction must be selected before generating scripts")
            
        art_direction = self.state["selected_art_direction"]
        num_shots = self.state["num_shots"]
        
        # Craft detailed prompt for unified generation with explicit JSON format 
        prompt_text = f"""
        As the creative director for this project, generate a complete plan for a {num_shots}-shot video based on:
        
        TITLE: {art_direction['title']}
        DESCRIPTION: {art_direction['description']}
        VISUAL STYLE: {art_direction['visual_style']}
        MOOD: {art_direction['mood']}
        PACING: {art_direction['pacing']}
        COLOR PALETTE: {art_direction['color_palette']}
        
        Original concept: "{self.state['prompt']}"
        
        Your task is to create a unified creative plan with EXACTLY {num_shots} shots formatted as a JSON object with these sections:
        
        1. "script" object with:
           - title: A captivating title
           - description: Overview description
           - narrative_continuity: {"How story flows from shot to shot" if num_shots > 1 else "The narrative flow within the shot"}
           - visual_continuity: {"How visual elements connect" if num_shots > 1 else "Visual flow within the shot"}
           - shots: Array of EXACTLY {num_shots} shot objects, each with:
             * shot_number: {"1 through " + str(num_shots) if num_shots > 1 else "1"}
             * description: Detailed visual description
             * shot_type: Type of shot (close-up, wide shot, etc.)
             * camera_movement: Camera movement (MUST include movement)
             * duration_seconds: Between 3-10 seconds
             * {"transition: Transition to next shot" if num_shots > 1 else ""}
             * motion_elements: Specific details of movement
           - music_direction: Music guidance
           - estimated_duration: Total duration estimate
        
        2. "image_prompts" array with {num_shots} items, each with:
           - shot_number: Matching the script
           - image_prompt: Detailed cinema-quality prompt
           - continuity_elements: {"Elements connecting to other shots" if num_shots > 1 else "Elements providing visual continuity"}
           - motion_indicators: How/where motion will occur
        
        3. "video_direction" object with:
           - overall_pacing: Pacing and rhythm
           - motion_direction: Animation guidance
           - {"continuity_guidance: Continuity between shots" if num_shots > 1 else ""}
           - {"transition_guidance: Transition specifications" if num_shots > 1 else ""}
        
        4. "music_prompt": Text prompt for AI music generation
        
        IMPORTANT:
        - Motion is ESSENTIAL in every shot - static frames are forbidden
        - {"Maintain perfect CONTINUITY between shots" if num_shots > 1 else "Create dynamic and engaging motion within the shot"}
        - Make prompts extremely detailed for best AI generation
        - {"Ensure the narrative flows coherently across all " + str(num_shots) + " shots" if num_shots > 1 else "Create a complete and coherent narrative in a single shot"}
        - Return ONLY valid JSON with no additional text or explanation
        - Use double quotes for all JSON fields and strings
        """
        
        # Generate comprehensive plan
        print("Generating complete video plan...")
        try:
            response = self.llm.chat_completion(
                user_prompt=prompt_text,
                system_prompt="You are an expert creative director who creates detailed, structured video production plans in JSON format. You always return valid JSON with exactly the requested fields and number of shots.",
                temperature=0.7
            )
            
            # Parse JSON response
            try:
                # Find JSON in the response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    parsed_response = json.loads(json_str)
                else:
                    # Try to extract from code blocks
                    import re
                    json_blocks = re.findall(r'```(?:json)?\s*([\s\S]*?)```', response)
                    if json_blocks:
                        parsed_response = json.loads(json_blocks[0])
                    else:
                        raise ValueError("No JSON found in response")
                
                # Store all components
                script = parsed_response.get("script", {})
                self.state["selected_script"] = script
                
                image_prompts = parsed_response.get("image_prompts", [])
                self.state["shot_prompts"] = image_prompts
                
                video_direction = parsed_response.get("video_direction", {})
                self.state["video_direction"] = video_direction
                
                music_prompt = parsed_response.get("music_prompt", "")
                self.state["music_prompt"] = music_prompt
                
                print(f"Generated complete plan with:")
                print(f"- Script: {script.get('title', 'Untitled')}")
                print(f"- {len(image_prompts)} image prompts")
                print(f"- Video direction")
                print(f"- Music prompt")
                
                return True
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print("Raw response:", response[:500])  # Print first 500 chars of response
                return False
        except Exception as e:
            print(f"Error generating script: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_images(self):
        """Generate images for each shot in parallel"""
        if not self.state["shot_prompts"]:
            raise ValueError("Shot prompts must be generated before creating images")
            
        print("\nGenerating images for all shots in parallel...")
        
        # Setup for parallel processing
        image_futures = []
        shot_prompts = self.state["shot_prompts"]
        
        # Function for parallel execution
        def generate_image_for_shot(shot_data):
            shot_num = shot_data["shot_number"]
            prompt = shot_data["image_prompt"]
            
            print(f"Generating image for shot {shot_num}...")
            
            # Generate the image with fixed 16:9 aspect ratio
            image_output = self.replicate.generate_image(
                prompt=prompt,
                aspect_ratio="16:9"  # Fixed aspect ratio
            )
            
            if not image_output:
                print(f"Failed to generate image for shot {shot_num}")
                return None
                
            # Store the original image URL (extract string URL if needed)
            if hasattr(image_output, 'url'):
                original_image_url = image_output.url
            elif isinstance(image_output, dict) and 'url' in image_output:
                original_image_url = image_output['url']
            else:
                original_image_url = str(image_output)
            
            # Download the image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"image_{shot_num:02d}_{self.project_id}_{timestamp}.jpg"
            output_dir = f"{self.output_dir}/images"  # Use relative path for download_file
            
            # Download the image
            image_path = download_file(url=image_output, output_dir=output_dir, filename=filename)
            
            if image_path:
                print(f"Downloaded image for shot {shot_num}: {image_path}")
                return {
                    "shot_number": shot_num,
                    "prompt": prompt,
                    "path": image_path,
                    "url": original_image_url  # Store the URL string
                }
            else:
                print(f"Failed to download image for shot {shot_num}")
                return None
        
        # Submit all image generation tasks to thread pool
        for shot_data in shot_prompts:
            future = self.executor.submit(generate_image_for_shot, shot_data)
            image_futures.append(future)
            
        # Wait for all images to be generated
        generated_images = []
        for future in concurrent.futures.as_completed(image_futures):
            result = future.result()
            if result:
                generated_images.append(result)
                
                # Preview the image
                self.preview_asset(result["path"], "image")
        
        # Sort images by shot number
        generated_images.sort(key=lambda x: x["shot_number"])
        
        # Store generated images
        self.state["generated_images"] = generated_images
        print(f"Generated {len(generated_images)} images")
        
        return generated_images
    
    def generate_music(self):
        """Generate background music based on the music prompt"""
        if not self.state["music_prompt"]:
            print("No music prompt available, using default")
            music_prompt = f"Background music for a video about {self.state['prompt']}"
        else:
            music_prompt = self.state["music_prompt"]
            
        # Calculate more accurate duration based on shot durations
        # Each shot produces around 5 seconds of video
        if self.state["selected_script"] and self.state["selected_script"].get("shots"):
            # Use actual shot durations if available
            total_duration = sum(shot.get("duration_seconds", 5) for shot in self.state["selected_script"]["shots"])
            
            # Add a small buffer but don't go too long
            # We'll trim to exact length when combining videos
            duration_seconds = min(30, total_duration + 2)
        else:
            # If no script shots available, base on number of shots
            num_shots = self.state.get("num_shots", 2)
            duration_seconds = min(30, num_shots * 5 + 2)  # ~5 seconds per shot + small buffer
            
        print(f"\nGenerating music ({duration_seconds} seconds)...")
        print(f"Music prompt: {music_prompt}")
        
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
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"music_{self.project_id}_{timestamp}.mp3"
            output_dir = f"{self.output_dir}/music"  # Use relative path for download_file
            
            # Download the music using the proper function
            music_path = download_file(url=music_url, output_dir=output_dir, filename=filename)
            
            if music_path:
                print(f"Downloaded music: {music_path}")
                # Store music info
                self.state["music_url"] = music_url
                self.state["music_path"] = music_path
                
                # Preview the music
                self.preview_asset(music_path, "audio")
                
                return music_path
            else:
                print("Failed to download music")
                return None
        except Exception as e:
            print(f"Error generating music: {e}")
            return None
    
    def generate_videos(self):
        """Generate videos from images in parallel"""
        if not self.state["generated_images"]:
            raise ValueError("Images must be generated before creating videos")
            
        images = self.state["generated_images"]
        shot_prompts = self.state["shot_prompts"]
        script_shots = self.state["selected_script"]["shots"] if self.state["selected_script"] else []
        video_direction = self.state.get("video_direction", {})
        
        # Create shot context map for better prompts
        shot_context = {}
        for shot in shot_prompts:
            shot_num = shot["shot_number"]
            shot_context[shot_num] = {
                "prompt": shot.get("image_prompt", ""),
                "motion_indicators": shot.get("motion_indicators", "")
            }
            
        # Add script details
        for shot in script_shots:
            shot_num = shot.get("shot_number")
            if shot_num in shot_context:
                shot_context[shot_num].update({
                    "camera_movement": shot.get("camera_movement", ""),
                    "motion_elements": shot.get("motion_elements", ""),
                    "shot_type": shot.get("shot_type", "")
                })
        
        # Function for parallel execution
        def generate_video_for_shot(image_info):
            shot_num = image_info["shot_number"]
            image_url = image_info["url"]  # This should be the original web URL
            
            # Get URL string from FileOutput object if needed
            if hasattr(image_url, 'url'):
                image_url = image_url.url
            elif isinstance(image_url, dict) and 'url' in image_url:
                image_url = image_url['url']
            
            # Validate the image URL
            if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
                print(f"Error: Invalid image URL for shot {shot_num}: {image_url}")
                print(f"The URL must be a web URL starting with http:// or https://")
                return None
                
            # Get context for this shot
            context = shot_context.get(shot_num, {})
            motion = context.get("motion_indicators", "")
            camera = context.get("camera_movement", "")
            
            # Create enhanced prompt
            if motion or camera:
                video_prompt = f"{context.get('prompt', '')} Movement: {motion} Camera: {camera}"
            else:
                video_prompt = f"{context.get('prompt', '')} with smooth camera motion"
                
            # Add overall video direction
            if video_direction:
                video_prompt += f" Following direction: {video_direction.get('motion_direction', '')}"
                
            print(f"Generating video for shot {shot_num}...")
            
            # Target duration for shot
            shot_duration = 5  # Default to 5 seconds per shot
            
            # If we have script shot info, use its duration
            for shot in script_shots:
                if shot.get("shot_number") == shot_num:
                    shot_duration = min(5, shot.get("duration_seconds", 5))  # Cap at 5 seconds
                    break
            
            # Generate the video
            try:
                # We don't need to pass custom parameters as the ReplicateAPI already 
                # has good defaults for the WAN models (81 frames at 16fps = ~5 seconds)
                video_url = self.replicate.generate_video(
                    prompt=video_prompt,
                    model=self.video_gen_model,
                    image_url=image_url  # Use the extracted URL string
                )
                
                if not video_url:
                    print(f"Failed to generate video for shot {shot_num}")
                    return None
                    
                # Download the video
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"video_{shot_num:02d}_{self.project_id}_{timestamp}.mp4"
                output_dir = f"{self.output_dir}/videos"  # Use relative path for download_file
                
                # Download the video using the proper function
                video_path = download_file(url=video_url, output_dir=output_dir, filename=filename)
                
                if video_path:
                    print(f"Downloaded video for shot {shot_num}: {video_path}")
                    
                    # Preview the video
                    self.preview_asset(video_path, "video")
                    
                    return {
                        "shot_number": shot_num,
                        "path": video_path,
                        "url": video_url,
                        "prompt": video_prompt
                    }
                else:
                    print(f"Failed to download video for shot {shot_num}")
                    return None
            except Exception as e:
                print(f"Error generating video for shot {shot_num}: {e}")
                return None
        
        # Submit all video generation tasks to thread pool
        print("\nGenerating videos from images in parallel...")
        video_futures = []
        
        for image_info in images:
            future = self.executor.submit(generate_video_for_shot, image_info)
            video_futures.append(future)
            
        # Wait for all videos to be generated
        generated_videos = []
        for future in concurrent.futures.as_completed(video_futures):
            result = future.result()
            if result:
                generated_videos.append(result)
        
        # Sort videos by shot number
        generated_videos.sort(key=lambda x: x["shot_number"])
        
        # Store generated videos
        self.state["generated_videos"] = generated_videos
        print(f"Generated {len(generated_videos)} videos")
        
        return generated_videos
    
    def combine_videos(self):
        """Combine all generated video clips into one final video with background music"""
        if not self.state["generated_videos"]:
            raise ValueError("Videos must be generated before combining")
        
        print("\nCombining videos and adding background music...")
        
        try:
            # Sort videos by shot number
            sorted_videos = sorted(self.state["generated_videos"], key=lambda x: x["shot_number"])
            video_paths = [video["path"] for video in sorted_videos]
            
            # Check that all video files exist and are valid
            missing_videos = []
            for i, path in enumerate(video_paths):
                if not os.path.exists(path):
                    missing_videos.append(f"Shot {sorted_videos[i]['shot_number']}: {path} (file not found)")
                elif os.path.getsize(path) == 0:
                    missing_videos.append(f"Shot {sorted_videos[i]['shot_number']}: {path} (file is empty)")
            
            if missing_videos:
                print("❌ Error: Some video files are missing or invalid:")
                for msg in missing_videos:
                    print(f"  - {msg}")
                print("Cannot combine videos. Please check the error messages above.")
                return {"status": "error", "error": "Missing or invalid video files"}
            
            # Estimate total duration (each shot is ~5 seconds)
            estimated_duration = len(sorted_videos) * 5
            print(f"Estimated final video duration: ~{estimated_duration} seconds")
            
            # Ensure all paths are absolute by converting string paths to Path objects if needed
            video_paths = [str(Path(path)) for path in video_paths]
            
            # Create output path for final video
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            final_filename = f"final_{self.project_id}_{timestamp}.mp4"
            output_dir = f"{self.output_dir}/final"  # Use relative path for download_file
            
            # Ensure output directory exists in data/output/vidsGen/final
            base_dir = Path("data/output")
            if not base_dir.exists():
                base_dir.mkdir(parents=True, exist_ok=True)
                
            # Create specific media directory
            final_dir = base_dir / output_dir
            if not final_dir.exists():
                final_dir.mkdir(parents=True, exist_ok=True)
                
            final_path = final_dir / final_filename
            
            # Create a temporary file with file list for ffmpeg
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                for video_path in video_paths:
                    # Ensure absolute paths are used and properly formatted for ffmpeg
                    abs_path = str(Path(video_path).absolute())
                    # Write path with proper escaping for ffmpeg
                    f.write(f"file '{abs_path}'\n")
                file_list_path = f.name
                
            print(f"File list for ffmpeg created at: {file_list_path}")
            print(f"Video paths to concatenate:")
            for i, path in enumerate(video_paths):
                print(f"  {i+1}. {path}")
            
            # Create temporary output for combined video (before adding music)
            temp_combined_fd, temp_combined_path = tempfile.mkstemp(suffix='.mp4')
            os.close(temp_combined_fd)
            
            # Run ffmpeg to concatenate videos
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", file_list_path,
                "-c", "copy",
                temp_combined_path
            ]
            
            # Print the command for debugging
            cmd_str = " ".join(concat_cmd)
            print(f"Running command: {cmd_str}")
            
            print("Concatenating videos...")
            result = subprocess.run(concat_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            if result.returncode != 0:
                print(f"Error concatenating videos. Exit code: {result.returncode}")
                print(f"Error output: {result.stderr}")
                
                # Check if the error is about missing files and suggest verification
                if "No such file or directory" in result.stderr:
                    print("\nSome files may not be accessible to ffmpeg. Let's verify their existence:")
                    for path in video_paths:
                        exists = os.path.exists(path)
                        size = os.path.getsize(path) if exists else 0
                        print(f" - {path}: {'✅ Exists' if exists else '❌ Missing'} ({size} bytes)")
                
                # Print file list content for debugging
                print("\nContent of file list:")
                with open(file_list_path, 'r') as f:
                    print(f.read())
                    
                return {"status": "error", "error": "Failed to concatenate videos"}
                
            # Wait for music generation to complete if it's running in background
            music_path = None
            if hasattr(self, 'music_future') and self.music_future:
                try:
                    print("Waiting for music generation to complete...")
                    # Remove timeout to wait as long as needed for music generation
                    music_path = self.music_future.result()  
                    if music_path:
                        self.state["music_path"] = music_path
                        print(f"Music generated successfully: {music_path}")
                    else:
                        print("⚠️ Music generation failed or was not completed.")
                except Exception as e:
                    print(f"⚠️ Error getting music generation result: {e}")
                    print("Continuing without music...")
            
            # If music is available, add it to the video
            if self.state.get("music_path") and os.path.exists(self.state["music_path"]):
                music_path = self.state["music_path"]
                
                # Get video duration to determine exact trim length
                video_info = get_video_info(str(temp_combined_path))
                video_duration = video_info.get('duration', 0)
                
                if video_duration > 0:
                    print(f"Actual video duration: {video_duration:.2f} seconds (each shot is ~5 seconds)")
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
                        "-i", str(temp_combined_path),
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
                        # Fallback to using the concatenated video without music
                        import shutil
                        shutil.copy2(temp_combined_path, final_path)
                    
                    # Clean up temporary files
                    os.unlink(temp_combined_path)
                    if trim_result.returncode == 0:  # Only remove if we created it successfully
                        os.unlink(temp_audio_path)
                else:
                    print("Could not determine video duration, using video without music")
                    import shutil
                    shutil.copy2(temp_combined_path, final_path)
            else:
                # If no music, just rename the temp file to the final path
                import shutil
                shutil.move(temp_combined_path, final_path)
                print("No music available. Creating video without audio.")
            
            # Create a GIF preview of the combined video
            gif_filename = f"preview_{self.project_id}_{timestamp}.gif"
            gif_path = final_dir / gif_filename
            
            print("Creating GIF preview of final video...")
            gif_success = video_to_gif(
                input_path=str(final_path),
                output_path=str(gif_path),
                fps=5,  # Lower FPS for smaller file size
                resize_factor=50.0,  # Half size for preview
                quality=70,
                optimize_size=True,
                sample_every=4,  # Sample fewer frames
                max_colors=128
            )
            
            if gif_success:
                print(f"Created GIF preview: {gif_path}")
            
            # Store final video info
            self.state["final_video_path"] = final_path
            
            print(f"\n✅ Combined video created: {final_path}")
            
            # Preview the final video
            self.preview_asset(final_path, "video")
            
            return {
                "status": "success",
                "final_video": final_path,
                "preview_gif": gif_path if gif_success else None
            }
            
        except Exception as e:
            print(f"Error combining videos: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up temporary file
            if 'file_list_path' in locals() and os.path.exists(file_list_path):
                os.unlink(file_list_path)
    
    def preview_asset(self, asset_path: str, asset_type: str = "image"):
        """
        Preview an asset using quick_look utility.
        
        Args:
            asset_path: Path to the asset file
            asset_type: Type of asset ("image", "video", or "audio")
        """
        print(f"Previewing {asset_type}: {asset_path}")
        quick_look(asset_path)
    
    def run_pipeline(self, prompt: str, num_shots: int = 1):
        """Run the entire video generation pipeline with a single command"""
        try:
            print("\n" + "="*70)
            print("🎬 VidsGenSimple: Streamlined AI Video Generation Pipeline")
            print("="*70)
            
            print(f"\nInput Prompt: {prompt}")
            print(f"Number of Shots: {num_shots}")
            
            # Step 1: Generate art directions
            print("\n>>> STEP 1: Generating Art Direction Options")
            art_directions = self.generate_art_directions(prompt, num_shots)
            
            if not art_directions:
                print("❌ Failed to generate art directions. Exiting.")
                return False
            
            # Display art directions
            print("\nArt Direction Options:")
            for direction in art_directions:
                print(f"\n{direction['id']}. {direction['title']}")
                print(f"   Description: {direction['description']}")
                print(f"   Visual Style: {direction['visual_style']}")
                print(f"   Mood: {direction['mood']}")
                print(f"   Color Palette: {direction['color_palette']}")
            
            # Step 2: User selects art direction
            while True:
                try:
                    selection = int(input(f"\nSelect art direction (1-{len(art_directions)}): "))
                    self.select_art_direction(selection)
                    break
                except (ValueError, IndexError) as e:
                    print(f"❌ Invalid selection: {e}. Please try again.")
            
            # Step 3: Generate script and prompts
            print("\n>>> STEP 3: Generating Script, Image Prompts, and Directions")
            success = self.generate_script_and_prompts()
            
            if not success:
                print("❌ Failed to generate script and prompts. Exiting.")
                return False
            
            # Display script info
            print("\nScript Information:")
            script = self.state["selected_script"]
            print(f"Title: {script['title']}")
            print(f"Description: {script['description']}")
            print(f"Number of Shots: {len(script['shots'])}")
            
            # Step 4: Generate assets in parallel
            # Start music generation in background and store the future
            print("\n>>> STEP 4a: Starting Music Generation in Background")
            self.music_future = self.executor.submit(self.generate_music)
            self.futures.append(self.music_future)
            
            # Generate images and wait for them
            print("\n>>> STEP 4b: Generating Images for All Shots")
            self.generate_images()
            
            # Generate videos from images without waiting for music
            print("\n>>> STEP 5: Generating Videos from Images")
            videos = self.generate_videos()
            
            if not videos or len(videos) == 0:
                print("❌ Failed to generate any videos. Please check the error messages above.")
                print("Most likely, the image URLs couldn't be used for video generation. Ensure they're valid web URLs.")
                return False
            
            # Combine everything into final video (will wait for music if needed)
            print("\n>>> STEP 6: Combining Videos with Music")
            print("(Note: Music is optional - videos will be combined even if music generation fails)")
            result = self.combine_videos()
            
            if result and result.get("status") == "success":
                print("\n" + "="*70)
                print("🎉 VidsGenSimple - Process Complete!")
                print("="*70)
                print(f"\nFinal video path: {result.get('final_video')}")
                
                # Make sure we play the final result with a clear message
                final_video_path = result.get("final_video")
                gif_path = result.get("preview_gif")
                
                print("\n🎬 FINAL RESULT - Playing completed video...")
                if os.path.exists(final_video_path):
                    self.preview_asset(final_video_path, "video")
                    
                    # Also show the gif preview if available
                    if gif_path and os.path.exists(gif_path):
                        print("\n📱 GIF Preview (for sharing):")
                        self.preview_asset(gif_path, "image")
                else:
                    print(f"❌ Error: Final video file not found at {final_video_path}")
                    
                return True
            else:
                print("\n❌ Failed to create final video.")
                return False
            
        except Exception as e:
            print(f"\n❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # Clean up resources
            self.cleanup()

def main():
    # List of product prompts for character creation/animation tools
    PRODUCT_PROMPTS = [
        "An advertisement for 'ArX Character Generator', Arvolve's revolutionary AI tool that creates production-ready 3D characters with complete rigging and textures in minutes, eliminating weeks of manual work for studios and independent creators.",
        
        "A showcase of 'Neural Animator', Arvolve's AI-powered animation system that transforms static character models into fluid, lifelike animations with customizable styles ranging from realistic motion to stylized cartoon movements.",
        
        "A demonstration of 'Character Evolution Engine', Arvolve's generative design technology that helps artists create unique creature and character concepts by exploring vast design spaces while maintaining artistic control over the final output.",
        
        "An overview of 'RigMaster AI', Arvolve's automated rigging solution that instantly generates production-ready character rigs with advanced facial controls and natural deformations for any humanoid or creature model.",
        
        "A presentation of 'Arvolve Character Pipeline', an end-to-end AI-powered workflow that handles concept design, 3D modeling, texturing, rigging, and animation previews for film and game production teams."
    ]
    
    # Print all prompts with pretty printing
    print("\n==== Available Product Prompts ====")
    pp = pprint.PrettyPrinter(width=100, compact=False)
    for i, prompt in enumerate(PRODUCT_PROMPTS, 1):
        print(f"\n{i}. ", end='')
        pp.pprint(prompt)
    
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
    
    # Let user select number of shots
    num_shots = 0
    while num_shots < 1 or num_shots > 5:
        try:
            num_shots = int(input("\nEnter number of shots (1-5): "))
            if num_shots < 1 or num_shots > 5:
                print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    print("\n==== Selected Options ====")
    print(f"Prompt: ", end='')
    pp.pprint(selected_prompt)
    print(f"Number of shots: {num_shots}")
    
    # Confirm before proceeding
    confirm = input("\nProceed with these options? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Exiting program.")
        return
    
    # Initialize and run the video generation pipeline
    vidsgen = VidsGenSimple()
    success = vidsgen.run_pipeline(selected_prompt, num_shots=num_shots)
    
    # Final message after everything completes
    if success:
        print("\n" + "="*70)
        print("🎉 CONGRATULATIONS! YOUR VIDEO IS READY!")
        print("="*70)
        print("\nThe video has been saved and previewed automatically.")
        print("You can find it at the path shown above.")
        print("\nThank you for using VidsGenSimple!")
    else:
        print("\n" + "="*70)
        print("⚠️ There were some issues during video creation.")
        print("Please check the error messages above.")
        print("="*70)

if __name__ == "__main__":
    main()