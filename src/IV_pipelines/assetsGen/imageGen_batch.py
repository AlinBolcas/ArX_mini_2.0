#!/usr/bin/env python3
"""
Batch Image Generation Tool

This script generates multiple images based on a user-provided concept, using:
1. OpenAI API to generate multiple detailed prompts from a concept
2. Replicate API to generate images from those prompts

Images are saved with unique filenames and displayed after generation.
"""

import os
import sys
import time
import json
import requests
import tempfile
import subprocess
import concurrent.futures
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from datetime import datetime

# Add the src directory to the path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

# Import the API wrappers
from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
from src.I_integrations.replicate_API import ReplicateAPI


def create_output_directories():
    """Ensure all necessary output directories exist."""
    output_dir = Path("data/output/assets/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_user_input():
    """Get concept and number of prompts from user."""
    print("\n===== BATCH IMAGE GENERATION =====\n")
    
    concept = input("Enter your idea/concept: ")
    while not concept.strip():
        print("Please enter a valid concept.")
        concept = input("Enter your idea/concept: ")
    
    try:
        num_prompts = int(input(f"How many images do you want to generate (1-12)? [5]: ") or "5")
        num_prompts = max(1, min(12, num_prompts))  # Limit between 1 and 12
    except ValueError:
        print("Using default value of 5 prompts.")
        num_prompts = 5
    
    return concept, num_prompts


def generate_prompts(concept: str, num_prompts: int) -> List[str]:
    """Generate detailed prompts using OpenAI API."""
    print(f"\nGenerating {num_prompts} detailed prompts for: {concept}")
    
    try:
        # Initialize OpenAI API
        openai_api = OpenAIResponsesAPI()
        
        # Define a schema for the structured output
        prompt_schema = {
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "description": f"Array of {num_prompts} unique, detailed image prompts for '{concept}'",
                    "items": {
                        "type": "string",
                        "description": "Detailed image generation prompt that would work well with Midjourney or DALL-E"
                    },
                    "minItems": num_prompts,
                    "maxItems": num_prompts
                }
            },
            "required": ["prompts"]
        }
        
        # System prompt to guide the generation
        system_prompt = f"""
        You are an expert at creating detailed image generation prompts. 
        Generate {num_prompts} unique prompts for '{concept}' that are:
        
        1. Highly detailed with specific visual elements, lighting, perspective, and style
        2. Varied to explore different aspects of the concept
        3. Photorealistic with terms like "high resolution", "detailed", "professional lighting"
        4. Around 50-75 words each (neither too short nor excessively long)
        5. Focused specifically on the concept provided
        
        Ensure each prompt is complete, standalone, and would produce a high-quality image.
        """
        
        # Generate the prompts
        response = openai_api.structured_response(
            user_prompt=f"Create {num_prompts} detailed image generation prompts for: {concept}",
            system_prompt=system_prompt,
            output_schema=prompt_schema,
            temperature=0.8
        )
        
        if "prompts" in response and isinstance(response["prompts"], list):
            prompts = response["prompts"]
            
            # Display the generated prompts
            print("\nGenerated prompts:")
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{i}. {prompt}")
            
            return prompts
        else:
            print("❌ Failed to generate prompts. Using a default prompt.")
            return [f"Detailed professional high-resolution image of {concept}, photorealistic, studio lighting"] * num_prompts
            
    except Exception as e:
        print(f"❌ Error generating prompts: {str(e)}")
        # Fallback to a default prompt
        return [f"Detailed professional high-resolution image of {concept}, photorealistic, studio lighting"] * num_prompts


def generate_images(prompts: List[str], output_dir: Path):
    """Generate images from prompts using Replicate API."""
    print("\n===== GENERATING IMAGES =====")
    
    # Initialize Replicate API
    try:
        api = ReplicateAPI()
        
        # Override the ReplicateAPI download_file method to use our custom output directory
        orig_download_file = api.download_file
        
        def custom_download_file(url, output_dir, filename):
            return orig_download_file(url, output_dir, filename)
            
        api.download_file = custom_download_file
        
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure to set the REPLICATE_API_TOKEN in your .env file.")
        return []
    
    results = []
    
    # Function to process a single prompt
    def generate_image_worker(prompt, index):
        print(f"\n🖼️ Generating image {index}/{len(prompts)}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        try:
            # Generate image
            image_url = api.generate_image(
                prompt=prompt,
                aspect_ratio="16:9",  # Widescreen aspect ratio better for most concepts
                safety_tolerance=6     # Maximum allowed is 6
            )
            
            if image_url:
                print(f"✅ Generation complete for image {index}")
                
                # Create unique filename with timestamp to avoid overwrites
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:19]
                filename = f"concept_{index}_{timestamp}.jpg"
                
                # Download image
                image_path = api.download_file(
                    image_url, 
                    output_dir=str(output_dir),
                    filename=filename
                )
                
                return {
                    "prompt": prompt,
                    "index": index,
                    "url": image_url,
                    "path": image_path,
                    "success": image_path is not None
                }
            else:
                print(f"❌ Generation failed for image {index}")
        except Exception as e:
            print(f"❌ Error generating image {index}: {str(e)}")
        
        # Return failure result
        return {
            "prompt": prompt,
            "index": index,
            "url": None,
            "path": None,
            "success": False
        }
    
    # Process prompts in parallel
    print(f"\nProcessing {len(prompts)} prompts in parallel (max 4 concurrent)...")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(prompts), 4)) as executor:
        # Submit tasks (max 4 concurrent to avoid rate limiting)
        future_to_prompt = {
            executor.submit(generate_image_worker, prompt, i+1): i+1
            for i, prompt in enumerate(prompts)
        }
        
        # Monitor progress
        pending = list(future_to_prompt.keys())
        completed = 0
        
        while pending:
            # Wait for the next task to complete
            done, pending = concurrent.futures.wait(
                pending,
                timeout=2.0,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for future in done:
                prompt_index = future_to_prompt[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                except Exception as e:
                    print(f"❌ Error processing result for prompt {prompt_index}: {str(e)}")
            
            # Print progress
            elapsed = time.time() - start_time
            print(f"⏳ Progress: {completed}/{len(prompts)} images complete ({elapsed:.1f}s elapsed)")
    
    # Display results summary
    print("\n===== GENERATION RESULTS =====")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Total prompts: {len(prompts)}")
    print(f"Successfully generated: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    return results


def display_images(results: List[Dict[str, Any]]):
    """Display all successfully generated images with a short delay between each."""
    successful = [r for r in results if r["success"]]
    
    if not successful:
        print("\n❌ No successful image generations to display")
        return
    
    print(f"\n===== DISPLAYING {len(successful)} IMAGES =====")
    print("Opening images with a slight delay between each...")
    
    # Sort by original index
    successful.sort(key=lambda x: x["index"])
    
    # Initialize Replicate API for the display_media method
    api = ReplicateAPI()
    
    # Display each image
    for result in successful:
        image_path = result["path"]
        print(f"Displaying image {result['index']}: {os.path.basename(image_path)}")
        api.display_media(image_path, "image")
        time.sleep(0.1)  # Small delay between opening images
    
    print("\nImages are now open. Close them manually when done viewing.")
    print(f"\nAll images saved to: {os.path.dirname(successful[0]['path'])}")


def main():
    """Main function to run the batch image generation process."""
    # Create output directories
    output_dir = create_output_directories()
    
    # Get user input
    concept, num_prompts = get_user_input()
    
    # Generate prompts
    prompts = generate_prompts(concept, num_prompts)
    
    # Generate images
    results = generate_images(prompts, output_dir)
    
    # Display images after all are generated
    if results:
        display_images(results)
    
    print("\n===== BATCH IMAGE GENERATION COMPLETE =====")


if __name__ == "__main__":
    main() 