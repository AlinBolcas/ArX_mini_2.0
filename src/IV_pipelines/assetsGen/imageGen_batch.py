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
    """Get concept, number of prompts, and desired models from user."""
    print("\n===== BATCH IMAGE GENERATION =====\n")
    
    concept = input("Enter your idea/concept: ")
    while not concept.strip():
        print("Please enter a valid concept.")
        concept = input("Enter your idea/concept: ")
    
    # Ask if user wants to generate prompts or use idea directly
    use_prompts = input("\nDo you want to generate multiple prompts from your idea? (y/n) [y]: ").lower().strip() != 'n'
    
    num_prompts = 1  # Default to 1 if not using prompt generation
    if use_prompts:
        try:
            num_prompts = int(input(f"How many prompts do you want to generate (1-12)? [5]: ") or "5")
            num_prompts = max(1, min(12, num_prompts))  # Limit between 1 and 12
        except ValueError:
            print("Using default value of 5 prompts.")
            num_prompts = 5
    
    # Model selection    
    available_models = [
        "flux-schnell", "flux-pro", "flux-pro-ultra", "flux-dev", 
        "recraft", "imagen-3", "imagen-3-fast"
    ]
    print("\nAvailable image generation models:")
    for i, model_name in enumerate(available_models, 1):
        print(f"{i}. {model_name}")
    print(f"{len(available_models) + 1}. all (use all models)")
    
    # Get multiple model selections
    selected_models = ["flux-dev"]  # Default model
    try:
        model_choices = input(f"\nChoose model numbers (comma-separated, e.g., '1,3,4' or '{len(available_models) + 1}' for all) [Default: 4 (flux-dev)]: ")
        if model_choices:
            # Check if user wants all models
            if str(len(available_models) + 1) in model_choices.replace(" ", "").split(","):
                selected_models = available_models
                print("Selected all available models")
            else:
                # Parse the comma-separated choices
                model_indices = [int(idx.strip()) - 1 for idx in model_choices.split(',')]
                # Filter valid indices and get unique models
                selected_models = []
                for idx in model_indices:
                    if 0 <= idx < len(available_models):
                        selected_models.append(available_models[idx])
                    else:
                        print(f"Ignoring invalid choice: {idx + 1}")
                
                # Remove duplicates while preserving order
                selected_models = list(dict.fromkeys(selected_models))
                
                if not selected_models:  # If no valid selections
                    selected_models = ["flux-dev"]
                    print(f"No valid choices. Using default model: {selected_models[0]}")
        else:
            print(f"No choice made. Using default model: {selected_models[0]}")
    except ValueError:
        print(f"Invalid input. Using default model: {selected_models[0]}")
    
    print(f"Selected models: {', '.join(selected_models)}")
    total_images = num_prompts * len(selected_models)
    print(f"Will generate {num_prompts} prompt{'s' if num_prompts > 1 else ''} × {len(selected_models)} model{'s' if len(selected_models) > 1 else ''} = {total_images} total images")
    
    return concept, num_prompts, selected_models, use_prompts


def generate_prompts(concept: str, num_prompts: int) -> Optional[List[str]]:
    """Generate detailed prompts using OpenAI API."""
    print(f"\nGenerating {num_prompts} detailed prompts for: {concept}")
    
    try:
        # Initialize OpenAI API
        openai_api = OpenAIResponsesAPI()
        
        # Define a schema for the structured output
        prompt_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "prompts": {
                    "type": "array",
                    "description": f"Array of {num_prompts} unique, detailed image prompts for '{concept}'",
                    "items": {
                        "type": "string",
                        "description": "Detailed image generation prompt that would work well with Midjourney or DALL-E"
                    }
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
            if len(prompts) < num_prompts:
                print(f"Warning: Only generated {len(prompts)} prompts instead of requested {num_prompts}")
            
            # Display the generated prompts
            print("\nGenerated prompts:")
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{i}. {prompt}")
            
            # Ask user if they want to proceed with these prompts
            proceed = input("\nDo you want to proceed with these prompts? (y/n) [y]: ").lower().strip() != 'n'
            if not proceed:
                return None
            
            return prompts
        else:
            print("❌ Failed to generate prompts.")
            return None
            
    except Exception as e:
        print(f"❌ Error generating prompts: {str(e)}")
        return None


def generate_images(prompts: List[str], output_dir: Path, selected_models: List[str]):
    """Generate images from prompts using Replicate API with the selected models."""
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
    
    # Function to process a single prompt with a specific model
    def generate_image_worker(prompt: str, prompt_index: int, model: str, total_index: int):
        print(f"\n🖼️ Generating image {total_index}/{len(prompts) * len(selected_models)}")
        print(f"Model: {model}")
        print(f"Prompt {prompt_index}: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")
        
        try:
            # Generate image
            image_url = api.generate_image(
                prompt=prompt,
                aspect_ratio="16:9",  # Widescreen aspect ratio better for most concepts
                safety_tolerance=6,    # Maximum allowed is 6
                model=model
            )
            
            if image_url:
                print(f"✅ Generation complete for image {total_index} ({model})")
                
                # Create unique filename with timestamp to avoid overwrites
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:19]
                filename = f"concept_{prompt_index}_{model}_{timestamp}.jpg"
                
                # Download image
                image_path = api.download_file(
                    image_url, 
                    output_dir=str(output_dir),
                    filename=filename
                )
                
                return {
                    "prompt": prompt,
                    "prompt_index": prompt_index,
                    "model": model,
                    "total_index": total_index,
                    "url": image_url,
                    "path": image_path,
                    "success": image_path is not None
                }
            else:
                print(f"❌ Generation failed for image {total_index} ({model})")
        except Exception as e:
            print(f"❌ Error generating image {total_index} ({model}): {str(e)}")
        
        # Return failure result
        return {
            "prompt": prompt,
            "prompt_index": prompt_index,
            "model": model,
            "total_index": total_index,
            "url": None,
            "path": None,
            "success": False
        }
    
    # Process prompts and models in parallel
    print(f"\nProcessing {len(prompts)} prompts × {len(selected_models)} models = {len(prompts) * len(selected_models)} total images")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Create tasks for each prompt-model combination
        futures = []
        total_index = 1
        
        for prompt_index, prompt in enumerate(prompts, 1):
            for model in selected_models:
                futures.append(
                    executor.submit(
                        generate_image_worker,
                        prompt=prompt,
                        prompt_index=prompt_index,
                        model=model,
                        total_index=total_index
                    )
                )
                total_index += 1
        
        # Monitor progress
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing result: {str(e)}")
    
    # Display results summary
    print("\n===== GENERATION RESULTS =====")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"Total attempts: {len(prompts) * len(selected_models)}")
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
    
    # Sort by prompt index first, then by model name
    successful.sort(key=lambda x: (x["prompt_index"], x["model"]))
    
    # Initialize Replicate API for the display_media method
    api = ReplicateAPI()
    
    # Display each image
    for result in successful:
        image_path = result["path"]
        print(f"Displaying image {result['total_index']} (Prompt {result['prompt_index']}, Model: {result['model']})")
        print(f"File: {os.path.basename(image_path)}")
        api.display_media(image_path, "image")
        time.sleep(0.1)  # Small delay between opening images
    
    print("\nImages are now open. Close them manually when done viewing.")
    print(f"\nAll images saved to: {os.path.dirname(successful[0]['path'])}")


def main():
    """Main function to run the batch image generation process."""
    # Create output directories
    output_dir = create_output_directories()
    
    # Get user input
    concept, num_prompts, selected_models, use_prompts = get_user_input()
    
    # Generate or prepare prompts
    if use_prompts:
        prompts = generate_prompts(concept, num_prompts)
        if prompts is None:
            print("\nPrompt generation failed or was rejected. Would you like to:")
            print("1. Try generating prompts again")
            print("2. Use your original idea as a single prompt")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ").strip()
            if choice == "1":
                prompts = generate_prompts(concept, num_prompts)
                if prompts is None:
                    print("Prompt generation failed again. Exiting.")
                    return
            elif choice == "2":
                prompts = [concept]
            else:
                print("Exiting.")
                return
    else:
        # Use the original concept as the only prompt
        prompts = [concept]
    
    # Final check before proceeding
    if not prompts:
        print("No prompts available. Exiting.")
        return
    
    # Generate images with all selected models
    results = generate_images(prompts, output_dir, selected_models)
    
    # Display images after all are generated
    if results:
        display_images(results)
    
    print("\n===== BATCH IMAGE GENERATION COMPLETE =====")


if __name__ == "__main__":
    main() 