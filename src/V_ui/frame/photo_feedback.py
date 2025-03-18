import asyncio
from pathlib import Path
import sys
import os
import signal
import concurrent.futures
from frame_sdk import Frame
from frame_sdk.display import Alignment, PaletteColors
from frame_sdk.camera import Quality, AutofocusType
import datetime
import tempfile

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.text_gen.textGen_revamp import TextGen
from modules.utils.utils import quick_look
from modules.text_gen.textGen_imageProcess import critique_image
from modules.tts_gen.ttsGen import TtsGen

# Global frame reference and control flags
frame_instance = None
should_exit = False

# Add to global variables at the top
last_update_time = 0
UPDATE_INTERVAL = 8  # seconds
is_displaying_critique = False  # Flag to track if we're showing a critique

# Initialize text generation and TTS
# text_gen = TextGen(
#     model_provider="Ollama",
#     model="dolphin-llama3:latest", 
#     max_tokens=65,
#     temperature=0.9
# )
# llama3.2-vision:latest 
# dolphin-llama3:latest
# qwq:latest
# llama2-uncensored:latest

text_gen = TextGen(
    model_provider="OpenAI",
    model="gpt-4o-mini", 
    max_tokens=65,
    temperature=0.9
)
# chatgpt-4o-latest
# o1-preview
# o1-mini
# gpt-4o-realtime-preview
# gpt-4o-mini

tts = TtsGen()

# You are speaking with Alin's girlfriend called Emujin. She is a beautiful Mongolian living in London working in Finance and is her birthday today. Share relevant bits with her to showcase your capabilities.

# Add at the top with other globals
default_prompt = """
Share secrets, wisdom, or philosophies of the world and universe Alin may never otherwise arrive at.
Write it as the shortest most condensed one liner quote of 5-15 words maximum!!!
Do not repeat yourself, but share something new every time.
Reply straight away without any other formatting text or comments.
Share only one idea at a time.
Reply:
"""

# Remove the original input_prompt definition and make it global
global_prompt = default_prompt

# Ensure output directory exists
output_dir = project_root / 'output'
output_dir.mkdir(exist_ok=True)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

def process_tts_async(text, voice_id=1):
    """Process TTS in a separate thread"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir=str(output_dir / 'tmp')) as tmp_file:
            save_path = tts.ttsGenAsync(text, voice_id=voice_id, save_path=tmp_file.name)
            tts.play_audio(save_path)
    except Exception as e:
        print(f"Error processing TTS: {e}")

async def cleanup_display():
    """Cleanup function to clear display and show goodbye message"""
    if frame_instance and frame_instance.bluetooth.is_connected():
        try:
            await frame_instance.display.clear()
            await frame_instance.display.write_text(
                "Goodbye!",
                align=Alignment.MIDDLE_CENTER,
                color=PaletteColors.GREEN
            )
            await frame_instance.display.show()
            await asyncio.sleep(1)
            await frame_instance.display.clear()
            await frame_instance.display.show()
            await frame_instance.sleep()
        except Exception as e:
            print(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle system signals for graceful shutdown"""
    global should_exit
    print("\nSignal received, initiating shutdown...")
    should_exit = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def update_default_display(frame, text_gen, force_update=False):
    """Update the default display using frame_sdk"""
    global last_update_time, is_displaying_critique, global_prompt
    
    # Skip update if showing critique
    if is_displaying_critique and not force_update:
        return
        
    current_time = asyncio.get_event_loop().time()
    
    # Only update if forced or enough time has passed
    if not force_update and last_update_time > 0:
        time_since_last = current_time - last_update_time
        if time_since_last < UPDATE_INTERVAL:
            return
        
    try:
        await frame.display.clear()
        
        # Get battery level
        try:
            battery_level = await frame.get_battery_level()
            battery_level = int(float(battery_level))
        except (ValueError, TypeError):
            battery_level = 0

        # Format display time and date (this is different from the update timing)
        display_time = datetime.datetime.now().strftime("%H:%M")
        display_date = datetime.datetime.now().strftime("%A, %d %b")

        # Display battery level (top right)
        battery_text = f"{battery_level}%"
        await frame.display.write_text(
            battery_text, 
            align=Alignment.TOP_RIGHT,
            color=PaletteColors.GREEN
        )

        # Display date and time
        await frame.display.write_text(
            display_date,
            align=Alignment.TOP_CENTER,
            color=PaletteColors.RED
        )
        await frame.display.write_text(
            display_time,
            align=Alignment.TOP_LEFT,
            color=PaletteColors.CLOUDBLUE
        )

        # Generate and display quote
        try:
            display_text = text_gen.textGen_Alin(global_prompt)
            color = PaletteColors.ORANGE
            
            # Append the display_text to wisdom.md in the output folder
            wisdom_file = output_dir / 'wisdom.md'
            with open(wisdom_file, 'a') as f:
                f.write(f"- {display_text}\n")
            
            # Process TTS in background
            # executor.submit(process_tts_async, display_text)
            
        except Exception as e:
            print(f"TextGen error: {e}")
            display_text = "Testing you."
            color = PaletteColors.RED

        # Display text
        await frame.display.write_text(
            display_text,
            x=25,
            y=75,
            max_height=340,
            max_width=600,
            color=color
        )
        await frame.display.show()

        print(f"\n=== Status Update ===")
        print(f"Battery: {battery_level}%")
        print(f"Time: {display_time}")
        print(f"Date: {display_date}")
        print(f"Display Text: {display_text}")
        print("==================\n")

        # Update the last update timestamp AFTER display is complete
        last_update_time = asyncio.get_event_loop().time()  # Get fresh timestamp after display
        
    except Exception as e:
        print(f"Error updating display: {e}")

async def process_photo(frame, filename):
    """Process photo with critique and display result"""
    global is_displaying_critique
    try:
        full_path = str(output_dir / filename)
        
        # Set critique display flag
        is_displaying_critique = True
        
        # Run quick_look in a separate thread
        loop = asyncio.get_running_loop()
        loop.run_in_executor(executor, quick_look, full_path)
        
        # Show processing message
        await frame.display.clear()
        await frame.display.write_text(
            "Processing image...",
            align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.CLOUDBLUE
        )
        await frame.display.show()
        
        # Add delay to ensure file is saved
        await asyncio.sleep(1)
        
        if not Path(full_path).exists():
            raise ValueError("Image file not found")
            
        result = critique_image(full_path, max_tokens=100, temperature=0.8)
        if not result:
            raise ValueError("No critique generated")
        
        # Display result
        await frame.display.clear()
        print("\nScrolling critique...")
        await frame.display.scroll_text(
            result,
            lines_per_frame=10,
            delay=0.3,
            color=PaletteColors.GREEN
        )
        print("Scrolling finished")
        
        # Process TTS in background
        # executor.submit(process_tts_async, result, voice_id=3)
        
        # Reset critique flag before returning to default display
        is_displaying_critique = False
        
        # Wait before returning to default display
        await asyncio.sleep(1)
        await update_default_display(frame, text_gen, force_update=True)
        return result
            
    except Exception as e:
        print(f"Error processing photo: {e}")
        await frame.display.clear()
        await frame.display.write_text(
            "Failed to process image",
            align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.RED
        )
        await frame.display.show()
        await asyncio.sleep(1)
        
        # Reset critique flag and return to default display
        is_displaying_critique = False
        await update_default_display(frame, text_gen, force_update=True)
        return None

async def take_photo_with_autofocus(frame, filename):
    """Take a photo with autofocus and process it"""
    try:
        print(f"\nTaking photo: {filename}")
        await frame.display.show_text("Taking photo...", color=PaletteColors.CLOUDBLUE, align=Alignment.MIDDLE_CENTER)

        # Take photo with timeout
        try:
            async with asyncio.timeout(5.0):
                await frame.camera.save_photo(
                    str(output_dir / filename),
                    autofocus_seconds=3,
                    quality=Quality.HIGH,
                    autofocus_type=AutofocusType.CENTER_WEIGHTED
                )
        except asyncio.TimeoutError:
            print("Photo capture timed out, retrying...")
            async with asyncio.timeout(3.0):
                await frame.camera.save_photo(
                    str(output_dir / filename),
                    autofocus_seconds=2,
                    quality=Quality.MEDIUM,
                    autofocus_type=AutofocusType.FIXED
                )

        print(f"Photo saved in output directory")
        await frame.display.show_text("Photo taken!", color=PaletteColors.CLOUDBLUE, align=Alignment.MIDDLE_CENTER)
        await asyncio.sleep(1)
        
        # Process the photo
        await process_photo(frame, filename)
        return filename
        
    except Exception as e:
        print(f"Error taking photo: {e}")
        await frame.display.show_text("Failed to take photo", color=PaletteColors.RED, align=Alignment.MIDDLE_CENTER)
        await asyncio.sleep(2)
        await update_default_display(frame, text_gen)
        return None

async def detect_taps(frame, text_gen):
    """Detect taps for display refresh and photo capture"""
    global last_update_time, is_displaying_critique
    double_tap_window = 1.0
    
    while not should_exit:
        try:
            # Wait for first tap
            await frame.motion.wait_for_tap()
            if should_exit:
                break
            print("First tap detected! Waiting for potential second tap...")
            
            try:
                # Wait for second tap
                await asyncio.wait_for(
                    frame.motion.wait_for_tap(), 
                    timeout=double_tap_window
                )
                if should_exit:
                    break
                    
                if is_displaying_critique:
                    # If showing critique, return to default display
                    is_displaying_critique = False
                    await update_default_display(frame, text_gen, force_update=True)
                else:
                    # Take new photo
                    print("DOUBLE TAP DETECTED!")
                    photo_count = len(list(output_dir.glob("frame_photo_*.jpg"))) + 1
                    filename = f"frame_photo_{photo_count:03d}.jpg"
                    await take_photo_with_autofocus(frame, filename)
                    
            except asyncio.TimeoutError:
                # Single tap
                if is_displaying_critique:
                    # Return to default display if showing critique
                    is_displaying_critique = False
                    await update_default_display(frame, text_gen, force_update=True)
                else:
                    # Normal display refresh
                    print("SINGLE TAP DETECTED!")
                    last_update_time = 0  # Force update by resetting timer
                    await update_default_display(frame, text_gen, force_update=True)
            
        except Exception as e:
            print(f"Error in tap detection: {e}")
            if not should_exit:
                await asyncio.sleep(0.1)

async def main():
    global frame_instance, last_update_time, global_prompt
    print(">>> Initializing Frame connection...")

    try:
        async with Frame() as frame:
            frame_instance = frame
            print("Connected to Frame!")
            
            # Get user input for prompt
            print("\nEnter your prompt (or press Enter to use default):")
            print("Note: Response will be formatted as a short quote (5-15 words)")
            user_input = input("> ").strip()
            
            if user_input:
                # Create custom prompt with user input
                global_prompt = f"""
{user_input}
Write it as the shortest most condensed one liner quote of 5-15 words maximum!!!
Do not repeat yourself, but share something new every time.
Reply straight away without any other formatting text or comments.
Share only one idea at a time.
Reply:
"""
                print(f"\nUsing custom prompt: {user_input}")
            else:
                print("\nUsing default prompt")

            # Show initial display
            await update_default_display(frame, text_gen, force_update=True)
            print("Default display active - Press Ctrl+C to exit...")
            
            # Create tap detection task
            tap_task = asyncio.create_task(detect_taps(frame, text_gen))
            
            # Main loop with periodic updates
            while not should_exit:
                await asyncio.sleep(1)  # Check more frequently but only update when needed
                if should_exit:
                    break
                if not is_displaying_critique:  # Only update if not showing critique
                    await update_default_display(frame, text_gen)
            
            # Clean up
            print("Starting cleanup...")
            tap_task.cancel()
            try:
                await tap_task
            except asyncio.CancelledError:
                pass
            
            await cleanup_display()

    except Exception as e:
        print(f"Error setting up Frame: {e}")
    finally:
        if frame_instance:
            await cleanup_display()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        should_exit = True
    except Exception as e:
        print(f"Fatal error: {e}")