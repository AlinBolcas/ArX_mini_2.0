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

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.text_gen.textGen_revamp import TextGen
from modules.utils.utils import quick_look

# Global frame reference for cleanup and control flags
frame_instance = None
should_exit = False

input_prompt = "Write the shortest one liner poem of 3-10 words maximum."
system_prompt = "Speak about anything that Alin would find interesting and be as diverse as possible. Provide inspiration the way Alin would but to Alin himself. Share secret wisdom of the world and universe Alin may never otherwise arrive at."

# input_prompt = "Generate inspiring key words for navigating a phone call with a girl. Simply list the words rather than writing full sentences. No bullet points! Just write 3-5 words to inspire Alin. WRITE THEM ALL IN THE SAME LINE, NOT A LIST!"
# system_prompt = "Provide calm, thoughtful, and supportive guidance tailored to phone conversation dynamics."

# Ensure output directory exists
output_dir = project_root / 'output'
output_dir.mkdir(exist_ok=True)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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
            await asyncio.sleep(2)  # Reduced sleep time for faster shutdown
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

async def update_default_display(frame, text_gen):
    """Update the default display using frame_sdk"""
    try:
        # Clear display
        await frame.display.clear()
        
        # Get battery level
        try:
            battery_level = await frame.get_battery_level()
            battery_level = int(float(battery_level))
        except (ValueError, TypeError):
            battery_level = 0

        # Format current time and date
        current_time = datetime.datetime.now().strftime("%H:%M")  # 24-hour format
        current_date = datetime.datetime.now().strftime("%A, %d %b")  # Weekday, day, abbreviated month

        # Display battery level (top right)
        battery_text = f"{battery_level}%"
        await frame.display.write_text(
            battery_text, 
            align=Alignment.TOP_RIGHT,
            color=PaletteColors.GREEN
        )

        # Display date (bottom center)
        await frame.display.write_text(
            current_date,
            align=Alignment.TOP_CENTER,
            color=PaletteColors.RED
        )

        # Display time (top left)
        await frame.display.write_text(
            current_time,
            align=Alignment.TOP_LEFT,
            color=PaletteColors.CLOUDBLUE
        )

        # Get and display quote
        try:
            quote = text_gen.textGen_Alin(
                input_prompt,
                system_prompt
            )
        except Exception as e:
            print(f"TextGen error: {e}")
            quote = "Testing you."

        # scroll_text
        await frame.display.write_text(
            quote,
            x=50,
            y=50,
            max_height=340,
            max_width=600,
            # align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.ORANGE
        )
        # Show everything
        await frame.display.show()

        print(f"\n=== Status Update ===")
        print(f"Battery: {battery_level}%")
        print(f"Time: {current_time}")
        print(f"Date: {current_date}")
        print(f"Quote: {quote}")
        print("==================\n")

    except Exception as e:
        print(f"Error updating display: {e}")

async def take_photo_with_autofocus(frame, filename):
    """Take a photo with autofocus and save it"""
    try:
        print(f"\nTaking photo: {filename}")
        await frame.display.show_text("Taking photo...", align=Alignment.MIDDLE_CENTER)

        await frame.camera.save_photo(
            str(output_dir / filename),
            autofocus_seconds=2,
            quality=Quality.HIGH,
            autofocus_type=AutofocusType.CENTER_WEIGHTED
        )

        print(f"Photo saved in output directory")
        await frame.display.show_text("Photo taken!", align=Alignment.MIDDLE_CENTER)
        await asyncio.sleep(1)

        # Run quick_look in a separate thread
        loop = asyncio.get_running_loop()
        loop.run_in_executor(executor, quick_look, str(output_dir / filename))

        await update_default_display(frame, text_gen=None)

        return filename
    except Exception as e:
        print(f"Error taking photo: {e}")
        return None

async def detect_taps(frame, text_gen):
    """Detect single and double taps with 1 second window"""
    double_tap_window = 1.0  # Set to 1 second window
    
    while not should_exit:
        try:
            # Wait for first tap
            await frame.motion.wait_for_tap()
            if should_exit:
                break
            print("First tap detected! Waiting for potential second tap...")
            
            try:
                # Wait for second tap within 1 second window
                await asyncio.wait_for(
                    frame.motion.wait_for_tap(), 
                    timeout=double_tap_window
                )
                if should_exit:
                    break
                # If we get here, it's a double tap - take photo
                print("DOUBLE TAP DETECTED!")
                photo_count = len(list(output_dir.glob("frame_photo_*.jpg"))) + 1
                filename = f"frame_photo_{photo_count:03d}.jpg"
                await take_photo_with_autofocus(frame, filename)
            except asyncio.TimeoutError:
                # If no second tap within window, update display
                print("SINGLE TAP DETECTED!")
                await update_default_display(frame, text_gen)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in tap detection: {e}")
            if not should_exit:
                await asyncio.sleep(0.1)

async def main():
    global frame_instance
    print(">>> Initializing Frame connection...")

    # Initialize TextGen
    text_gen = TextGen(
        model_provider="Ollama",
        model="dolphin-llama3:latest",
        max_tokens=50,
        temperature=1.2
    )
    
    try:
        test_quote = text_gen.textGen_Alin(
            input_prompt,
            system_prompt
        )
        print(f"Test quote: {test_quote}\n\n")
    except Exception as e:
        print(f"TextGen error: {e}")

    try:
        async with Frame() as frame:
            frame_instance = frame
            print("Connected to Frame!")

            # Show initial display
            await update_default_display(frame, text_gen)
            print("Default display active - Press Ctrl+C to exit...")
            
            # Create tap detection task
            tap_task = asyncio.create_task(detect_taps(frame, text_gen))
            
            # Main loop with periodic updates
            while not should_exit:
                await asyncio.sleep(5)  # Update every 5 seconds
                if should_exit:  # Exit immediately if shutdown signal is received
                    break
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