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
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from modules.stt_gen.sttGen_simplified import STTSimplified
from modules.tts_gen.ttsGen import TtsGen
from modules.text_gen.textGen_revamp import TextGen
from modules.utils.utils import quick_look
from modules.text_gen.textGen_imageProcess import describe_image, critique_image

# Global frame reference for cleanup and control flags
frame_instance = None
should_exit = False

input_prompt = "Share a philosophical and rare piece of information someone would not normally come about. Keep it condensed within 5-10 words."
system_prompt = "Speak about anything that Alin would find interesting and be as diverse as possible. Provide inspiration the way Alin would but to Alin himself. Share secret wisdom of the world and universe Alin may never otherwise arrive at."

# input_prompt = "Generate inspiring key words for navigating a phone call with a girl. Simply list the words rather than writing full sentences. No bullet points! Just write 3-5 words to inspire Alin. WRITE THEM ALL IN THE SAME LINE, NOT A LIST!"
# system_prompt = "Provide calm, thoughtful, and supportive guidance tailored to phone conversation dynamics."

# Ensure output directory exists
output_dir = project_root / 'output'
output_dir.mkdir(exist_ok=True)

executor = ThreadPoolExecutor(max_workers=3)  # Increase workers for parallel processing

# Add new global instances
stt = STTSimplified()
tts = TtsGen()
# text_gen = TextGen(
#     model_provider="OpenAI", 
#     model="gpt-4o-mini", 
#     max_tokens=50, 
#     temperature=1.2)

text_gen = TextGen(
    model_provider="Ollama",
    model="dolphin-llama3:latest",
    max_tokens=50,
    temperature=1.2
)

# Add new function for TTS processing
def process_tts_async(text, voice_id=1):
    """Process TTS in a separate thread with proper cleanup"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False, dir=str(output_dir / 'tmp')) as tmp_file:
            save_path = tts.ttsGenAsync(text, voice_id=voice_id, save_path=tmp_file.name)
            
            def play_and_cleanup():
                try:
                    tts.play_audio(save_path)
                finally:
                    # Clean up the temp file after playing
                    try:
                        os.unlink(save_path)
                    except:
                        pass
            
            # Create a new thread for playback
            playback_thread = threading.Thread(target=play_and_cleanup)
            playback_thread.daemon = True  # Make thread daemon so it doesn't block program exit
            playback_thread.start()
            
    except Exception as e:
        print(f"Error processing TTS: {e}")

async def cleanup_display():
    """Cleanup function to clear display and show goodbye message"""
    if frame_instance and frame_instance.bluetooth.is_connected():
        try:
            # Clean up audio files
            executor.submit(cleanup_audio_files)
            
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

# Add at the top with other globals
display_lock = asyncio.Lock()  # Add this for display synchronization

async def update_default_display(frame, text_gen):
    """Update the default display using frame_sdk"""
    async with display_lock:  # Use lock for display updates
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
            current_time = datetime.datetime.now().strftime("%H:%M")
            current_date = datetime.datetime.now().strftime("%A, %d %b")

            # Display battery level (top right)
            battery_text = f"{battery_level}%"
            await frame.display.write_text(
                battery_text, 
                align=Alignment.TOP_RIGHT,
                color=PaletteColors.GREEN
            )

            # Display date and time
            await frame.display.write_text(
                current_date,
                align=Alignment.TOP_CENTER,
                color=PaletteColors.RED
            )
            await frame.display.write_text(
                current_time,
                align=Alignment.TOP_LEFT,
                color=PaletteColors.CLOUDBLUE
            )

            # Generate and display quote
            try:
                display_text = text_gen.textGen_Alin(input_prompt, system_prompt)
                color = PaletteColors.ORANGE
                
                # Process TTS in background
                executor.submit(process_tts_async, display_text)
                
            except Exception as e:
                print(f"TextGen error: {e}")
                display_text = "Testing you."
                color = PaletteColors.RED

            # Display text
            await frame.display.write_text(
                display_text,
                x=50,
                y=50,
                max_height=340,
                max_width=600,
                color=color
            )
            await frame.display.show()

            print(f"\n=== Status Update ===")
            print(f"Battery: {battery_level}%")
            print(f"Time: {current_time}")
            print(f"Date: {current_date}")
            print(f"Display Text: {display_text}")
            print("==================\n")

        except Exception as e:
            print(f"Error updating display: {e}")

async def process_photo(frame, filename):
    """Process photo with critique and display result"""
    try:
        full_path = str(output_dir / filename)
        
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
        
        try:
            # Add delay to ensure file is saved
            await asyncio.sleep(1)
            
            if not Path(full_path).exists():
                raise ValueError("Image file not found")
                
            result = critique_image(full_path)
            if not result:
                raise ValueError("No critique generated")
            
            # Display result
            await frame.display.clear()
            print("\nScrolling critique...")
            await frame.display.scroll_text(
                result,
                lines_per_frame=10,
                delay=0.08,
                color=PaletteColors.SEABLUE
            )
            print("Scrolling finished")
            
            # Process TTS in background
            executor.submit(process_tts_async, result, voice_id=3)
            
            # Wait a moment before returning
            await asyncio.sleep(2)
            return result
            
        except Exception as e:
            print(f"Error processing image critique: {str(e)}")
            await frame.display.clear()
            await frame.display.write_text(
                "Failed to process image",
                align=Alignment.MIDDLE_CENTER,
                color=PaletteColors.RED
            )
            await frame.display.show()
            await asyncio.sleep(2)
            return None
            
    except Exception as e:
        print(f"General error in process_photo: {str(e)}")
        return None

async def take_photo_with_autofocus(frame, filename):
    """Take a photo with autofocus and process it"""
    try:
        print(f"\nTaking photo: {filename}")
        await frame.display.show_text("Taking photo...", color=PaletteColors.CLOUDBLUE, align=Alignment.MIDDLE_CENTER)

        # Add shorter timeout and error handling for photo capture
        try:
            async with asyncio.timeout(5.0):  # 5 second timeout for photo capture
                await frame.camera.save_photo(
                    str(output_dir / filename),
                    autofocus_seconds=1,  # Reduced from 2
                    quality=Quality.HIGH,
                    autofocus_type=AutofocusType.CENTER_WEIGHTED
                )
        except asyncio.TimeoutError:
            print("Photo capture timed out, retrying...")
            # Second attempt with basic settings
            async with asyncio.timeout(3.0):
                await frame.camera.save_photo(
                    str(output_dir / filename),
                    autofocus_seconds=0.5,
                    quality=Quality.MEDIUM,
                    autofocus_type=AutofocusType.FIXED
                )

        print(f"Photo saved in output directory")
        await frame.display.show_text("Photo taken!", color=PaletteColors.CLOUDBLUE, align=Alignment.MIDDLE_CENTER)
        await asyncio.sleep(1)
        
        # Process the photo and get critique
        result = await process_photo(frame, filename)
        
        if result:
            # Wait a bit after scrolling before updating default display
            await asyncio.sleep(2)
            await update_default_display(frame, text_gen=None)
        
        return filename
    except Exception as e:
        print(f"Error taking photo: {e}")
        await frame.display.show_text("Failed to take photo", color=PaletteColors.RED, align=Alignment.MIDDLE_CENTER)
        await asyncio.sleep(2)
        await update_default_display(frame, text_gen=None)
        return None

# Add new constant at the top
RECORDING_CLEANUP_DELAY = 0.5  # Delay for cleanup after recording

# Add audio cleanup function
def cleanup_audio_files():
    """Clean up temporary audio files"""
    try:
        tmp_dir = output_dir / 'tmp'
        for file in tmp_dir.glob('*.wav'):
            try:
                file.unlink()
            except:
                pass
        for file in tmp_dir.glob('*.mp3'):
            try:
                file.unlink()
            except:
                pass
    except Exception as e:
        print(f"Error cleaning up audio files: {e}")

async def process_voice_input(frame):
    """Record and process voice input using Frame's microphone"""
    async with display_lock:
        try:
            global should_update_display
            should_update_display = False  # Pause updates while recording
            
            audio_path = output_dir / 'tmp' / 'frame_audio.wav'
            if audio_path.exists():
                audio_path.unlink()
            
            # Show recording message
            await frame.display.clear()
            await frame.display.write_text(
                "Listening . . . tap to finish",
                align=Alignment.MIDDLE_CENTER,
                color=PaletteColors.CLOUDBLUE
            )
            await frame.display.show()
            
            try:
                # Start recording with timeout
                async with asyncio.timeout(10.0):  # 10 second max recording time
                    recording_task = asyncio.create_task(
                        frame.microphone.save_audio_file(
                            str(audio_path),
                            max_length_in_seconds=10
                        )
                    )
                    
                    # Wait for tap or recording completion
                    while True:
                        try:
                            await asyncio.wait_for(frame.motion.wait_for_tap(), timeout=0.1)
                            if not recording_task.done():
                                recording_task.cancel()
                                print("\nRecording stopped by tap")
                            break
                        except asyncio.TimeoutError:
                            if recording_task.done():
                                break
                        except Exception as e:
                            print(f"Error during tap detection: {e}")
                            break
                    
                    try:
                        await recording_task
                    except asyncio.CancelledError:
                        pass
                    
                    if not audio_path.exists() or audio_path.stat().st_size == 0:
                        raise ValueError("No audio recorded")
                    
                    # Process the recording
                    await frame.display.clear()
                    await frame.display.write_text(
                        "Processing message...",
                        align=Alignment.MIDDLE_CENTER,
                        color=PaletteColors.CLOUDBLUE
                    )
                    await frame.display.show()
                    
                    transcription = stt.transcribe_whisper(str(audio_path))
                    if not transcription:
                        raise ValueError("Failed to transcribe audio")
                    
                    print(f"\nTranscribed: {transcription}")
                    response = text_gen.textGen_Alin(transcription)
                    
                    # Display response
                    await frame.display.clear()
                    await frame.display.write_text(
                        response,
                        x=50,
                        y=50,
                        max_height=340,
                        max_width=600,
                        color=PaletteColors.ORANGE
                    )
                    await frame.display.show()
                    
                    # Process TTS in background
                    executor.submit(process_tts_async, response, voice_id=2)
                    await asyncio.sleep(2)
                    
                    return response
                    
            except asyncio.TimeoutError:
                print("Recording timed out")
                raise ValueError("Recording timed out")
                
        except Exception as e:
            print(f"Error in voice processing: {str(e)}")
            await frame.display.clear()
            await frame.display.write_text(
                "Error processing voice input",
                align=Alignment.MIDDLE_CENTER,
                color=PaletteColors.RED
            )
            await frame.display.show()
            await asyncio.sleep(2)
        finally:
            should_update_display = True  # Resume updates
            await update_default_display(frame, text_gen=None)
        return None

# Modify the reset_frame_memory function
async def reset_frame_memory(frame):
    """Reset Frame's Lua VM when memory issues occur"""
    try:
        print("Resetting Frame memory...")
        # First try to clear the display
        await frame.display.clear()
        await frame.display.show()
        
        # Then try to reset the Frame
        if hasattr(frame.bluetooth, 'send_reset_signal'):
            await frame.bluetooth.send_reset_signal()
        else:
            # Alternative reset method if send_reset_signal isn't available
            await frame.sleep()
            await asyncio.sleep(1)
            await frame.wake()
            
        await asyncio.sleep(0.5)  # Wait for reset to complete
        return True
    except Exception as e:
        print(f"Error resetting Frame memory: {e}")
        return False

# Modify detect_taps to use a simpler memory error detection
async def detect_taps(frame, text_gen):
    """Detect single tap for voice input and double tap for photo"""
    double_tap_window = 1.0
    memory_error_count = 0
    MAX_MEMORY_ERRORS = 3
    
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
                print("DOUBLE TAP DETECTED!")
                photo_count = len(list(output_dir.glob("frame_photo_*.jpg"))) + 1
                filename = f"frame_photo_{photo_count:03d}.jpg"
                await take_photo_with_autofocus(frame, filename)
                memory_error_count = 0  # Reset error count on successful operation
            except asyncio.TimeoutError:
                print("SINGLE TAP DETECTED - Starting voice recording")
                await process_voice_input(frame)
                memory_error_count = 0  # Reset error count on successful operation
            
        except Exception as e:
            error_msg = str(e).lower()  # Convert to lowercase for easier matching
            if any(err in error_msg for err in ["memory", "lua", "stack"]):
                memory_error_count += 1
                print(f"Memory error detected (count: {memory_error_count})")
                
                if memory_error_count >= MAX_MEMORY_ERRORS:
                    print("Multiple memory errors detected, attempting reset...")
                    if await reset_frame_memory(frame):
                        memory_error_count = 0
                        # Reinitialize display after reset
                        await update_default_display(frame, text_gen)
                    else:
                        print("Failed to reset Frame memory")
                        
                await asyncio.sleep(0.5)  # Add delay between retries
            else:
                print(f"Error in tap detection: {e}")
            
            if not should_exit:
                await asyncio.sleep(0.1)

# Modify main to include initial memory reset
async def main():
    global frame_instance, should_update_display
    should_update_display = True
    print(">>> Initializing Frame connection...")

    try:
        async with Frame() as frame:
            frame_instance = frame
            print("Connected to Frame!")
            
            # Reset memory on startup
            await reset_frame_memory(frame)
            
            # Show initial display
            await update_default_display(frame, text_gen)
            print("Default display active - Press Ctrl+C to exit...")
            
            # Create tap detection task
            tap_task = asyncio.create_task(detect_taps(frame, text_gen))
            
            # Main loop with periodic updates
            while not should_exit:
                if should_update_display:  # Only update if flag is True
                    await asyncio.sleep(8)  # Update every 8 seconds
                    if should_exit:
                        break
                    await update_default_display(frame, text_gen)
                else:
                    await asyncio.sleep(0.1)  # Short sleep when not updating
            
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