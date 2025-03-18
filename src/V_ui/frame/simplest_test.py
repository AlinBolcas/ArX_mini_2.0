import asyncio
from pathlib import Path
import sys
import signal
from frame_sdk import Frame
from frame_sdk.display import Alignment, PaletteColors
import time

# Global frame reference for cleanup
frame_instance = None
should_exit = False  # Flag to control the main loop

async def cleanup_display():
    """Clean up display before exit"""
    if frame_instance and frame_instance.bluetooth.is_connected():
        try:
            print("\nCleaning up display...")
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

async def show_default_display(frame):
    """Show the default Hello World display"""
    try:
        # Clear display first
        await frame.display.clear()
        
        # Write text elements
        await frame.display.write_text(
            "BIG!",
            y=100,
            align=Alignment.TOP_CENTER,
            color=PaletteColors.SKYBLUE
        )
        
        await frame.display.write_text(
            "Hello World!",
            align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.YELLOW
        )
        
        # Show everything at once
        await frame.display.show()
        
    except Exception as e:
        print(f"Error showing default display: {e}")

async def handle_tap(frame):
    """Handle single tap"""
    try:
        print("Single tap detected!")
        await frame.display.clear()
        await frame.display.write_text(
            "I've been touched!",
            align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.ORANGE
        )
        await frame.display.show()
        await asyncio.sleep(1)
        await show_default_display(frame)
    except Exception as e:
        print(f"Error handling tap: {e}")

async def handle_double_tap(frame):
    """Handle double tap"""
    try:
        print("Double tap detected!")
        await frame.display.clear()
        await frame.display.write_text(
            "I'm hot!",
            align=Alignment.MIDDLE_CENTER,
            color=PaletteColors.RED
        )
        await frame.display.show()
        await asyncio.sleep(1)
        await show_default_display(frame)
    except Exception as e:
        print(f"Error handling double tap: {e}")

async def detect_taps(frame):
    """Detect single and double taps with longer window"""
    double_tap_window = 1.3  # Increased to 1.3s for better detection
    
    while not should_exit:  # Check exit flag
        try:
            # Wait for first tap
            await frame.motion.wait_for_tap()
            current_time = time.time()
            print("First tap detected! Waiting for potential second tap...")
            
            try:
                # Wait longer for second tap
                await asyncio.wait_for(
                    frame.motion.wait_for_tap(), 
                    timeout=double_tap_window
                )
                # If we get here, it's a double tap
                print("DOUBLE TAP DETECTED!")
                await handle_double_tap(frame)
            except asyncio.TimeoutError:
                # If no second tap within window, it's a single tap
                print("SINGLE TAP DETECTED!")
                await handle_tap(frame)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in tap detection: {e}")
            if not should_exit:  # Only sleep if we're not exiting
                await asyncio.sleep(0.1)

async def main():
    global frame_instance
    print("Connecting to Frame...")
    
    try:
        async with Frame() as frame:
            frame_instance = frame
            print("Connected!")
            
            # Show initial display
            await show_default_display(frame)
            print("Displaying 'Hello World!' - Press Ctrl+C to exit...")
            
            # Create tap detection task
            tap_task = asyncio.create_task(detect_taps(frame))
            
            # Main loop
            while not should_exit:
                await asyncio.sleep(0.1)
            
            # Clean up
            print("Starting cleanup...")
            tap_task.cancel()
            try:
                await tap_task
            except asyncio.CancelledError:
                pass
            
            await cleanup_display()
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        should_exit = True  # Ensure the exit flag is set
    except Exception as e:
        print(f"Fatal error: {e}")
