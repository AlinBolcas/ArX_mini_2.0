import asyncio
from pathlib import Path
import datetime
from frame_sdk import Frame
from frame_sdk.display import Alignment, PaletteColors

async def load_lua_functions(frame):
    """Load Lua functions from file"""
    try:
        lua_file_path = Path(__file__).parent / 'lua_functions.lua'
        with open(lua_file_path, 'r') as file:
            lua_functions = file.read()
        await frame.run_lua(lua_functions, checked=True)  # Using checked=True to ensure it loads properly
        print("Lua functions loaded successfully!")
    except Exception as e:
        print(f"Error loading Lua functions: {e}")
        raise e

async def main():
    print("Connecting to Frame...")
    
    try:
        async with Frame() as frame:
            # Load our Lua functions first
            await load_lua_functions(frame)
            
            # Initialize display function index
            current_display = 0
            
            while True:
                try:
                    # Get current time and date
                    now = datetime.datetime.now()
                    battery = 85  # Example battery value
                    date = now.strftime("%d.%m.%y")
                    time_str = now.strftime("%I:%M %p")
                    
                    # Display current function
                    if current_display == 0:
                        print(f"Displaying battery: {battery}% and date: {date}")
                        await frame.run_lua(f'display_battery_and_date({battery}, "{date}")')
                        
                    elif current_display == 1:
                        print(f"Displaying time: {time_str}")
                        await frame.run_lua(f'display_time("{time_str}")')
                        
                    elif current_display == 2:
                        print("Displaying 'Hello World'")
                        await frame.run_lua('display_hello_world()')
                        
                    else:  # current_display == 3
                        quote = "Time is money."
                        print(f"Displaying all together")
                        await frame.run_lua(
                            f'display_all({battery}, "{date}", "{time_str}", "{quote}")'
                        )
                    
                    print("\nTap Frame to cycle display (Ctrl+C to exit)...")
                    await frame.motion.wait_for_tap()
                    
                    # Cycle to next display
                    current_display = (current_display + 1) % 4
                    
                except KeyboardInterrupt:
                    print("\nExiting...")
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    await asyncio.sleep(1)
            
            # Clean exit
            await frame.display.clear()
            await frame.display.write_text("Bye!", align=Alignment.MIDDLE_CENTER)
            await frame.display.show()
            await asyncio.sleep(0.1)
            
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting gracefully...") 