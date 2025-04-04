# src/IV_pipelines/blender_gen/ui_assetsGen_kivy.py

# make the scrolling way more responsive and faster pls
# when pressing upload image, can you not make it open the operating system default file browser (instead of this kivy one?)
# can you add placeholders for only 3 images? can make them bigger pls so they fill the full width of the container
# if I double click on any of the images it should open them up in quicklook for MacOS or whatever is similar for windows.
# move the selected image path down where the generated model path is.
# make the 3d model preview bigger to fill the whole width of the container.
# render preview preview do the same, should fill the whole width of the container
# add a button which opens the output folder of the renders/gifs/mp4s in stage 3.
# rename the title of the app to ArX - AssetsGen

import kivy
kivy.require('2.1.0') # Ensure compatibility

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.switch import Switch
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty, ObjectProperty, BooleanProperty, ListProperty
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle, RoundedRectangle
from kivy.lang import Builder
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.widget import Widget
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from pathlib import Path
import time
import random
import os
import shutil
import platform
import subprocess
import threading

# Set much faster scrolling speed with inertia
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')  # Allow mouse multi-touch
Config.set('kivy', 'scroll_timeout', '25')  # Even lower timeout for faster responsiveness
Config.set('kivy', 'scroll_distance', '2')  # Shorter distance needed to start scrolling
Config.set('kivy', 'scroll_friction', '0.05')  # Much lower friction = faster scrolling

# Add scroll effect with inertia
from kivy.effects.scroll import ScrollEffect
from kivy.effects.kinetic import KineticEffect

# Custom scroll effect with increased speed and inertia
class FastScrollEffect(ScrollEffect):
    """Scroll effect that feels faster and has more inertia"""
    
    friction = 0.05  # Lower friction = more inertia (default is 0.5)
    min_velocity = 0.5  # Lower value means longer scrolling after touch release
    
    def __init__(self, **kwargs):
        super(FastScrollEffect, self).__init__(**kwargs)
        
    def on_touch_up(self, touch):
        # Increase scroll velocity on touch release for more inertia
        if hasattr(self, 'velocity'):
            self.velocity = self.velocity * 1.5  # Boost velocity
        return super(FastScrollEffect, self).on_touch_up(touch)

# Constants
ACCENT_BLUE = (0.25, 0.5, 0.95, 1)  # Accent blue color used throughout UI
DARK_GRAY = (0.15, 0.17, 0.19, 1)   # Dark gray for sidebar

# Simulate backend calls (Replace with actual API calls later)
# from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
# from src.I_integrations.replicate_API import ReplicateAPI
# from src.I_integrations.tripo_API import TripoAPI
# from src.IV_pipelines.blender_gen.blenderGen import BlenderGen

# Define custom classes for labels
class HeaderLabel(Label):
    """Header label with styling"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (0.95, 0.95, 0.95, 1)
        self.font_size = '20sp'
        self.bold = True
        self.size_hint_y = None
        self.height = dp(40)
        
        with self.canvas.before:
            Color(0, 0, 0, 0)  # Transparent
            Rectangle(pos=self.pos, size=self.size)

class ModernLabel(Label):
    """Modern styled label"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.color = (0.9, 0.9, 0.9, 1)
        self.font_size = '16sp'
        self.text_size = self.size
        self.halign = 'left'
        self.valign = 'middle'
        self.padding = (dp(5), dp(5))
        self.bind(size=self._update_text_size)
    
    def _update_text_size(self, instance, value):
        self.text_size = value

class ModernTextInput(TextInput):
    """Modern styled text input"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0.12, 0.14, 0.18, 1)
        self.foreground_color = (0.9, 0.9, 0.9, 1)
        self.cursor_color = (0.9, 0.9, 0.9, 1)
        self.font_size = '16sp'
        self.padding = dp(12)
        self.multiline = True
        self.size_hint_y = None
        if not 'height' in kwargs:
            self.height = dp(40)
        self.bind(minimum_height=self._update_height)
        
    def _update_height(self, instance, value):
        if value > self.height:
            self.height = max(dp(40), value)

class ModernButton(Button):
    """Modern styled button with rounded corners"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)  # Transparent for custom drawing
        self.color = (0.9, 0.9, 0.9, 1)
        self.font_size = '16sp'
        self.bold = True
        self.size_hint_y = None
        self.height = dp(50)
        self.bind(pos=self._update_canvas, size=self._update_canvas, state=self._update_canvas)
        
    def _update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'normal':
                Color(0.25, 0.5, 0.95, 1)
            else:
                Color(0.3, 0.6, 1, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(8)])

class ModernProgressBar(ProgressBar):
    """Modern styled progress bar with rounded corners"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max = 100
        self.value = 0
        self.size_hint_y = None
        self.height = dp(8)
        self.bind(pos=self._update_canvas, size=self._update_canvas, value=self._update_canvas)
        
    def _update_canvas(self, *args):
        self.canvas.clear()
        with self.canvas:
            # Background
            Color(0.14, 0.16, 0.20, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(4)])
            # Progress
            Color(0.25, 0.5, 0.95, 1)
            progress_width = self.width * (self.value / self.max)
            if progress_width > 0:
                RoundedRectangle(
                    pos=self.pos, 
                    size=(progress_width, self.height), 
                    radius=[dp(4)]
                )

class ModelTypeButton(ToggleButton):
    """Toggle button for model type selection"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group = 'model_type'
        self.background_color = (0, 0, 0, 0)
        self.color = (0.9, 0.9, 0.9, 1)
        self.font_size = '16sp'
        self.size_hint_y = None
        self.height = dp(45)
        self.bind(pos=self._update_canvas, size=self._update_canvas, state=self._update_canvas)
        
    def _update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'down':
                Color(*ACCENT_BLUE)  # Use the same accent blue as buttons
            else:
                Color(0.14, 0.18, 0.22, 1)
            RoundedRectangle(
                pos=self.pos, 
                size=self.size, 
                radius=[dp(10)]  # More rounded corners
            )

# Custom KV language string for styling
KV_STYLING = '''
<SectionCard>:
    orientation: 'vertical'
    padding: dp(15)
    spacing: dp(10)
    size_hint_y: None
    canvas.before:
        Color:
            rgba: 0.08, 0.1, 0.15, 1  # Dark blue/black background
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]

<SegmentButton>:
    group: 'segments'
    background_color: 0, 0, 0, 0
    color: 0.9, 0.9, 0.9, 1
    font_size: '16sp'
    bold: True
    size_hint_y: None
    height: dp(50)
    canvas.before:
        Color:
            rgba: (0.25, 0.5, 0.95, 1) if self.state == 'down' else (0.12, 0.18, 0.25, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]
            
<ImageSelectButton>:
    canvas.before:
        Color:
            rgba: (0.25, 0.5, 0.95, 1) if self.selected else (0.14, 0.16, 0.20, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(8)]
'''

# Load the custom KV language styling
Builder.load_string(KV_STYLING)

class SectionCard(BoxLayout):
    """Card-style container for section content with rounded corners and background"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set minimum height based on children dynamically
        self.bind(children=self._update_height, padding=self._update_height, spacing=self._update_height)
    
    def _update_height(self, *args):
        # Calculate height based on children
        height = sum(c.height + self.spacing for c in self.children) + 2 * self.padding[1]
        self.height = max(dp(100), height)  # Minimum height of 100dp

class SegmentButton(ToggleButton):
    """Toggle button styled as a segment"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.group = 'segments'
        self.background_color = (0, 0, 0, 0)  # Transparent for custom canvas
        self.color = (0.9, 0.9, 0.9, 1)
        self.font_size = '16sp'
        self.bold = True
        self.size_hint_y = None
        self.height = dp(50)
        self.bind(pos=self._update_canvas, size=self._update_canvas, state=self._update_canvas)
        
    def _update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            if self.state == 'down':
                Color(*ACCENT_BLUE)  # Use the same accent blue as buttons
            else:
                Color(0.12, 0.18, 0.25, 1)
            RoundedRectangle(
                pos=self.pos, 
                size=self.size, 
                radius=[dp(10)]  # More rounded corners
            )

class ImageSelectButton(ButtonBehavior, BoxLayout):
    """Custom selectable image button with proper styling"""
    source = StringProperty("")
    selected = BooleanProperty(False)
    index = NumericProperty(0)
    last_click_time = NumericProperty(0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, None)
        self.height = dp(200)  # Make images taller
        
        # Create image widget
        self.img = Image(source=self.source, size_hint=(1, 1))
        self.add_widget(self.img)
        
        # Bind properties
        self.bind(source=self._update_source, selected=self._update_selected)
        
        # Register the on_double_click event
        self.register_event_type('on_double_click')
    
    def _update_source(self, instance, value):
        self.img.source = value
    
    def _update_selected(self, instance, value):
        # Trigger redraw
        self.canvas.ask_update()
    
    def on_press(self):
        current_time = time.time()
        # Check for double click (within 0.3 seconds)
        if current_time - self.last_click_time < 0.3:
            # This is a double-click
            self.dispatch('on_double_click')
        else:
            # This is a single click
            self.selected = True
            
        self.last_click_time = current_time
    
    def on_double_click(self):
        """Called when the image is double-clicked"""
        pass  # Default implementation does nothing

class StatusBar(BoxLayout):
    """Status bar for displaying current state"""
    status_text = StringProperty("Ready")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        self.size_hint_y = None
        self.height = dp(40)
        self.padding = [dp(10), dp(5)]
        
        # Create status label
        self.status_label = ModernLabel(text=f"Status: {self.status_text}")
        self.add_widget(self.status_label)
        
        # Bind the status text to update the label
        self.bind(status_text=self._update_status)
    
    def _update_status(self, instance, value):
        self.status_label.text = f"Status: {value}"

class LogPanel(BoxLayout):
    """Side panel for displaying process logs"""
    log_text = StringProperty("")
    is_expanded = BooleanProperty(False)  # Start collapsed

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint_x = None
        self.width = dp(300) if self.is_expanded else dp(40)
        self.padding = dp(0)
        self.spacing = dp(0)

        # Add dark gray background instead of blue
        with self.canvas.before:
            Color(*DARK_GRAY)  # Dark gray instead of bright blue
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)

        # Toggle Button
        self.toggle_button = Button(
            text="<" if self.is_expanded else ">",
            size_hint_y=None,
            height=dp(40),
            background_color=(0.2, 0.22, 0.25, 1),  # Darker shade for button
            color=(1, 1, 1, 1),
            font_size='18sp',
            on_press=self.toggle_panel
        )
        self.add_widget(self.toggle_button)

        # Scrollable Log Area
        self.scroll_view = ScrollView(size_hint=(1, 1))
        self.log_label = Label(
            text=self.log_text,
            size_hint_y=None,
            halign='left',
            valign='top',
            text_size=(dp(280), None),
            color=(1, 1, 1, 1),
            padding=(dp(10), dp(10)),
            markup=True  # Enable simple text styling
        )
        self.log_label.bind(texture_size=self.log_label.setter('size'))
        self.scroll_view.add_widget(self.log_label)
        self.add_widget(self.scroll_view)

        # Initial visibility based on is_expanded
        self.scroll_view.opacity = 1 if self.is_expanded else 0
        self.log_label.opacity = 1 if self.is_expanded else 0
        
    def _update_rect(self, instance, value):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size

    def add_log(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text += f"\n[{timestamp}] {message}"
        self.log_label.text = self.log_text
        # Scroll to bottom
        self.scroll_view.scroll_y = 0

    def toggle_panel(self, instance):
        self.is_expanded = not self.is_expanded
        if self.is_expanded:
            self.width = dp(300)
            self.toggle_button.text = "<"
            self.scroll_view.opacity = 1
            self.log_label.opacity = 1
        else:
            self.width = dp(40)
            self.toggle_button.text = ">"
            self.scroll_view.opacity = 0
            self.log_label.opacity = 0

class BlenderGenUI(BoxLayout):
    """Main UI Root Widget"""
    log_panel = ObjectProperty(None)
    status_bar = ObjectProperty(None)
    concept_image_path = StringProperty("")  # Store path of selected/generated image
    model_3d_path = StringProperty("")       # Store path of generated 3d model
    render_output_path = StringProperty("")  # Store path of rendered output (GIF/MP4)
    selected_image_index = NumericProperty(-1)  # Index of selected image, -1 means none selected

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'  # Main layout: Content | Log Panel
        
        # App title and logo
        self.app_title = "ArX - AssetsGen"  # Updated title
        
        # Store image paths
        self.image_paths = []
        
        # --- Main Content Area ---
        self.main_content = BoxLayout(orientation='vertical', padding=dp(0), spacing=dp(0))
        
        # App header with logo and title
        self.header = BoxLayout(
            orientation='horizontal', 
            size_hint_y=None, 
            height=dp(80),
            padding=[dp(20), dp(10)],
        )
        
        # Add blue background to header
        with self.header.canvas.before:
            Color(0.05, 0.07, 0.12, 1)  # Dark blue
            Rectangle(pos=self.header.pos, size=self.header.size)
        self.header.bind(pos=self._update_header_bg, size=self._update_header_bg)
        
        # Add logo - use a placeholder icon initially
        logo_path = self.get_logo_path()
        self.logo_img = Image(
            source=logo_path,
            size_hint=(None, None),
            size=(dp(60), dp(60))
        )
        self.header.add_widget(self.logo_img)
        
        # Add title
        self.title_label = Label(
            text=self.app_title,
            font_size='24sp',
            bold=True,
            color=(0.9, 0.9, 0.9, 1),
            size_hint=(1, 1)
        )
        self.header.add_widget(self.title_label)
        
        self.main_content.add_widget(self.header)
        
        # Create a ScrollView for the sections - with rounded scrollbars and faster scrolling
        self.scroll_view = ScrollView(
            size_hint=(1, 1),
            bar_width=dp(10),
            scroll_type=['bars', 'content'],
            bar_color=ACCENT_BLUE,
            bar_inactive_color=(0.2, 0.3, 0.5, 0.5),
            effect_cls=FastScrollEffect,  # Use our custom fast scroll effect
            do_scroll_x=False  # Only scroll vertically
        )
        self.scroll_content = BoxLayout(
            orientation='vertical', 
            size_hint_y=None,
            padding=[dp(20), dp(20)],
            spacing=dp(20)
        )
        self.scroll_content.bind(minimum_height=self.scroll_content.setter('height'))
        self.scroll_view.add_widget(self.scroll_content)
        self.main_content.add_widget(self.scroll_view)
        
        # Create the sections
        self.setup_concept_section()
        self.setup_model_section()
        self.setup_render_section()
        
        # Add status bar
        self.status_bar = StatusBar()
        self.main_content.add_widget(self.status_bar)
        
        self.add_widget(self.main_content)
        
        # --- Log Panel ---
        self.log_panel = LogPanel()
        self.add_widget(self.log_panel)
        
        self.log("ArX - AssetsGen initialized")
    
    def _update_header_bg(self, instance, value):
        instance.canvas.before.clear()
        with instance.canvas.before:
            Color(0.05, 0.07, 0.12, 1)  # Dark blue
            Rectangle(pos=instance.pos, size=instance.size)
            
    def get_logo_path(self):
        """Get the path to the logo image, copy from local assets if available"""
        # Create the assets directory if it doesn't exist
        assets_dir = Path('assets')
        assets_dir.mkdir(exist_ok=True)
        
        # Check for local logo
        app_root = Path(__file__).parent
        local_logo_path = app_root / 'logo.png'
        destination_logo_path = assets_dir / 'logo.png'
        
        if local_logo_path.exists():
            # Copy the local logo to assets if it exists
            try:
                shutil.copy(local_logo_path, destination_logo_path)
                self.log(f"Copied logo from {local_logo_path} to {destination_logo_path}")
                return str(destination_logo_path)
            except Exception as e:
                self.log(f"Error copying logo: {e}")
                
        # If destination logo exists, use it
        if destination_logo_path.exists():
            return str(destination_logo_path)
            
        # Fallback to Kivy atlas image
        return 'atlas://data/images/defaulttheme/filechooser_folder'  # Better placeholder
    
    def open_system_file_dialog(self, file_type='image'):
        """Open the system's native file dialog for selecting files
        
        Args:
            file_type: 'image' or 'model' to control file filters
        """
        self.log(f"Opening system file browser for {file_type}...")
        
        # Start file selection in a separate thread to avoid blocking the UI
        thread = threading.Thread(target=lambda: self._system_file_dialog_thread(file_type))
        thread.daemon = True
        thread.start()
    
    def _system_file_dialog_thread(self, file_type='image'):
        """Thread function for system file dialog
        
        Args:
            file_type: 'image' or 'model' to control file filters
        """
        selected_file = None
        
        try:
            # Set up file filters based on type
            if file_type == 'image':
                file_types = ["png", "jpg", "jpeg", "webp"]
                prompt_text = "Select an Image:"
            else:  # model
                file_types = ["glb", "fbx", "obj", "blend"]
                prompt_text = "Select a 3D Model:"
            
            # Different approaches based on platform
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                # Use AppleScript to show the file picker
                from subprocess import Popen, PIPE
                
                # Create file type string for AppleScript
                type_str = ", ".join(f'"{ft}"' for ft in file_types)
                
                script = f'''
                tell application "System Events"
                    activate
                    set theFile to choose file with prompt "{prompt_text}" of type {{{type_str}}}
                    set thePath to POSIX path of theFile
                    return thePath
                end tell
                '''
                
                process = Popen(['osascript', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)
                stdout, stderr = process.communicate(script)
                
                if stdout.strip():
                    selected_file = stdout.strip()
                    
            elif system == 'Windows':
                # Use Windows file dialog
                try:
                    import ctypes
                    import ctypes.wintypes as wintypes
                    
                    class BROWSEINFO(ctypes.Structure):
                        _fields_ = [
                            ("hwndOwner", wintypes.HWND),
                            ("pidlRoot", ctypes.c_void_p),
                            ("pszDisplayName", ctypes.c_char_p),
                            ("lpszTitle", ctypes.c_char_p),
                            ("ulFlags", ctypes.c_uint),
                            ("lpfn", ctypes.c_void_p),
                            ("lParam", ctypes.c_long),
                            ("iImage", ctypes.c_int)
                        ]
                    
                    # Try to use Windows native dialog 
                    from tkinter import Tk, filedialog
                    root = Tk()
                    root.withdraw()  # Hide the main window
                    
                    # Create file types for tkinter dialog
                    tk_filetypes = [(f"{file_type.title()} files", 
                                     " ".join(f"*.{ft}" for ft in file_types))]
                    
                    file_path = filedialog.askopenfilename(
                        title=prompt_text,
                        filetypes=tk_filetypes
                    )
                    root.destroy()
                    
                    if file_path:
                        selected_file = file_path
                        
                except ImportError:
                    # Fallback if tkinter is not available
                    self.log(f"Cannot open system file dialog - tkinter not available")
                    # Schedule showing Kivy file chooser instead
                    Clock.schedule_once(lambda dt: self.show_file_chooser(file_type=file_type))
                    return
                    
            elif system == 'Linux':
                # Try to use zenity, kdialog, or other common dialog programs
                try:
                    import subprocess
                    
                    # Create file filter string for dialog programs
                    filter_str = " ".join(f"*.{ft}" for ft in file_types)
                    
                    # Try zenity first (common on many distributions)
                    try:
                        result = subprocess.check_output([
                            'zenity', '--file-selection',
                            f'--title={prompt_text}',
                            f'--file-filter={file_type.title()} files | {filter_str}'
                        ], universal_newlines=True)
                        selected_file = result.strip()
                    except (subprocess.SubprocessError, FileNotFoundError):
                        # Try kdialog (KDE)
                        try:
                            result = subprocess.check_output([
                                'kdialog', '--getopenfilename',
                                '.', f'{file_type.title()} files ({filter_str})'
                            ], universal_newlines=True)
                            selected_file = result.strip()
                        except (subprocess.SubprocessError, FileNotFoundError):
                            # Fallback to Kivy file chooser
                            Clock.schedule_once(lambda dt: self.show_file_chooser(file_type=file_type))
                            return
                except Exception as e:
                    self.log(f"Error using Linux file dialog: {e}")
                    Clock.schedule_once(lambda dt: self.show_file_chooser(file_type=file_type))
                    return
            
            if selected_file:
                # Process the file in the main thread
                if file_type == 'image':
                    Clock.schedule_once(lambda dt: self.process_selected_file([selected_file], None))
                else:  # model
                    Clock.schedule_once(lambda dt: self.process_selected_model([selected_file], None))
            else:
                # If no file was selected, log a message
                Clock.schedule_once(lambda dt: self.log(f"No {file_type} file selected"))
                
        except Exception as e:
            # Log any errors and fallback to Kivy file chooser
            Clock.schedule_once(lambda dt: self.log(f"Error using system file dialog: {e}"))
            Clock.schedule_once(lambda dt: self.show_file_chooser(file_type=file_type))
    
    def open_folder_with_system(self, folder_path):
        """Open a folder with the system's file explorer"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                folder_path.mkdir(parents=True, exist_ok=True)
                
            if platform.system() == 'Darwin':  # macOS
                subprocess.Popen(['open', str(folder_path)])
            elif platform.system() == 'Windows':
                os.startfile(str(folder_path))
            else:
                # Linux or other systems
                subprocess.Popen(['xdg-open', str(folder_path)])
                
            self.log(f"Opened folder: {folder_path}")
                
        except Exception as e:
            self.log(f"Error opening folder: {e}")
    
    def process_selected_file(self, selection, popup):
        """Process the selected file from file chooser"""
        if selection and len(selection) > 0:
            # Create assets directory if needed
            assets_dir = Path('assets')
            assets_dir.mkdir(exist_ok=True)
            
            # Copy selected file to assets directory
            src_path = selection[0]
            filename = os.path.basename(src_path)
            dest_path = assets_dir / filename
            
            try:
                shutil.copy(src_path, dest_path)
                self.log(f"Copied image: {filename}")
                
                # Update image paths and gallery - now just a single image
                self.image_paths = [str(dest_path)]
                self.update_image_gallery()
                if self.image_paths:
                    self.select_image(0)
                
            except Exception as e:
                self.log(f"Error copying image: {e}")
        
        # Close popup if provided
        if popup:
            popup.dismiss()
            
        # If it was uploaded via system dialog, reset the toggle button
        if hasattr(self, 'upload_image_btn') and self.upload_image_btn.state == 'down':
            # Switch back to text-to-image mode
            self.upload_image_btn.state = 'normal'
            self.text_to_image_btn.state = 'down'
    
    def show_file_chooser(self, file_type='image'):
        """Show the Kivy file chooser popup (fallback)
        
        Args:
            file_type: 'image' or 'model' to control file filters
        """
        # Set up file filters based on type
        if file_type == 'image':
            filters = ['*.png', '*.jpg', '*.jpeg', '*.webp']
            title = "Select Image"
        else:  # model
            filters = ['*.glb', '*.fbx', '*.obj', '*.blend']
            title = "Select 3D Model"
        
        content = BoxLayout(orientation='vertical')
        
        # Use a dark theme for the file chooser
        file_chooser = FileChooserListView(
            path=os.path.expanduser("~"),
            filters=filters,
            size_hint=(1, 1)
        )
        content.add_widget(file_chooser)
        
        # Buttons row
        buttons = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        
        cancel_btn = Button(
            text="Cancel",
            size_hint_x=0.5
        )
        select_btn = Button(
            text="Select",
            size_hint_x=0.5
        )
        
        buttons.add_widget(cancel_btn)
        buttons.add_widget(select_btn)
        content.add_widget(buttons)
        
        # Create and open popup
        popup = Popup(
            title=title,
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        # Bind events
        cancel_btn.bind(on_release=popup.dismiss)
        
        # Different processing based on file type
        if file_type == 'image':
            select_btn.bind(on_release=lambda x: self.process_selected_file(file_chooser.selection, popup))
        else:  # model
            select_btn.bind(on_release=lambda x: self.process_selected_model(file_chooser.selection, popup))
        
        popup.open()
    
    def process_selected_model(self, selection, popup):
        """Process the selected 3D model file from file chooser"""
        if selection and len(selection) > 0:
            # Create models directory if needed
            models_dir = Path('output/models')
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy selected file to models directory
            src_path = selection[0]
            filename = os.path.basename(src_path)
            dest_path = models_dir / filename
            
            try:
                shutil.copy(src_path, dest_path)
                self.log(f"Copied model: {filename}")
                
                # Update model path
                self.model_3d_path = str(dest_path)
                self.model_path_label.text = self.model_3d_path
                
                # Use a placeholder image for preview
                self.model_preview.source = 'atlas://data/images/defaulttheme/filechooser_folder'
                
                # Update status
                self.update_status(f"Model loaded: {filename}")
                
            except Exception as e:
                self.log(f"Error copying model: {e}")
        
        # Close popup if provided
        if popup:
            popup.dismiss()
    
    def setup_concept_section(self):
        """Set up the concept & images section"""
        # Create section card
        concept_section = SectionCard()
        
        # Add section header
        concept_section.add_widget(HeaderLabel(text="1. Concept & Images"))
        
        # Add segment buttons for input type
        segment_container = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(10)
        )
        
        self.text_to_image_btn = SegmentButton(
            text="Text to Image",
            state='down'  # Default selected
        )
        self.upload_image_btn = SegmentButton(
            text="Upload Image",
            on_press=self.toggle_upload_mode
        )
        
        segment_container.add_widget(self.text_to_image_btn)
        segment_container.add_widget(self.upload_image_btn)
        concept_section.add_widget(segment_container)
        
        # Label for prompt
        concept_section.add_widget(ModernLabel(
            text="Concept or Prompt",
            size_hint_y=None,
            height=dp(30)
        ))
        
        # Add concept input field
        self.prompt_input = ModernTextInput(
            hint_text="Enter a concept or detailed description...",
            height=dp(120),
            multiline=True
        )
        concept_section.add_widget(self.prompt_input)
        
        # Add AI refinement toggle
        refine_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(40),
            padding=[0, dp(10)]
        )
        refine_row.add_widget(ModernLabel(
            text="Use AI to refine prompt",
            size_hint_x=0.7
        ))
        self.refine_switch = Switch(
            active=False,
            size_hint_x=0.3
        )
        refine_row.add_widget(self.refine_switch)
        concept_section.add_widget(refine_row)
        
        # Add generate button
        self.generate_btn = ModernButton(
            text="Generate Images",
            on_press=self.generate_images
        )
        concept_section.add_widget(self.generate_btn)
        
        # Add progress bar
        self.image_progress = ModernProgressBar()
        concept_section.add_widget(self.image_progress)
        
        # Add image gallery label
        concept_section.add_widget(ModernLabel(
            text="Generated Images",
            size_hint_y=None,
            height=dp(30)
        ))
        
        # Create image gallery - now with 3 images in a row instead of 4
        self.image_gallery = GridLayout(
            cols=3,
            spacing=dp(10),
            size_hint_y=None,
            height=dp(210)  # Increased height for taller images
        )
        concept_section.add_widget(self.image_gallery)
        
        # Add to main content
        self.scroll_content.add_widget(concept_section)
    
    def toggle_upload_mode(self, instance):
        """Toggle between text-to-image and upload modes"""
        if instance.state == 'down':
            # When upload button is active
            self.text_to_image_btn.state = 'normal'
            # Show system file dialog instead of Kivy's dialog
            self.open_system_file_dialog()
        else:
            # Default to text-to-image if neither is selected
            self.text_to_image_btn.state = 'down'
    
    def setup_model_section(self):
        """Set up the 3D model generation section"""
        # Create section card
        model_section = SectionCard()
        
        # Add section header
        model_section.add_widget(HeaderLabel(text="2. 3D Model Generation"))
        
        # Add model type row
        model_type_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(10)
        )
        
        model_type_row.add_widget(ModernLabel(
            text="Model Type:",
            size_hint_x=0.3
        ))
        
        self.model_type_container = BoxLayout(
            orientation='horizontal',
            size_hint_x=0.7,
            spacing=dp(10)
        )
        
        # Model type buttons
        self.tripo_btn = ModelTypeButton(text="Tripo", state='down')
        self.trellis_btn = ModelTypeButton(text="Trellis")
        self.hunyuan_btn = ModelTypeButton(text="Hunyuan")
        
        self.model_type_container.add_widget(self.tripo_btn)
        self.model_type_container.add_widget(self.trellis_btn)
        self.model_type_container.add_widget(self.hunyuan_btn)
        
        model_type_row.add_widget(self.model_type_container)
        model_section.add_widget(model_type_row)
        
        # Add model preview container - now full width
        preview_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(10),
            height=dp(300)  # Taller container for bigger preview
        )
        
        # Model preview image - now full width
        self.model_preview = Image(
            source='atlas://data/images/defaulttheme/filechooser_folder',  # Better placeholder icon
            size_hint_y=1
        )
        preview_container.add_widget(self.model_preview)
        
        model_section.add_widget(preview_container)
        
        # Path information (concept path and model path together)
        paths_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(100),
            spacing=dp(5)
        )
        
        # Concept path
        paths_container.add_widget(ModernLabel(
            text="Selected Concept Image:",
            size_hint_y=None,
            height=dp(25)
        ))
        
        self.concept_path_label = ModernLabel(
            text="No image selected",
            font_size='14sp',
            color=(0.7, 0.7, 0.7, 1),
            size_hint_y=None,
            height=dp(25)
        )
        paths_container.add_widget(self.concept_path_label)
        
        # Model path
        paths_container.add_widget(ModernLabel(
            text="Generated Model Path:",
            size_hint_y=None,
            height=dp(25)
        ))
        
        self.model_path_label = ModernLabel(
            text="N/A",
            font_size='14sp',
            color=(0.7, 0.7, 0.7, 1)
        )
        paths_container.add_widget(self.model_path_label)
        
        model_section.add_widget(paths_container)
        
        # Action buttons row - layout for both generate and import buttons
        buttons_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(10)
        )
        
        # Add generate button
        self.generate_model_btn = ModernButton(
            text="Generate 3D Model",
            size_hint_x=0.7,
            on_press=self.generate_3d_model
        )
        buttons_row.add_widget(self.generate_model_btn)
        
        # Add import model button
        self.import_model_btn = ModernButton(
            text="Import Model",
            size_hint_x=0.3,
            on_press=self.import_model
        )
        buttons_row.add_widget(self.import_model_btn)
        
        model_section.add_widget(buttons_row)
        
        # Add progress bar
        self.model_progress = ModernProgressBar()
        model_section.add_widget(self.model_progress)
        
        # Add to main content
        self.scroll_content.add_widget(model_section)
    
    def import_model(self, instance):
        """Import an existing 3D model file"""
        self.open_system_file_dialog(file_type='model')
    
    def log(self, message):
        """Helper function to add log messages"""
        print(f"LOG: {message}")  # Also print to console
        if self.log_panel:
            self.log_panel.add_log(message)
    
    def update_status(self, message):
        """Update status bar message"""
        if self.status_bar:
            self.status_bar.status_text = message
    
    def update_image_gallery(self):
        """Update the image gallery with the available images"""
        # Clear existing images
        self.image_gallery.clear_widgets()
        
        # If no images, show placeholders - now only 3
        if not self.image_paths:
            for i in range(3):  # Only 3 placeholders
                placeholder = ImageSelectButton(
                    source='atlas://data/images/defaulttheme/document',  # Better placeholder icon
                    index=i,
                )
                self.image_gallery.add_widget(placeholder)
            return
        
        # Add image buttons
        for i, path in enumerate(self.image_paths):
            img_btn = ImageSelectButton(
                source=path,
                index=i,
                selected=(i == self.selected_image_index)
            )
            # Bind single click for selection
            img_btn.bind(on_press=lambda btn=img_btn: self.select_image(btn.index))
            
            # Bind double click to open file
            img_btn.bind(on_double_click=lambda btn=img_btn: self.open_file_with_system(btn.source))
            
            self.image_gallery.add_widget(img_btn)
    
    def select_image(self, index):
        """Handle image selection in gallery"""
        if 0 <= index < len(self.image_paths):
            self.selected_image_index = index
            self.concept_image_path = self.image_paths[index]
            self.log(f"Selected image {index+1}")
            
            # Update concept path in model section
            if hasattr(self, 'concept_path_label'):
                self.concept_path_label.text = self.concept_image_path
            
            # Update image buttons
            for i, child in enumerate(self.image_gallery.children):
                if isinstance(child, ImageSelectButton):
                    child.selected = (child.index == index)
    
    # --- Placeholder functionality ---
    
    def generate_images(self, instance):
        """Generate images from prompt"""
        prompt = self.prompt_input.text.strip()
        if not prompt:
            self.log("Error: Please enter a prompt")
            self.update_status("Error: Prompt required")
            return
        
        self.log(f"Generating images for prompt: {prompt}")
        self.update_status("Generating images...")
        self.image_progress.value = 0
        
        # Simulate API call and progress
        Clock.schedule_interval(self._update_image_progress, 0.1)
    
    def _update_image_progress(self, dt):
        """Simulate progress updates for image generation"""
        if self.image_progress.value < 100:
            self.image_progress.value += random.randint(1, 5)
            self.image_progress.value = min(self.image_progress.value, 100)
            return True
        
        # Progress complete
        self.log("Image generation complete")
        self.update_status("Images ready")
        
        # Create placeholder directory
        assets_dir = Path('assets')
        assets_dir.mkdir(exist_ok=True)
        
        # Create placeholder images - now only 3
        colors = [(52, 152, 219), (46, 204, 113), (231, 76, 60)]  # RGB tuples for colors
        self.image_paths = []
        
        try:
            # Try using PIL if available
            from PIL import Image as PILImage, ImageDraw
            have_pil = True
        except ImportError:
            have_pil = False
            self.log("PIL not available, using simple placeholder files")
        
        for i in range(3):  # Only generate 3 images
            img_path = assets_dir / f"placeholder_img_{i+1}.png"
            self.image_paths.append(str(img_path))
            
            if not img_path.exists():
                if have_pil:
                    try:
                        # Create a colored image with PIL
                        img_size = (400, 400)  # Larger size for better quality
                        img = PILImage.new('RGB', img_size, colors[i])
                        draw = ImageDraw.Draw(img)
                        draw.rectangle(
                            [(50, 50), (350, 350)],
                            outline=(255, 255, 255),
                            width=3  # Thicker outline
                        )
                        img.save(img_path)
                    except Exception as e:
                        self.log(f"Error creating image with PIL: {e}")
                        # If PIL fails, create a simple placeholder file
                        with open(img_path, 'w') as f:
                            f.write(f"placeholder_{i+1}")
                else:
                    # Create a simple placeholder file
                    with open(img_path, 'w') as f:
                        f.write(f"placeholder_{i+1}")
        
        # Update gallery and select first image
        self.update_image_gallery()
        if self.image_paths:
            self.select_image(0)
        
        return False  # Stop scheduling
    
    def generate_3d_model(self, instance):
        """Generate 3D model from selected image"""
        if self.selected_image_index < 0:
            self.log("Error: No image selected")
            self.update_status("Error: Select an image first")
            return
        
        # Get selected model type
        model_type = "Tripo"
        if self.trellis_btn.state == 'down':
            model_type = "Trellis"
        elif self.hunyuan_btn.state == 'down':
            model_type = "Hunyuan"
        
        self.log(f"Generating {model_type} 3D model from selected image")
        self.update_status(f"Generating {model_type} model...")
        self.model_progress.value = 0
        
        # Simulate API call and progress
        Clock.schedule_interval(self._update_model_progress, 0.2)
    
    def _update_model_progress(self, dt):
        """Simulate progress updates for model generation"""
        if self.model_progress.value < 100:
            self.model_progress.value += random.randint(1, 3)
            self.model_progress.value = min(self.model_progress.value, 100)
            return True
        
        # Progress complete
        self.log("3D model generation complete")
        self.update_status("Model ready")
        
        # Create placeholder directory
        output_dir = Path('output/models')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Get selected model type
        model_type = "Tripo"
        if self.trellis_btn.state == 'down':
            model_type = "Trellis"
        elif self.hunyuan_btn.state == 'down':
            model_type = "Hunyuan"
        
        # Create placeholder model file
        model_path = output_dir / f"generated_{model_type}_{timestamp}.glb"
        with open(model_path, 'w') as f:
            f.write("placeholder model file")
        
        # Update UI
        self.model_3d_path = str(model_path)
        self.model_path_label.text = self.model_3d_path
        
        # Use the selected image as a preview for the model
        self.model_preview.source = self.image_paths[self.selected_image_index]
        
        return False  # Stop scheduling
    
    def render_output(self, instance):
        """Render output from 3D model"""
        if not self.model_3d_path:
            self.log("Error: No 3D model generated")
            self.update_status("Error: Generate a model first")
            return
        
        interactive_mode = self.interactive_switch.active
        
        self.log(f"Rendering {'interactive' if interactive_mode else 'automated'} output from 3D model")
        self.update_status("Rendering in progress...")
        self.render_progress.value = 0
        
        # Simulate API call and progress
        Clock.schedule_interval(self._update_render_progress, 0.3)
    
    def _update_render_progress(self, dt):
        """Simulate progress updates for rendering"""
        if self.render_progress.value < 100:
            self.render_progress.value += random.randint(1, 2)
            self.render_progress.value = min(self.render_progress.value, 100)
            return True
        
        # Progress complete
        self.log("Rendering complete")
        self.update_status("Render complete")
        
        # Create placeholder directory
        output_dir = Path('output/renders')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Create placeholder render file
        render_path = output_dir / f"render_{timestamp}.gif"
        with open(render_path, 'w') as f:
            f.write("placeholder render file")
        
        # Update UI
        self.render_output_path = str(render_path)
        self.output_path_label.text = self.render_output_path
        
        # Use the selected image as a preview for the render
        self.output_preview.source = self.image_paths[self.selected_image_index]
        
        return False  # Stop scheduling

    def open_file_with_system(self, file_path):
        """Open a file with the system's default application"""
        try:
            if platform.system() == 'Darwin':  # macOS
                # Use 'open' command (supports QuickLook)
                subprocess.Popen(['open', file_path])
            elif platform.system() == 'Windows':
                # Use the default Windows application
                os.startfile(file_path)
            else:
                # Linux or other systems
                subprocess.Popen(['xdg-open', file_path])
                
            self.log(f"Opened file: {file_path}")
                
        except Exception as e:
            self.log(f"Error opening file: {e}")

    def open_output_folder(self, instance):
        """Open the output folder in the system's file explorer"""
        output_dir = Path('output/renders')
        output_dir.mkdir(parents=True, exist_ok=True)
        self.open_folder_with_system(str(output_dir))

    def scroll_to_section(self, section):
        """Scroll the view to show a specific section"""
        self.scroll_view.scroll_to(section)

    def setup_render_section(self):
        """Set up the rendering & output section"""
        # Create section card
        render_section = SectionCard()
        
        # Add section header
        render_section.add_widget(HeaderLabel(text="3. Rendering & Output"))
        
        # Add options row
        options_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(10)
        )
        
        options_row.add_widget(ModernLabel(
            text="Interactive Mode:",
            size_hint_x=0.5
        ))
        
        # Switch container - no ON/OFF labels
        switch_container = BoxLayout(
            orientation='horizontal',
            size_hint_x=0.5
        )
        
        self.interactive_switch = Switch(active=False)
        switch_container.add_widget(self.interactive_switch)
        
        options_row.add_widget(switch_container)
        render_section.add_widget(options_row)
        
        # Add output preview container - now full width
        preview_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(10),
            height=dp(300)  # Taller container for bigger preview
        )
        
        # Output preview image - now full width
        self.output_preview = Image(
            source='atlas://data/images/defaulttheme/filechooser_folder',  # Better placeholder icon
            size_hint_y=1
        )
        preview_container.add_widget(self.output_preview)
        
        render_section.add_widget(preview_container)
        
        # Output path
        output_path_container = BoxLayout(
            orientation='vertical',
            size_hint_y=None,
            height=dp(50),
            spacing=dp(5)
        )
        
        output_path_container.add_widget(ModernLabel(
            text="Rendered Output Path:",
            size_hint_y=None,
            height=dp(25)
        ))
        
        self.output_path_label = ModernLabel(
            text="N/A",
            font_size='14sp',
            color=(0.7, 0.7, 0.7, 1)
        )
        output_path_container.add_widget(self.output_path_label)
        
        render_section.add_widget(output_path_container)
        
        # Action buttons row
        buttons_row = BoxLayout(
            orientation='horizontal',
            size_hint_y=None,
            height=dp(60),
            spacing=dp(10)
        )
        
        # Add render button - renamed to "Render Outputs"
        self.render_btn = ModernButton(
            text="Render Outputs",
            size_hint_x=0.7,
            on_press=self.render_output
        )
        buttons_row.add_widget(self.render_btn)
        
        # Add open folder button
        self.open_output_folder_btn = ModernButton(
            text="Open Output Folder",
            size_hint_x=0.3,
            on_press=self.open_output_folder
        )
        buttons_row.add_widget(self.open_output_folder_btn)
        
        render_section.add_widget(buttons_row)
        
        # Add progress bar
        self.render_progress = ModernProgressBar()
        render_section.add_widget(self.render_progress)
        
        # Add to main content
        self.scroll_content.add_widget(render_section)

class BlenderGenApp(App):
    def build(self):
        # Set window background color
        Window.clearcolor = (0.04, 0.06, 0.1, 1)  # Dark blue/black
        return BlenderGenUI()

if __name__ == '__main__':
    BlenderGenApp().run()