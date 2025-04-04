import sys
import os
import time
import threading
import asyncio
import io
from pathlib import Path
from datetime import datetime

# Add project root to Python path to fix import issues
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# PySide6 imports (modern Qt framework)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QScrollArea, 
                             QFileDialog, QMessageBox, QProgressBar, QCheckBox, QRadioButton, 
                             QButtonGroup, QFrame, QGroupBox, QSizePolicy, QSpacerItem)
from PySide6.QtCore import Qt, QSize, Signal, Slot, QThread, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette, QIcon

# Try imports with better error handling
try:
    # Blender Gen imports with proper path
    from src.IV_pipelines.blender_gen.blenderGen import BlenderGen
    
    # API imports with better error handling 
    try:
        from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
    except ImportError as e:
        print(f"Warning: Could not import OpenAI API: {e}")
        OpenAIResponsesAPI = None
        
    try:
        from src.I_integrations.replicate_API import ReplicateAPI
    except ImportError as e:
        print(f"Warning: Could not import Replicate API: {e}")
        ReplicateAPI = None
    
    try:
        # Fix for Tripo API import - try multiple possible paths
        try:
            from src.I_integrations.tripo_API import TripoAPI
        except ImportError:
            try:
                # Try without src prefix (common issue)
                from I_integrations.tripo_API import TripoAPI
            except ImportError:
                # Try archive folder (from looking at the file paths)
                from src.I_integrations.archive.tripo_API import TripoAPI
    except ImportError as e:
        print(f"Warning: Could not import Tripo API: {e}")
        TripoAPI = None
    
    # Utilities imports with better error handling
    try:
        from src.VI_utils.video_utils import video_to_gif, images_to_gif
        from src.VI_utils.image_utils import get_image_info
    except ImportError as e:
        print(f"Warning: Could not import utility functions: {e}")
        video_to_gif = images_to_gif = get_image_info = None
        
except ImportError as e:
    print(f"Error importing modules: {e}")
    # Fallback imports for direct directory run
    try:
        from blenderGen import BlenderGen
    except ImportError:
        print("Critical import error. Please run from the project root.")
        BlenderGen = None

# Import PIL for image handling
try:
    from PIL import Image, ImageQt
except ImportError:
    print("Warning: PIL not found, image preview will be disabled")
    Image = ImageQt = None

# Modern stylesheet (will be expanded)
STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}

QPushButton {
    background-color: #89b4fa;
    color: #1e1e2e;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #b4befe;
}

QPushButton:pressed {
    background-color: #74c7ec;
}

QPushButton:disabled {
    background-color: #6c7086;
    color: #9399b2;
}

QTextEdit, QLineEdit {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #89b4fa;
}

QGroupBox {
    border: 1px solid #45475a;
    border-radius: 4px;
    margin-top: 12px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #89b4fa;
}

QLabel {
    font-size: 14px;
}

QLabel#sectionTitle {
    font-size: 16px;
    font-weight: bold;
    color: #f5c2e7;
    padding-bottom: 4px;
    border-bottom: 1px solid #45475a;
}

QLabel#imageLabel {
    border: 1px solid #45475a;
    border-radius: 4px;
}

QProgressBar {
    border: 1px solid #45475a;
    border-radius: 4px;
    text-align: center;
    background-color: #313244;
}

QProgressBar::chunk {
    background-color: #89b4fa;
    width: 10px;
    margin: 0.5px;
}

QCheckBox, QRadioButton {
    spacing: 5px;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 15px;
    height: 15px;
}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    background-color: #89b4fa;
}

QScrollArea {
    border: none;
}

/* Section styling */
#PipelineSection {
    background-color: #181825;
    border-radius: 8px;
    margin: 5px;
    padding: 10px;
}

#PipelineSection:hover {
    background-color: #1e1e2e;
}

#SectionStatus {
    font-size: 12px;
    color: #a6adc8;
}

#SectionStatus[status="pending"] {
    color: #a6adc8;
}

#SectionStatus[status="active"] {
    color: #f9e2af;
}

#SectionStatus[status="complete"] {
    color: #a6e3a1;
}

#SectionStatus[status="error"] {
    color: #f38ba8;
}
"""

class ImageButton(QPushButton):
    """Custom image button for selectable thumbnails"""
    selected = Signal(int, str)  # Signal emitting index and path when selected
    
    def __init__(self, index, image_path, parent=None):
        super().__init__(parent)
        self.index = index
        self.image_path = image_path
        self.setCheckable(True)
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                border: 2px solid #45475a;
                border-radius: 6px;
                padding: 0px;
            }
            QPushButton:checked {
                border: 2px solid #89b4fa;
            }
        """)
        
        # Load and set image
        self.load_image(image_path)
        
        # Connect signal
        self.clicked.connect(self.on_clicked)
        
    def load_image(self, path):
        try:
            img = Image.open(path)
            # Resize for thumbnail
            img.thumbnail((140, 140))
            qimg = ImageQt.ImageQt(img)
            pixmap = QPixmap.fromImage(qimg)
            self.setIcon(QIcon(pixmap))
            self.setIconSize(QSize(140, 140))
        except Exception as e:
            print(f"Error loading image: {e}")
            # Set fallback
            self.setText(f"Image {self.index+1}")
            
    def on_clicked(self):
        self.selected.emit(self.index, self.image_path)
        
class PipelineSection(QGroupBox):
    """A collapsible section in the pipeline"""
    def __init__(self, title, parent=None):
        super().__init__(title, parent)
        self.setObjectName("PipelineSection")
        
        # Create main layout
        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)
        
        # Header with title and status
        self.header = QWidget()
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Status indicator
        self.status_label = QLabel("Pending")
        self.status_label.setObjectName("SectionStatus")
        self.status_label.setProperty("status", "pending")
        self.header_layout.addWidget(self.status_label, alignment=Qt.AlignRight)
        
        self.main_layout.addWidget(self.header)
        
        # Content container
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.addWidget(self.content)
        
        # Default state
        self.is_expanded = True
        self.is_enabled = True
    
    def set_status(self, status):
        """Update section status
        status: 'pending', 'active', 'complete', 'error'
        """
        self.status_label.setText(status.capitalize())
        self.status_label.setProperty("status", status.lower())
        self.style().unpolish(self.status_label)
        self.style().polish(self.status_label)
    
    def set_enabled(self, enabled):
        """Enable or disable section"""
        self.is_enabled = enabled
        self.content.setEnabled(enabled)
        self.status_label.setEnabled(enabled)
        if enabled:
            self.set_status("active")
        else:
            self.set_status("pending")
            
    def add_widget(self, widget):
        """Add widget to content layout"""
        self.content_layout.addWidget(widget)

class ComprehensivePipelineUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comprehensive Creation Pipeline")
        self.setMinimumSize(1000, 800)
        
        # Set up central widget with scrolling area
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(15)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create scrollable area for pipeline
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Container for sections
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(20)
        self.scroll_layout.setContentsMargins(15, 15, 15, 15)
        
        # Add scroll area to main layout
        self.scroll_area.setWidget(self.scroll_widget)
        self.main_layout.addWidget(self.scroll_area)
        
        # Progress bar at bottom of window
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.hide()
        self.main_layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFixedHeight(100)
        self.log_output.setPlaceholderText("Pipeline log will appear here...")
        self.main_layout.addWidget(self.log_output)
        
        # State variables
        self.generated_images = []
        self.selected_image_index = None
        self.generated_3d_model_path = None
        self.project_folder = None
        self.renders_folder = None
        
        # API clients (initialized on demand)
        self.openai_client = None
        self.replicate_client = None
        self.tripo_client = None
        
        # Output directory
        self.output_dir = project_root / "data" / "output" / "pipeline"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup UI sections
        self.setup_concept_section()
        self.setup_image_section()
        self.setup_3d_model_section()
        self.setup_blender_section()
        self.setup_output_section()
        
        # Add a spacer at the end
        self.scroll_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
        
        # Initialize pipeline state
        self.concept_section.set_enabled(True)
        self.image_section.set_enabled(False)
        self.model_section.set_enabled(False)
        self.blender_section.set_enabled(False)
        self.output_section.set_enabled(False)
        
        # Display welcome message
        self.log_message("Welcome to the Comprehensive Creation Pipeline!")
        self.log_message("Start by entering a concept or prompt in the first section.")
    
    def setup_concept_section(self):
        """Set up the concept input section"""
        self.concept_section = PipelineSection("1. Concept & Prompt")
        
        # Concept text input
        self.concept_input = QTextEdit()
        self.concept_input.setPlaceholderText("Enter a concept or detailed description of what you want to create...")
        self.concept_input.setMinimumHeight(80)
        self.concept_section.add_widget(self.concept_input)
        
        # AI refinement checkbox
        refine_widget = QWidget()
        refine_layout = QHBoxLayout(refine_widget)
        refine_layout.setContentsMargins(0, 0, 0, 0)
        
        self.refine_checkbox = QCheckBox("Refine with AI")
        self.refine_checkbox.setChecked(True)
        self.refine_checkbox.stateChanged.connect(self.toggle_refine_mode)
        refine_layout.addWidget(self.refine_checkbox)
        
        # Buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.generate_prompt_btn = QPushButton("Generate Enhanced Prompt")
        self.generate_prompt_btn.clicked.connect(self.generate_prompt)
        buttons_layout.addWidget(self.generate_prompt_btn)
        
        self.generate_images_btn = QPushButton("Generate Images →")
        self.generate_images_btn.clicked.connect(self.generate_images)
        self.generate_images_btn.setEnabled(False)
        buttons_layout.addWidget(self.generate_images_btn)
        
        # Enhanced prompt result (initially hidden)
        self.prompt_result = QTextEdit()
        self.prompt_result.setPlaceholderText("Enhanced prompt will appear here...")
        self.prompt_result.setReadOnly(True)
        self.prompt_result.setVisible(False)
        
        # Add widgets to section
        self.concept_section.add_widget(refine_widget)
        self.concept_section.add_widget(buttons_widget)
        self.concept_section.add_widget(self.prompt_result)
        
        # Add to pipeline
        self.scroll_layout.addWidget(self.concept_section)
    
    def setup_image_section(self):
        """Set up the image generation and selection section"""
        self.image_section = PipelineSection("2. Image Selection")
        
        # Image gallery container
        self.image_gallery = QWidget()
        self.image_gallery_layout = QHBoxLayout(self.image_gallery)
        self.image_gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.image_gallery_layout.setSpacing(10)
        self.image_gallery_layout.setAlignment(Qt.AlignLeft)
        
        # Add gallery to section
        self.image_section.add_widget(QLabel("Select an image to use for 3D model generation:"))
        self.image_section.add_widget(self.image_gallery)
        
        # Add proceed button (initially hidden)
        self.proceed_to_3d_btn = QPushButton("Generate 3D Model →")
        self.proceed_to_3d_btn.clicked.connect(self.proceed_to_3d_model)
        self.proceed_to_3d_btn.setEnabled(False)
        self.proceed_to_3d_btn.setVisible(False)
        self.image_section.add_widget(self.proceed_to_3d_btn)
        
        # Add to pipeline
        self.scroll_layout.addWidget(self.image_section)
    
    def setup_3d_model_section(self):
        """Set up the 3D model generation section"""
        self.model_section = PipelineSection("3. 3D Model Generation")
        
        # 3D model settings
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        
        # Quality settings
        settings_layout.addWidget(QLabel("Quality:"), 0, 0)
        quality_widget = QWidget()
        quality_layout = QHBoxLayout(quality_widget)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        
        self.quality_group = QButtonGroup(quality_widget)
        self.quality_standard = QRadioButton("Standard")
        self.quality_detailed = QRadioButton("Detailed")
        self.quality_standard.setChecked(True)
        self.quality_group.addButton(self.quality_standard, 1)
        self.quality_group.addButton(self.quality_detailed, 2)
        
        quality_layout.addWidget(self.quality_standard)
        quality_layout.addWidget(self.quality_detailed)
        settings_layout.addWidget(quality_widget, 0, 1)
        
        # Face limit
        settings_layout.addWidget(QLabel("Max Faces:"), 1, 0)
        self.face_limit_input = QLineEdit("200000")
        settings_layout.addWidget(self.face_limit_input, 1, 1)
        
        # Selected image preview
        self.selected_image_preview = QLabel("No image selected")
        self.selected_image_preview.setObjectName("imageLabel")
        self.selected_image_preview.setAlignment(Qt.AlignCenter)
        self.selected_image_preview.setMinimumHeight(200)
        self.selected_image_preview.setScaledContents(False)
        
        # 3D model generation button
        self.generate_3d_btn = QPushButton("Generate 3D Model")
        self.generate_3d_btn.clicked.connect(self.generate_3d_model)
        
        # 3D model result display
        self.model_result = QLabel("No model generated yet")
        
        # Add widgets to section
        self.model_section.add_widget(settings_widget)
        self.model_section.add_widget(QLabel("Selected Image:"))
        self.model_section.add_widget(self.selected_image_preview)
        self.model_section.add_widget(self.generate_3d_btn)
        self.model_section.add_widget(self.model_result)
        
        # Add proceed button
        self.proceed_to_blender_btn = QPushButton("Proceed to Blender →")
        self.proceed_to_blender_btn.clicked.connect(self.proceed_to_blender)
        self.proceed_to_blender_btn.setEnabled(False)
        self.proceed_to_blender_btn.setVisible(False)
        self.model_section.add_widget(self.proceed_to_blender_btn)
        
        # Add to pipeline
        self.scroll_layout.addWidget(self.model_section)
    
    def setup_blender_section(self):
        """Set up the Blender rendering section"""
        self.blender_section = PipelineSection("4. Blender Rendering")
        
        # Blender settings grid
        settings_widget = QWidget()
        settings_layout = QGridLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        
        # Template path
        settings_layout.addWidget(QLabel("Template (.blend):"), 0, 0)
        self.template_input = QLineEdit()
        settings_layout.addWidget(self.template_input, 0, 1)
        template_browse_btn = QPushButton("Browse...")
        template_browse_btn.clicked.connect(lambda: self.browse_file(
            self.template_input, "Select Blender Template", "Blender Files (*.blend)")
        )
        settings_layout.addWidget(template_browse_btn, 0, 2)
        
        # Material name
        settings_layout.addWidget(QLabel("Material Name:"), 1, 0)
        self.material_input = QLineEdit()
        self.material_input.setPlaceholderText("e.g., grey_procedural_MAT")
        settings_layout.addWidget(self.material_input, 1, 1, 1, 2)
        
        # Height
        settings_layout.addWidget(QLabel("Target Height (m):"), 2, 0)
        self.height_input = QLineEdit("1.8")
        settings_layout.addWidget(self.height_input, 2, 1, 1, 2)
        
        # Options
        options_widget = QWidget()
        options_layout = QHBoxLayout(options_widget)
        options_layout.setContentsMargins(0, 0, 0, 0)
        
        self.texture_checkbox = QCheckBox("Use Textures")
        self.texture_checkbox.setChecked(True)
        options_layout.addWidget(self.texture_checkbox)
        
        self.interactive_checkbox = QCheckBox("Interactive Mode")
        options_layout.addWidget(self.interactive_checkbox)
        
        # 3D model path (readonly)
        settings_layout.addWidget(QLabel("Selected 3D Model:"), 3, 0)
        self.model_path_input = QLineEdit()
        self.model_path_input.setReadOnly(True)
        settings_layout.addWidget(self.model_path_input, 3, 1, 1, 2)
        
        # Run Blender button
        self.run_blender_btn = QPushButton("Run Blender Process")
        self.run_blender_btn.clicked.connect(self.start_blender_process)
        
        # Add widgets to section
        self.blender_section.add_widget(settings_widget)
        self.blender_section.add_widget(options_widget)
        self.blender_section.add_widget(self.run_blender_btn)
        
        # Add to pipeline
        self.scroll_layout.addWidget(self.blender_section)
    
    def setup_output_section(self):
        """Set up the output processing section"""
        self.output_section = PipelineSection("5. Output Processing")
        
        # Render folder display
        folder_widget = QWidget()
        folder_layout = QHBoxLayout(folder_widget)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        
        folder_layout.addWidget(QLabel("Render Folder:"))
        self.render_folder_display = QLineEdit()
        self.render_folder_display.setReadOnly(True)
        self.render_folder_display.setPlaceholderText("No render completed yet")
        folder_layout.addWidget(self.render_folder_display)
        
        # GIF options
        gif_options = QGroupBox("GIF Options")
        gif_layout = QHBoxLayout(gif_options)
        
        gif_layout.addWidget(QLabel("Frame Duration (s):"))
        self.duration_input = QLineEdit("0.1")
        gif_layout.addWidget(self.duration_input)
        
        gif_layout.addWidget(QLabel("Resize (%):"))
        self.resize_input = QLineEdit("100")
        gif_layout.addWidget(self.resize_input)
        
        # Output buttons
        buttons_widget = QWidget()
        buttons_layout = QHBoxLayout(buttons_widget)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        
        self.create_gif_btn = QPushButton("Create GIF")
        self.create_gif_btn.clicked.connect(self.create_gif)
        self.create_gif_btn.setEnabled(False)
        buttons_layout.addWidget(self.create_gif_btn)
        
        self.create_mp4_btn = QPushButton("Optimize MP4")
        self.create_mp4_btn.clicked.connect(self.create_mp4)
        self.create_mp4_btn.setEnabled(False)
        buttons_layout.addWidget(self.create_mp4_btn)
        
        # Output result
        self.output_result = QLabel("No outputs generated yet")
        
        # Add widgets to section
        self.output_section.add_widget(folder_widget)
        self.output_section.add_widget(gif_options)
        self.output_section.add_widget(buttons_widget)
        self.output_section.add_widget(self.output_result)
        
        # Add to pipeline
        self.scroll_layout.addWidget(self.output_section)

    # Helper methods
    def log_message(self, message):
        """Add message to log output"""
        self.log_output.append(message)
        # Ensure visible
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )
        print(message)  # Also print to console
    
    def toggle_progress(self, show):
        """Show or hide progress bar"""
        if show:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
    
    def browse_file(self, input_widget, title, file_filter):
        """Browse for file and update input widget"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, str(project_root), file_filter
        )
        if file_path:
            input_widget.setText(file_path)
            
            # Special handling for 3D model files
            if "3D Model" in title:
                self.generated_3d_model_path = file_path
                self.model_path_input.setText(file_path)
                self.run_blender_btn.setEnabled(True)
    
    def toggle_refine_mode(self, state):
        """Toggle between direct prompt and AI refinement mode"""
        if state == Qt.Checked:
            self.generate_prompt_btn.setEnabled(True)
            self.generate_images_btn.setEnabled(False)
        else:
            self.generate_prompt_btn.setEnabled(False)
            self.generate_images_btn.setEnabled(True)
            # Show the prompt result field
            self.prompt_result.setVisible(True)
    
    def scroll_to_section(self, section):
        """Scroll the view to show a specific section"""
        self.scroll_area.ensureWidgetVisible(section)
        
    def update_image_gallery(self, image_paths):
        """Update the image gallery with generated images"""
        # Clear existing images
        for i in reversed(range(self.image_gallery_layout.count())): 
            widget = self.image_gallery_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # Add new images
        for i, path in enumerate(image_paths):
            image_btn = ImageButton(i, path, self.image_gallery)
            image_btn.selected.connect(self.on_image_selected)
            self.image_gallery_layout.addWidget(image_btn)
        
        # Store paths and enable next step button
        self.generated_images = image_paths
        self.proceed_to_3d_btn.setVisible(True)
        
        # Set section status
        self.image_section.set_status("complete")
    
    def on_image_selected(self, index, path):
        """Handle image selection for 3D model generation"""
        # Deselect all other buttons
        for i in range(self.image_gallery_layout.count()):
            btn = self.image_gallery_layout.itemAt(i).widget()
            if isinstance(btn, ImageButton) and btn.index != index:
                btn.setChecked(False)
        
        # Set selected image
        self.selected_image_index = index
        
        # Update preview image
        try:
            pixmap = QPixmap(path)
            scaled_pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.selected_image_preview.setPixmap(scaled_pixmap)
        except Exception as e:
            self.log_message(f"Error displaying preview: {e}")
            
        # Enable proceed button
        self.proceed_to_3d_btn.setEnabled(True)
        self.log_message(f"Selected image {index+1} for 3D model generation")
    
    def generate_prompt(self):
        """Generate enhanced prompt using AI"""
        # Get user input
        concept_text = self.concept_input.toPlainText().strip()
        if not concept_text:
            QMessageBox.warning(self, "Missing Input", "Please enter a concept description first.")
            return
        
        # Initialize OpenAI client if needed
        if not self.openai_client and OpenAIResponsesAPI:
            try:
                self.openai_client = OpenAIResponsesAPI()
                self.log_message("OpenAI API initialized successfully")
            except Exception as e:
                self.log_message(f"Error initializing OpenAI API: {e}")
                QMessageBox.critical(self, "API Error", f"Failed to initialize OpenAI API: {e}")
                return
        
        # Show progress
        self.toggle_progress(True)
        self.concept_section.set_status("active")
        self.log_message("Generating enhanced prompt...")
        
        # Create worker thread to avoid freezing UI
        thread = threading.Thread(
            target=self._generate_prompt_worker,
            args=(concept_text,),
            daemon=True
        )
        thread.start()
    
    def _generate_prompt_worker(self, concept_text):
        """Worker thread for prompt generation"""
        enhanced_prompt = ""
        try:
            # Call API to enhance prompt
            if self.openai_client:
                system_prompt = """You are an expert at creating detailed, visually rich text prompts for image generation.
                Convert the user's concept into a detailed prompt that will create a good reference image for 3D modeling.
                Focus on a single clear subject with details about perspective, lighting, and style.
                Keep the prompt under 200 characters if possible."""
                
                response = self.openai_client.generate_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": concept_text}
                    ]
                )
                
                enhanced_prompt = response.strip()
            else:
                # Fallback if API not available
                enhanced_prompt = concept_text + " (high quality, detailed, centered, 3D reference)"
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self, 
                QMessageBox.critical(self, "API Error", f"Failed to generate enhanced prompt: {e}")
            )
            enhanced_prompt = concept_text
        finally:
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._prompt_generation_done(enhanced_prompt),
                    'type': lambda: None
                })()
            )
    
    def _prompt_generation_done(self, enhanced_prompt):
        """Handle completion of prompt generation"""
        # Update UI
        self.prompt_result.setPlainText(enhanced_prompt)
        self.prompt_result.setVisible(True)
        self.generate_images_btn.setEnabled(True)
        self.toggle_progress(False)
        self.concept_section.set_status("complete")
        self.log_message("Enhanced prompt generated successfully")
    
    def generate_images(self):
        """Generate images from prompt using Replicate API"""
        # Get prompt text (either enhanced or direct)
        if self.refine_checkbox.isChecked():
            prompt_text = self.prompt_result.toPlainText().strip()
            if not prompt_text:
                QMessageBox.warning(self, "Missing Input", "Please generate an enhanced prompt first.")
                return
        else:
            prompt_text = self.concept_input.toPlainText().strip()
            if not prompt_text:
                QMessageBox.warning(self, "Missing Input", "Please enter a concept description first.")
                return
        
        # Initialize Replicate client if needed
        if not self.replicate_client and ReplicateAPI:
            try:
                self.replicate_client = ReplicateAPI()
                self.log_message("Replicate API initialized successfully")
            except Exception as e:
                self.log_message(f"Error initializing Replicate API: {e}")
                QMessageBox.critical(self, "API Error", f"Failed to initialize Replicate API: {e}")
                return
        
        # Show progress
        self.toggle_progress(True)
        self.image_section.set_enabled(True)
        self.image_section.set_status("active")
        self.log_message(f"Generating images from prompt: {prompt_text[:50]}...")
        
        # Create worker thread
        thread = threading.Thread(
            target=self._generate_images_worker,
            args=(prompt_text,),
            daemon=True
        )
        thread.start()
    
    def _generate_images_worker(self, prompt_text):
        """Worker thread for image generation"""
        try:
            # Set up output directory for this run
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_subdir = self.output_dir / f"images_{timestamp}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate images with Replicate API
            image_paths = []
            if self.replicate_client:
                # Use SDXL or another model as configured
                model_id = "stability-ai/sdxl:c221b2b8ef527988fb59bf24a8b97c4561f1c671f73bd389f866bfb27c061316"
                
                # Generate 4 images
                for i in range(4):
                    output_path = output_subdir / f"image_{i+1}.png"
                    image_url = self.replicate_client.generate_image(
                        prompt=prompt_text,
                        model_id=model_id,
                        negative_prompt="blurry, distorted, low quality, unrealistic, pixelated"
                    )
                    
                    # Download the image
                    if image_url:
                        self.replicate_client.download_image(image_url, str(output_path))
                        image_paths.append(str(output_path))
                        # Log progress
                        msg = f"Generated image {i+1}/4"
                        print(msg)
                        QApplication.instance().postEvent(
                            self,
                            type('Event', (object,), {
                                'callback': lambda msg=msg: self.log_message(msg),
                                'type': lambda: None
                            })()
                        )
            else:
                # Fallback with dummy images if API not available
                # In a real app, you'd show a proper error
                for i in range(4):
                    image_path = self.output_dir / f"dummy_image_{i+1}.png"
                    image_paths.append(str(image_path))
                    
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._image_generation_done(image_paths),
                    'type': lambda: None
                })()
            )
                
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._handle_image_generation_error(str(e)),
                    'type': lambda: None
                })()
            )
    
    def _image_generation_done(self, image_paths):
        """Handle completion of image generation"""
        if image_paths:
            self.update_image_gallery(image_paths)
            self.toggle_progress(False)
            self.log_message(f"Generated {len(image_paths)} images successfully")
            # Scroll to image section
            self.scroll_to_section(self.image_section)
        else:
            self._handle_image_generation_error("No images were generated")
    
    def _handle_image_generation_error(self, error_msg):
        """Handle image generation errors"""
        self.toggle_progress(False)
        self.image_section.set_status("error")
        self.log_message(f"Error generating images: {error_msg}")
        QMessageBox.critical(self, "Image Generation Error", f"Failed to generate images: {error_msg}")
    
    def proceed_to_3d_model(self):
        """Proceed to 3D model generation step"""
        if self.selected_image_index is None:
            QMessageBox.warning(self, "No Selection", "Please select an image first.")
            return
        
        # Enable 3D model section
        self.model_section.set_enabled(True)
        self.model_section.set_status("active")
        self.log_message("Proceeding to 3D model generation...")
        
        # Scroll to 3D model section
        self.scroll_to_section(self.model_section)
    
    def generate_3d_model(self):
        """Generate 3D model from selected image using Tripo API"""
        if self.selected_image_index is None or self.selected_image_index >= len(self.generated_images):
            QMessageBox.warning(self, "Invalid Selection", "Please select a valid image.")
            return
        
        # Get selected image path
        image_path = self.generated_images[self.selected_image_index]
        
        # Get quality settings
        quality = "detailed" if self.quality_detailed.isChecked() else "standard"
        face_limit = self.face_limit_input.text().strip()
        try:
            face_limit = int(face_limit)
            if face_limit <= 0 or face_limit > 500000:
                raise ValueError("Face limit must be between 1 and 500,000")
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Setting", f"Invalid face limit: {e}")
            return
        
        # Initialize Tripo client if needed
        if not self.tripo_client and TripoAPI:
            try:
                self.tripo_client = TripoAPI()
                self.log_message("Tripo API initialized successfully")
            except Exception as e:
                self.log_message(f"Error initializing Tripo API: {e}")
                QMessageBox.critical(self, "API Error", f"Failed to initialize Tripo API: {e}")
                return
        
        # Show progress
        self.toggle_progress(True)
        self.model_section.set_status("active")
        self.log_message(f"Generating 3D model from image (quality: {quality}, faces: {face_limit})...")
        
        # Create worker thread
        thread = threading.Thread(
            target=self._generate_3d_model_worker,
            args=(image_path, quality, face_limit),
            daemon=True
        )
        thread.start()
    
    def _generate_3d_model_worker(self, image_path, quality, face_limit):
        """Worker thread for 3D model generation"""
        try:
            # Set up output directory
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_subdir = self.output_dir / f"model_{timestamp}"
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            model_path = None
            if self.tripo_client:
                # Generate 3D model
                output_path = output_subdir / f"model_{timestamp}.glb"
                
                # Call API to generate model
                model_url = self.tripo_client.generate_3d_model(
                    image_path=image_path,
                    quality=quality,
                    face_limit=face_limit
                )
                
                # Download the model
                if model_url:
                    self.tripo_client.download_3d_model(model_url, str(output_path))
                    model_path = str(output_path)
                    
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._3d_model_generation_done(model_path),
                    'type': lambda: None
                })()
            )
                
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._handle_3d_model_generation_error(str(e)),
                    'type': lambda: None
                })()
            )
    
    def _3d_model_generation_done(self, model_path):
        """Handle completion of 3D model generation"""
        if model_path and Path(model_path).exists():
            self.generated_3d_model_path = model_path
            self.model_path_input.setText(model_path)
            self.model_result.setText(f"Model generated: {Path(model_path).name}")
            self.proceed_to_blender_btn.setEnabled(True)
            self.proceed_to_blender_btn.setVisible(True)
            self.toggle_progress(False)
            self.model_section.set_status("complete")
            self.log_message(f"3D model generated successfully: {Path(model_path).name}")
        else:
            self._handle_3d_model_generation_error("No model was generated or model file not found")
    
    def _handle_3d_model_generation_error(self, error_msg):
        """Handle 3D model generation errors"""
        self.toggle_progress(False)
        self.model_section.set_status("error")
        self.log_message(f"Error generating 3D model: {error_msg}")
        QMessageBox.critical(self, "Model Generation Error", f"Failed to generate 3D model: {error_msg}")
    
    def proceed_to_blender(self):
        """Proceed to Blender rendering step"""
        if not self.generated_3d_model_path or not Path(self.generated_3d_model_path).exists():
            QMessageBox.warning(self, "Missing Model", "No valid 3D model is available.")
            return
        
        # Enable Blender section
        self.blender_section.set_enabled(True)
        self.blender_section.set_status("active")
        self.log_message("Proceeding to Blender rendering...")
        
        # Set default material name
        if not self.material_input.text():
            self.material_input.setText("grey_procedural_MAT")
        
        # Scroll to Blender section
        self.scroll_to_section(self.blender_section)
    
    def start_blender_process(self):
        """Start Blender rendering process"""
        # Validate inputs
        template_path = self.template_input.text().strip()
        if not template_path or not Path(template_path).exists():
            QMessageBox.warning(self, "Missing Template", "Please select a valid Blender template file.")
            return
        
        material_name = self.material_input.text().strip()
        if not material_name:
            QMessageBox.warning(self, "Missing Material", "Please enter a material name.")
            return
        
        height_str = self.height_input.text().strip()
        try:
            height = float(height_str)
            if height <= 0:
                raise ValueError("Height must be positive")
        except ValueError:
            QMessageBox.warning(self, "Invalid Height", "Please enter a valid positive number for height.")
            return
        
        # Get options
        use_textures = self.texture_checkbox.isChecked()
        interactive_mode = self.interactive_checkbox.isChecked()
        
        # Validate model path
        if not self.generated_3d_model_path or not Path(self.generated_3d_model_path).exists():
            QMessageBox.warning(self, "Missing Model", "No valid 3D model is available.")
            return
        
        # Base output directory
        output_path = str(self.output_dir / "blender")
        os.makedirs(output_path, exist_ok=True)
        
        # Show progress
        self.toggle_progress(True)
        self.blender_section.set_status("active")
        self.log_message("Starting Blender rendering process...")
        
        # Create worker thread
        thread = threading.Thread(
            target=self._blender_process_worker,
            args=(template_path, self.generated_3d_model_path, material_name, 
                  use_textures, output_path, height, interactive_mode),
            daemon=True
        )
        thread.start()
    
    def _blender_process_worker(self, template_path, asset_path, material_name, 
                               use_textures, output_path, height, interactive_mode):
        """Worker thread for Blender rendering process"""
        try:
            # Initialize BlenderGen
            blender_gen = BlenderGen()
            
            # Run Blender pipeline
            project_folder = blender_gen.blenderGen_pipeline(
                template_path=template_path,
                asset_path=asset_path,
                mtl_name=material_name,
                texture_bool=use_textures,
                output_path=output_path,
                height=height,
                interactive=interactive_mode
            )
            
            # Store project folder for output processing
            self.project_folder = project_folder
            self.renders_folder = project_folder / "renders"
            
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._blender_process_done(project_folder),
                    'type': lambda: None
                })()
            )
                
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._handle_blender_process_error(str(e)),
                    'type': lambda: None
                })()
            )
    
    def _blender_process_done(self, project_folder):
        """Handle completion of Blender rendering process"""
        if project_folder and Path(project_folder).exists():
            self.toggle_progress(False)
            self.blender_section.set_status("complete")
            self.render_folder_display.setText(str(project_folder / "renders"))
            
            # Enable output section
            self.output_section.set_enabled(True)
            self.output_section.set_status("active")
            self.create_gif_btn.setEnabled(True)
            self.create_mp4_btn.setEnabled(True)
            
            self.log_message(f"Blender rendering completed successfully in {project_folder}")
            
            # Scroll to output section
            self.scroll_to_section(self.output_section)
        else:
            self._handle_blender_process_error("Project folder not created or not found")
    
    def _handle_blender_process_error(self, error_msg):
        """Handle Blender process errors"""
        self.toggle_progress(False)
        self.blender_section.set_status("error")
        self.log_message(f"Error in Blender process: {error_msg}")
        QMessageBox.critical(self, "Blender Error", f"Failed to complete Blender rendering: {error_msg}")
    
    def create_gif(self):
        """Create GIF from rendered images"""
        if not self.renders_folder or not Path(self.renders_folder).exists():
            QMessageBox.warning(self, "Missing Renders", "No render folder available.")
            return
        
        # Get options
        duration_str = self.duration_input.text().strip()
        resize_str = self.resize_input.text().strip()
        
        try:
            duration = float(duration_str)
            if duration <= 0:
                raise ValueError("Duration must be positive")
                
            resize_percent = float(resize_str)
            if resize_percent <= 0 or resize_percent > 100:
                raise ValueError("Resize percentage must be between 1 and 100")
                
            # Convert to scale factor
            scale_factor = resize_percent / 100
            
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Options", f"Invalid GIF options: {e}")
            return
        
        # Show progress
        self.toggle_progress(True)
        self.output_section.set_status("active")
        self.log_message("Creating GIF from rendered images...")
        
        # Create worker thread
        thread = threading.Thread(
            target=self._create_gif_worker,
            args=(self.renders_folder, duration, scale_factor),
            daemon=True
        )
        thread.start()
    
    def _create_gif_worker(self, renders_folder, duration, scale_factor):
        """Worker thread for GIF creation"""
        try:
            # Find all rendered images
            render_files = sorted([f for f in Path(renders_folder).glob("*.png")])
            
            if not render_files:
                raise FileNotFoundError("No image files found in render folder")
            
            # Create GIF filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            gif_path = Path(renders_folder) / f"animation_{timestamp}.gif"
            
            # Create GIF using utility function
            if images_to_gif:
                images_to_gif(
                    input_folder=str(renders_folder),
                    output_path=str(gif_path),
                    pattern="*.png",
                    duration=duration,
                    loop=0,
                    optimize=True,
                    resize_factor=scale_factor
                )
            else:
                raise ImportError("GIF creation utility not available")
            
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._gif_creation_done(gif_path),
                    'type': lambda: None
                })()
            )
                
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._handle_output_processing_error("GIF", str(e)),
                    'type': lambda: None
                })()
            )
    
    def _gif_creation_done(self, gif_path):
        """Handle completion of GIF creation"""
        if gif_path and Path(gif_path).exists():
            self.toggle_progress(False)
            self.output_section.set_status("complete")
            self.output_result.setText(f"GIF created: {Path(gif_path).name}")
            self.log_message(f"GIF created successfully: {gif_path}")
        else:
            self._handle_output_processing_error("GIF", "GIF file not created or not found")
    
    def create_mp4(self):
        """Create optimized MP4 from rendered images"""
        if not self.renders_folder or not Path(self.renders_folder).exists():
            QMessageBox.warning(self, "Missing Renders", "No render folder available.")
            return
        
        # Check for FFMPEG
        try:
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
            if "ffmpeg version" not in result.stdout:
                raise FileNotFoundError("FFmpeg not found")
        except:
            QMessageBox.warning(self, "Missing Dependency", 
                               "FFmpeg is required for MP4 creation but was not found on your system.")
            return
        
        # Show progress
        self.toggle_progress(True)
        self.output_section.set_status("active")
        self.log_message("Creating optimized MP4 from rendered images...")
        
        # Create worker thread
        thread = threading.Thread(
            target=self._create_mp4_worker,
            args=(self.renders_folder,),
            daemon=True
        )
        thread.start()
    
    def _create_mp4_worker(self, renders_folder):
        """Worker thread for MP4 creation"""
        try:
            # Find all rendered images
            render_files = sorted([f for f in Path(renders_folder).glob("*.png")])
            
            if not render_files:
                raise FileNotFoundError("No image files found in render folder")
            
            # Create MP4 filename
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            mp4_path = Path(renders_folder) / f"animation_{timestamp}.mp4"
            
            # Create MP4 using FFmpeg
            import subprocess
            
            # Ensure images are sequentially numbered
            # If using current Blender output like "render_001.png", "render_002.png", etc.
            # then we can use a pattern like "render_%03d.png"
            
            # Determine image pattern
            pattern_match = None
            for f in render_files:
                import re
                match = re.search(r'(\D+)(\d+)\.png$', f.name)
                if match:
                    pattern_match = match
                    break
            
            if pattern_match:
                # Use pattern for FFmpeg
                prefix = pattern_match.group(1)
                digits = len(pattern_match.group(2))
                pattern = f"{prefix}%0{digits}d.png"
                
                cmd = [
                    "ffmpeg", "-y",
                    "-framerate", "30",
                    "-i", str(Path(renders_folder) / pattern),
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    str(mp4_path)
                ]
                
                subprocess.run(cmd, check=True)
            else:
                # Alternative: create a file list
                list_file = Path(renders_folder) / "files.txt"
                with open(list_file, "w") as f:
                    for image in render_files:
                        f.write(f"file '{image.name}'\n")
                        f.write(f"duration 0.033\n")  # ~30fps
                
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(list_file),
                    "-c:v", "libx264",
                    "-preset", "slow",
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",
                    str(mp4_path)
                ]
                
                subprocess.run(cmd, check=True)
                
                # Clean up list file
                list_file.unlink()
            
            # Update UI in main thread
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._mp4_creation_done(mp4_path),
                    'type': lambda: None
                })()
            )
                
        except Exception as e:
            # Handle error
            QApplication.instance().postEvent(
                self,
                type('Event', (object,), {
                    'callback': lambda: self._handle_output_processing_error("MP4", str(e)),
                    'type': lambda: None
                })()
            )
    
    def _mp4_creation_done(self, mp4_path):
        """Handle completion of MP4 creation"""
        if mp4_path and Path(mp4_path).exists():
            self.toggle_progress(False)
            self.output_section.set_status("complete")
            self.output_result.setText(f"MP4 created: {Path(mp4_path).name}")
            self.log_message(f"MP4 created successfully: {mp4_path}")
        else:
            self._handle_output_processing_error("MP4", "MP4 file not created or not found")
    
    def _handle_output_processing_error(self, output_type, error_msg):
        """Handle output processing errors"""
        self.toggle_progress(False)
        self.output_section.set_status("error")
        self.log_message(f"Error creating {output_type}: {error_msg}")
        QMessageBox.critical(self, f"{output_type} Creation Error", 
                            f"Failed to create {output_type}: {error_msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set stylesheet
    app.setStyleSheet(STYLESHEET)
    
    # Create and show main window
    main_window = ComprehensivePipelineUI()
    main_window.show()
    
    sys.exit(app.exec())