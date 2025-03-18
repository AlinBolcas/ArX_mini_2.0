# ArX Mini 2.0

> A lightweight, modular framework for generative AI projects with minimal overhead.

![ArX Mini Demo](assets/media/test_output.gif)

## Key AI Capabilities

### Integrations
- **OpenAI**: Complete API wrapper with chat completion, vision analysis, image generation, embeddings, and TTS
- **Replicate & Ollama**: Alternative AI model providers for flexibility
- **Social Media**: Twitter/X, Instagram, and Meta posting capabilities
- **Data Sources**: News API, Weather API, Web Crawling for rich context
- **3D Content**: Meshy API integration for 3D model generation

### Smart Pipelines
- **Social Media Manager**: Automated content creation and posting across platforms
- **Content Generation**: Daily media creation with automatic formatting and branding
- **CV/Resume Generation**: Automated professional document creation
- **Fine-tuning Workflows**: OpenAI model fine-tuning pipeline
- **Personalized Content**: Generate customized content based on user personas

### Advanced Text Generation
- **Memory Management**: Long and short-term memory for contextual responses
- **RAG Implementation**: Retrieval-augmented generation for knowledge-based responses
- **Knowledge Graph**: Structured information representation and querying
- **Tool Integration**: Extensible function calling framework

## Getting Started

### Image Generation with OpenAI

```python
from src.I_integrations.openai_API import OpenAIAPI

# Initialize the API wrapper
openai_api = OpenAIAPI(model="gpt-4o")

# Generate an image
image_response = openai_api.generate_image(
    prompt="A futuristic city with flying cars and neon lights",
    size="1024x1024",
    style="vivid"
)

# Save the image
openai_api.save_image(image_response, "future_city.png")
```

### Video Processing

```python
from src.VI_utils.video_utils import video_loop, video_to_gif

# Create a seamless looping video
video_loop(
    input_path="raw_footage.mp4",
    output_path="seamless_loop.mp4",
    fade_duration=0.5
)

# Convert video to optimized GIF
video_to_gif(
    input_path="seamless_loop.mp4",
    output_path="animation.gif",
    resize_factor=0.5,
    fps=15
)
```

### Text Generation with Memory

```python
from src.II_textGen.textGen import TextGen

# Initialize text generation with memory
text_gen = TextGen(
    model="gpt-4o-mini",
    memory_enabled=True
)

# Generate text with context
response = text_gen.generate(
    prompt="What are three creative applications of generative AI?",
    context={"project_type": "interactive art installation"}
)

print(response)

# Follow-up keeps previous context in memory
follow_up = text_gen.generate(
    prompt="Elaborate on the first idea"
)

print(follow_up)
```

### Social Media Content Pipeline

```python
from src.IV_pipelines.socialMedia.socialManager import SocialMediaManager

# Initialize the social media manager
sm_manager = SocialMediaManager(platforms=["twitter", "instagram"])

# Generate and post content
sm_manager.create_and_post(
    topic="AI art trends",
    media_type="image",
    hashtags=["#AIart", "#GenerativeAI", "#DigitalCreation"]
)
```

## Requirements

- Python 3.8+
- FFmpeg
- OpenCV
- PIL/Pillow
- See `requirements.txt` for full list

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ArX_mini_2.0.git
cd ArX_mini_2.0

# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install FFmpeg if needed
# macOS: brew install ffmpeg
# Ubuntu/Debian: apt-get install ffmpeg
# Windows: Download from ffmpeg.org
```

## Project Structure

```
ArX_mini_2.0/
├── src/
│   ├── I_integrations/     # API integrations (OpenAI, Replicate, Social, etc.)
│   ├── II_textGen/         # Advanced text generation with memory and RAG
│   ├── III_agents/         # Autonomous agent frameworks
│   ├── IV_pipelines/       # Content generation pipelines
│   ├── V_ui/               # User interfaces
│   └── VI_utils/           # Core utilities
├── data/                   # Data storage
└── requirements.txt        # Dependencies
```

## License

MIT License

---

Built for creative AI projects. 