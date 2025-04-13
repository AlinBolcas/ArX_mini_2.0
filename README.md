# ArX Mini 2.0

> A lightweight, modular AI framework for creative content generation and workflow automation.

![ArX Mini Demo](assets/media/test_output.gif)

## Core Features

- **Multi-Modal Generation**: Create images, text, video, and 3D assets using various AI providers
- **Provider Flexibility**: Support for OpenAI (primary), Replicate (multi-modal), and Ollama (local LLMs)
- **Cognitive Architecture**: Built-in memory systems, RAG capabilities, and tool integration
- **Agentic Design**: Framework for building autonomous AI agents for creative tasks
- **Creative Pipelines**: Specialized workflows for content creation and automation

## Getting Started

### Image Generation with OpenAI

```python
# src/I_integrations/openai_API.py
from src.I_integrations.openai_API import OpenAIWrapper

# Initialize the API wrapper
openai_api = OpenAIWrapper(model="gpt-4o") # Ensure API keys are configured

# Generate an image
image_response = openai_api.generate_image(
    prompt="A futuristic city with flying cars and neon lights",
    size="1024x1024",
    style="vivid"
)

# Save the image
# Note: Ensure the output directory exists or adjust the path
output_path = "data/output/images/future_city.png"
openai_api.save_image(image_response, output_path)
print(f"Image saved to {output_path}")
```

### Local LLM with Ollama

```python
# src/I_integrations/ollama_API.py
from src.I_integrations.ollama_API import OllamaWrapper

# Initialize Ollama (ensure Ollama is running locally)
ollama_api = OllamaWrapper(model="llama3")

# Generate text with a local model
response = ollama_api.generate_text(
    prompt="Explain how generative AI can be used in creative workflows",
    max_tokens=500
)

print(response)
```

### Multi-modal Generation with Replicate

```python
# src/I_integrations/replicate_API.py
from src.I_integrations.replicate_API import ReplicateAPI

# Initialize the Replicate API wrapper
replicate = ReplicateAPI()

# Generate an image with Stable Diffusion
image_url = replicate.generate_image(
    prompt="Surreal landscape with floating islands and bioluminescent plants",
    model="flux-dev",  # Options: "flux-schnell", "flux-pro", "flux-pro-ultra", etc.
    aspect_ratio="16:9"
)

# Download the generated image
image_path = replicate.download_file(
    image_url, 
    output_dir="images",
    filename="surreal_landscape.png"
)

# Generate video from the image
video_url = replicate.generate_video(
    prompt="Camera smoothly panning across a surreal landscape with floating islands",
    model="wan-i2v-480p",  # Image-to-video model
    image_url=image_url,
    aspect_ratio="16:9"
)

# Download the generated video
video_path = replicate.download_file(
    video_url,
    output_dir="videos",
    filename="surreal_landscape_animation.mp4"
)

print(f"Image saved to: {image_path}")
print(f"Video saved to: {video_path}")
```

### Text Generation with Memory and RAG

```python
# src/II_textGen/textGen.py
from src.II_textGen.textGen import TextGen

# Initialize text generation with provider, model, and memory settings
text_gen = TextGen(
    provider="openai",  # Options: "openai" or "ollama"
    default_model="gpt-4o-mini",  # Will use provider-appropriate default if not specified
    short_term_limit=8000  # Token limit for conversation memory
)

# Add RAG to enhance responses with external knowledge
# The TextGen instance already has RAG capabilities built in
context_doc = """
Architectural visualization is the art of creating rendered images of architectural designs
before they are built. Modern approaches include photorealistic rendering, VR experiences,
and interactive walkthroughs using game engines like Unreal and Unity.
"""

# Generate contextually-aware responses
response = text_gen.chat_completion(
    user_prompt="Suggest creative ways to use AI in architectural visualization",
    system_prompt="You are an expert in architectural visualization and AI integration.",
    context=context_doc,  # External knowledge is automatically processed
    tool_names=["get_current_datetime", "search_web"],  # Optional tools
    temperature=0.7,
    max_tokens=500
)

print(response)

# Follow-up keeps context through memory system
follow_up = text_gen.chat_completion(
    user_prompt="How would these approaches impact client presentations?",
    # No need to provide system_prompt or context again - memory keeps track
)

print(follow_up)
```

### Multi-Agent System

```python
# src/III_agents/agentsGen.py
from src.III_agents.agentsGen import AgentGen

# Initialize the multi-agent system with provider and model settings
agent_gen = AgentGen(
    provider="openai",  # Can also use "ollama" for local models
    default_model="gpt-4o-mini",
    short_term_limit=8000  # Memory token limit for agents
)

# Create specialized agents with different capabilities
agent_gen.create_agent(
    name="Creative",
    system_prompt="You are a highly creative AI specializing in novel ideas and artistic concepts. Be imaginative and original.",
    temperature=0.8,
    max_tokens=300,
    description="Generates creative concepts and ideas",
    log_color="magenta"  # Visual distinction in logs
)

agent_gen.create_agent(
    name="Technical",
    system_prompt="You are a technical AI expert focusing on implementation details and practical solutions. Be precise and detailed.",
    temperature=0.2,
    max_tokens=500,
    tool_names=["get_current_datetime", "search_web"],  # Give this agent specific tools
    description="Provides technical solutions and implementation details",
    log_color="blue"
)

agent_gen.create_agent(
    name="Critic",
    system_prompt="You are an analytical critic who evaluates ideas objectively. Identify strengths and weaknesses concisely.",
    max_tokens=400,
    context="Evaluation criteria: originality, feasibility, market potential, technical complexity",
    description="Evaluates and refines concepts",
    log_color="yellow"
)

# Use the ReAct pattern (Reasoning-Acting) with a specific agent
project_brief = "Design an interactive installation using AI and projection mapping"
creative_response = agent_gen.loop_react(
    user_prompt=project_brief,
    agent_name="Creative",  # Use the Creative agent for this task
    max_depth=3,  # Maximum number of reasoning cycles
    verbose=True  # Show intermediate steps
)
print(f"Creative Agent Response:\n{creative_response}")

# Automatically determine which agent is best suited for a task
question = "What are the technical requirements for implementing a real-time computer vision system?"

# Get all available agents for triage
agent_configs = [
    {"name": "Creative", "description": "Generates creative concepts and ideas"},
    {"name": "Technical", "description": "Provides technical solutions and implementation details"},
    {"name": "Critic", "description": "Evaluates and refines concepts"}
]

# Let the system determine which agent is best suited
selected_agents = agent_gen.triage_agent(
    user_prompt=question,
    handoff_agents=agent_configs
)

# Get response from the selected agent(s)
if selected_agents:
    for agent_name in selected_agents:
        response = agent_gen.loop_simple(
            user_prompt=question,
            agent_name=agent_name,
            max_depth=2
        )
        print(f"\n{agent_name} Agent Response:\n{response}")
else:
    print("No suitable agent found for this question")
```

## Requirements

- Python 3.8+
- FFmpeg (Required for video utilities)
- Optional: Ollama for local LLM deployment
- `requirements.txt` dependencies

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ArX_mini_2.0.git
cd ArX_mini_2.0

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install FFmpeg (if not already installed)
#    macOS: brew install ffmpeg
#    Ubuntu/Debian: sudo apt-get update && sudo apt-get install ffmpeg
#    Windows: Download from ffmpeg.org and add to system PATH

# 5. Optional: Install Ollama for local LLM support
#    Follow instructions at: https://ollama.ai/

# 6. Configure API Keys (Important!)
#    - Create a .env file in the root directory
#    - Add your API keys:
#      OPENAI_API_KEY='your_openai_key'
#      REPLICATE_API_TOKEN='your_replicate_token'
#      # Add other keys as needed (NewsAPI, WeatherAPI, etc.)
```

## Key AI Capabilities

### Integrations (Module `I_integrations`)
- **OpenAI**: Complete API integration (GPT models, DALL-E, TTS, embeddings, vision analysis)
- **Alternative Providers**: Replicate for multi-modal generation, Ollama for local LLM deployment
- **Data & Search**: Google API, News API, OpenWeather API, Hunter API (Lead Generation)
- **3D Content**: Meshy API, Tripo API for 3D asset generation
- **Visualization**: GraphViz for knowledge graph creation
- **Web Interaction**: Web Crawling for data collection
- **System Interaction**: Local System API for file operations

### Advanced Text Generation (Module `II_textGen`)
- **Memory Management**: Long and short-term memory for persistent conversations
- **RAG Implementation**: Knowledge retrieval and augmentation for grounded responses
- **Tool Integration**: Extensible framework for adding specialized functions to AI models
- **Multi-provider Support**: Unified interface across OpenAI, Replicate, and Ollama

### Agentic Systems (Module `III_agents`)
- **Agent Framework**: Tools to build autonomous AI agents for creative tasks
- **Multi-agent Coordination**: Systems for orchestrating multiple specialized agents
- **OpenAI Assistant SDK**: Integration with OpenAI's Assistants API
- **Legacy Examples**: Reference implementations for various agent architectures

### Creative Pipelines (Module `IV_pipelines`)
- **Blender Integration**: Automated 3D asset generation and rendering with Blender
- **Daily Content**: Automated routine content creation with scheduling
- **Factory Pattern**: Template-based content generation for consistent output
- **Asset Generation**: Tools for creating various creative assets programmatically

### User Interfaces (Module `V_ui`)
- **Web UI**: Flask-based interface components
- **Desktop**: Tkinter implementations
- **Data Apps**: Streamlit dashboard examples
- **Framework**: Reusable UI components and patterns

### Core Utilities (Module `VI_utils`)
- **Media Processing**: Image and video manipulation utilities
- **File System**: Helpers for file and directory operations
- **General Utilities**: Common helper functions for AI projects

## License

MIT License

---

Built for creative AI projects.