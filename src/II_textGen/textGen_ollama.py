"""
textGen.py

Unified TextGen interface integrating OpenAI API, memory management, 
retrieval augmented generation (RAG), and tools functionality.

This module serves as the central integration point for AI text generation,
providing a simplified interface for accessing all functionality with proper context
and memory management.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import subprocess

# Configure logging - set to WARNING to reduce noise
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only show our module's logs
logger.setLevel(logging.INFO)

# Add paths for importing other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "I_integrations")))

# Import required modules
try:
    # Try both import methods silently
    try:
        from I_integrations.openai_API import OpenAIAPI
        from II_textGen.tools import Tools
        from II_textGen.memory import Memory
        from II_textGen.rag import RAG
    except ImportError:
        from src.I_integrations.openai_API import OpenAIAPI
        from src.II_textGen.tools import Tools
        from src.II_textGen.memory import Memory
        from src.II_textGen.rag import RAG
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure all required modules are available in their respective directories.")
    raise


class TextGen:
    """
    Unified TextGen: Integration Hub for OpenAI API, Memory, RAG, and Tools.

    Provides a simplified interface for:
    - Context-aware LLM completions and chat
    - Short and long-term memory retrieval
    - Tool integration with LLM calls
    - Vision and multimodal capabilities
    - Structured output parsing
    """
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None, 
                 replicate_api_token: Optional[str] = None, 
                 default_model: str = "gpt-4o",
                 short_term_limit: int = 8000, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Initialize TextGen with all required components.
        
        Args:
            openai_api_key: Optional OpenAI API key
            replicate_api_token: Optional Replicate API token
            default_model: Default model to use
            short_term_limit: Token limit for short-term memory
            chunk_size: Size of chunks for RAG
            chunk_overlap: Overlap between chunks for RAG
        """
        # Initialize tools with API keys
        self.tools = Tools(
            openai_api_key=openai_api_key,
            replicate_api_token=replicate_api_token
        )
        
        # Initialize OpenAI directly with the key
        self.openai = OpenAIAPI(
            model=default_model,
            api_key=openai_api_key
        )
        
        # Store default model
        self.default_model = default_model
        
        # Initialize memory and RAG
        self.memory = Memory(short_term_limit)
        self.rag = RAG(chunk_size, chunk_overlap)
        self.embedding = lambda text: self.openai.create_embeddings(text)
        
        # Register available tools
        self._register_tools()
        
    def _register_tools(self):
        """Register available tools from the Tools class."""
        self.available_tools = {}
        
        if not hasattr(self, "tools") or not self.tools:
            logger.warning("Tools not available for registration")
            return
            
        for attr_name in dir(self.tools):
            if not attr_name.startswith("_") and callable(getattr(self.tools, attr_name)):
                tool_func = getattr(self.tools, attr_name)
                self.available_tools[attr_name] = tool_func
                
    def _prepare_prompts(self, user_prompt: str, system_context: str = None, context: str = None) -> tuple:
        """
        Prepare final system and user messages with appropriate context.
        
        Args:
            user_prompt: The main user prompt
            system_context: Additional context to add to system message
            context: Additional context to add to user message
            
        Returns:
            Tuple of (final_system, final_user) with all context included
        """
        # Retrieve long-term context
        long_term_context = self.retrieve_memory_context(user_prompt)
            
        # Build final system message
        final_system = ""
        if system_context:
            final_system += system_context + "\n"
        if long_term_context:
            final_system += "Long term context:\n" + long_term_context
            
        # Build final user message
        final_user = ""
        if context:
            final_user += context + "\n"
        final_user += user_prompt
        
        return final_system, final_user
        
    def retrieve_memory_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context from long-term memory.
        
        Args:
            query: The query to search for relevant context
            top_k: Number of top results to include
            
        Returns:
            String containing relevant context from memory
        """
        try:
            # Get insights from long-term memory
            insights = self.memory.retrieve_long_term()
            if not insights or len(insights) == 0:
                return ""
                
            # Process through RAG
            document = "\n".join([str(insight) for insight in insights])
            self.rag.ingest_documents(document, self.embedding)
            
            # Get context from RAG
            context = self.rag.retrieve_context(query, self.embedding, top_k=top_k)
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving memory context: {e}")
            return ""
            
    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Get a list of all available tools and their descriptions.
        
        Returns:
            List of dictionaries with tool names and descriptions
        """
        tool_list = []
        
        for tool_name, tool_func in self.available_tools.items():
            description = tool_func.__doc__.strip() if tool_func.__doc__ else "No description available"
            tool_list.append({
                "name": tool_name,
                "description": description
            })
            
        return tool_list
        
    def chat_completion(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        system_context: str = None,
        context: str = None,
        tool_names: List[str] = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """
        Generate a chat completion with context and tool integration.
        
        Args:
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            tool_names: List of tool names to make available
            temperature: Generation temperature (higher = more random)
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Generated response text
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
        
        # Prepare tools if specified
        tools = None
        if tool_names and self.tools:
            tools = []
            for tool_name in tool_names:
                if tool_name in self.available_tools:
                    tool_func = self.available_tools[tool_name]
                    # Convert tool function to schema
                    tool_schema = self._convert_function_to_schema(tool_func)
                    tools.append(tool_schema)
        
        # Make API call with tools
        response = self.openai.chat_completion(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            available_tools=self.available_tools if tools else None,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, response)
            
        return response

    def _convert_function_to_schema(self, func) -> Dict:
        """Convert a Python function to OpenAI function schema format."""
        import inspect
        from typing import get_type_hints
        
        # Get function signature and docstring
        sig = inspect.signature(func)
        doc = func.__doc__ or ""
        type_hints = get_type_hints(func)
        
        # Create parameter schema
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        # Map Python types to JSON schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        
        # Process each parameter
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # Get parameter type
            param_type = type_hints.get(param_name, str)
            json_type = type_map.get(param_type, "string")
            
            # Get parameter description from docstring
            param_desc = ""
            if doc:
                for line in doc.split('\n'):
                    if f":param {param_name}:" in line:
                        param_desc = line.split(f":param {param_name}:")[1].strip()
                        break
            
            # Add parameter to schema
            parameters["properties"][param_name] = {
                "type": json_type,
                "description": param_desc
            }
            
            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                parameters["required"].append(param_name)
        
        # Create function schema
        schema = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": doc.split('\n')[0] if doc else "",
                "parameters": parameters
            }
        }
        
        return schema
        
    def structured_output(
        self,
        user_prompt: str,
        system_prompt: str = "Return the output in structured JSON format.",
        system_context: str = None,
        context: str = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> Any:
        """
        Generate structured JSON output from the LLM.
        
        Args:
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Parsed JSON response
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
            
        # Get structured output
        response = self.openai.structured_output(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, str(response))
            
        return response
    
    def vision_analysis(
        self,
        image_url: str,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant with image analysis capabilities.",
        system_context: str = None,
        context: str = None,
        temperature: float = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """
        Analyze an image with vision capabilities.
        
        Args:
            image_url: URL of the image to analyze
            user_prompt: The main user prompt
            system_prompt: Base system prompt for the LLM
            system_context: Additional context for system message
            context: Additional context for user message
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            store_interaction: Whether to store the interaction in memory
            
        Returns:
            Generated response text
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
            
        # Use vision_analysis instead of vision_completion
        response = self.openai.vision_analysis(
            image_path=image_url,
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, response)
            
        return response

    def reasoned_completion(
        self,
        user_prompt: str,
        reasoning_effort: str = "low",
        context: str = None,
        max_tokens: int = None,
        store_interaction: bool = True,
        **kwargs
    ) -> str:
        """Generate a reasoned completion."""
        # Get message history - filter out system messages for o1-mini
        message_history = self.memory.retrieve_short_term_formatted()
        if message_history:
            message_history = [msg for msg in message_history if msg.get("role") != "system"]
        
        # Prepare context-enriched user prompt (don't use system_context)
        final_user = user_prompt
        if context:
            final_user = context + "\n\n" + user_prompt
        
        # Log what we're doing
        print(f"🧠 Using reasoning model with context-enriched prompt")
        
        # Remove temperature and system_prompt for o1-mini model
        response = self.openai.reasoned_completion(
            user_prompt=final_user,
            message_history=message_history,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            **{k:v for k,v in kwargs.items() if k not in ['temperature', 'system_prompt']}
        )
        
        # Store the interaction if requested
        if store_interaction:
            self.memory.save_short_term("Reasoned Completion", user_prompt, response)
        
        return response
        
    # Memory delegation methods
    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")
        
    def save_history(self, filename: Optional[str] = None) -> str:
        """
        Save conversation history.
        
        Args:
            filename: Optional filename (ignored in current implementation)
            
        Returns:
            Path to the memory directory
        """
        # With the updated memory implementation, we don't need a separate filename
        # since it now just syncs the existing memory files
        memory_dir = self.memory.save()
        logger.info(f"Memory saved to directory: {memory_dir}")
        return memory_dir
        
    def load_history(self, filename: str) -> bool:
        """
        Load conversation history from a file.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure we have the correct JSON file format
            if not filename.lower().endswith('.json'):
                filename += '.json'
            
            # Use memory's load method (which now expects a specific format)
            self.memory.load(filename)
            logger.info(f"Loaded memory from: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False


# Example usage focused on demonstrating TextGen functionality
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 TEXTGEN FUNCTIONALITY DEMONSTRATION 🚀")
    print("="*60 + "\n")
    
    text_gen = TextGen()
    
    # Function to prompt for input with default value
    def prompt(message, default=None):
        result = input(f"{message} [{default}]: ") if default else input(f"{message}: ")
        return result if result.strip() else default
    
    # Menu-driven test approach
    while True:
        print("\nAvailable Tests:")
        print("1. Basic Chat Completion")
        print("2. Multi-Tool Completion")
        print("3. Structured Output")
        print("4. Reasoned Completion")
        print("5. Vision Analysis")
        print("6. Memory & History Management")
        print("0. Exit")
        
        choice = prompt("Select a test to run", "0")
        
        if choice == "0":
            print("Exiting test suite.")
            break
            
        elif choice == "1":
            # Demo 1: Basic Chat Completion
            print("\n" + "="*50)
            print("💬 TEST 1: BASIC CHAT COMPLETION")
            print("="*50)
            custom_prompt = prompt("Enter your prompt (or use default)", "Explain the concept of API integration in one short paragraph.")
            response = text_gen.chat_completion(
                user_prompt=custom_prompt,
                system_prompt="You are a concise technical writer. Keep all responses under 150 tokens in exactly one paragraph.",
                max_tokens=150
            )
            print("\n" + "-"*50)
            print("📋 RESPONSE:")
            print("-"*50)
            print(f"\n{response}\n")
            
        elif choice == "2":
            # Demo 2: Multi-Tool Completion
            print("\n" + "="*50)
            print("🧰 TEST 2: MULTI-TOOL COMPLETION")
            print("="*50)
            
            # Get a topic to process with multiple tools
            topic = prompt("Enter a topic to research and visualize", "climate change impacts")
            
            # List available tools for reference
            print("\nAvailable tools that might be used:")
            print("- web_crawl: Search the web for information")
            print("- generate_image: Create an image based on a description")
            print("- get_datetime: Get current date and time")
            print("- get_weather: Get current weather for a location")
            print("- get_forecast: Get weather forecast for a location")
            print("- text_to_speech: Convert text to spoken audio")
            
            # Create a system message that encourages multi-tool use
            system_message = f"""You are a research assistant with access to multiple tools.
            
For this task, use any tools that would help provide a comprehensive response about '{topic}'.
Consider searching for current information, generating relevant images, checking date/time 
relevance, or any other tools that would enhance your response.

Important: First decide which tools would be helpful, then use them in sequence.
After gathering information from tools, synthesize everything into a helpful response.

When using text_to_speech, only use one of these voices: alloy, echo, fable, onyx, nova, shimmer, ash, sage, coral.

Be resourceful and creative with the tools available to you."""

            # Run the completion with multiple tools available
            try:
                response = text_gen.chat_completion(
                    user_prompt=f"Research '{topic}' thoroughly. Use multiple tools to gather information and create visual aids. Then provide a comprehensive summary of what you've learned.",
                    system_prompt=system_message,
                    tool_names=["web_crawl", "generate_image", "get_datetime", "get_weather", "get_forecast", "text_to_speech"],
                    system_context="You have access to multiple tools. Use them strategically to provide the best response.",
                    max_tokens=500
                )
                
                print("\n" + "-"*50)
                print("🔍 MULTI-TOOL RESEARCH RESULTS:")
                print("-"*50)
                print(f"\n{response}\n")
            except Exception as e:
                print(f"\n❌ Error during multi-tool completion: {e}")
                response = "Error occurred during processing."
            
            # Check if the model generated any images and display them
            if "output/images" in response:
                potential_image_paths = [line.strip() for line in response.split('\n') if "output/images" in line]
                for path in potential_image_paths:
                    # Extract path by finding text that contains 'output/images'
                    import re
                    image_path_match = re.search(r'(output\/images\/[^\s:,\'"]+)', path)
                    if image_path_match:
                        image_path = image_path_match.group(1)
                        if os.path.exists(image_path):
                            print(f"\nOpening generated image: {image_path}")
                            try:
                                if sys.platform == "darwin":  # macOS
                                    subprocess.run(["qlmanage", "-p", image_path], 
                                                  stdout=subprocess.DEVNULL, 
                                                  stderr=subprocess.DEVNULL)
                                elif sys.platform == "win32":  # Windows
                                    os.startfile(image_path)
                                else:  # Linux
                                    subprocess.run(["xdg-open", image_path])
                            except Exception as e:
                                print(f"Couldn't open image for preview: {e}")
            
        elif choice == "3":
            # Demo 3: Structured Output
            print("\n" + "="*50)
            print("💾 TEST 3: STRUCTURED OUTPUT")
            print("="*50)
            topic = prompt("Enter a topic for structured analysis", "email automation system")
            structured = text_gen.structured_output(
                user_prompt=f"Create a concise system overview for a {topic}. Include key components, inputs, outputs, and potential challenges.",
                system_prompt="You are a systems analyst. Provide structured, concise information.",
                max_tokens=200
            )
            print("\n" + "-"*50)
            print("📊 STRUCTURED OUTPUT:")
            print("-"*50)
            print(json.dumps(structured, indent=2))
            
        elif choice == "4":
            # Demo 4: Reasoned Completion
            print("\n" + "="*50)
            print("🧠 TEST 4: REASONED COMPLETION")
            print("="*50)
            try:
                reasoning_topic = prompt("Enter a topic that requires reasoning", "how to prioritize features in a software project")
                reasoned_response = text_gen.reasoned_completion(
                    user_prompt=f"Explain a systematic approach to {reasoning_topic} in one concise paragraph.",
                    reasoning_effort="low",
                    max_tokens=150
                )
                print("\n" + "-"*50)
                print("🔍 REASONED RESPONSE:")
                print("-"*50)
                print(f"\n{reasoned_response}\n")
            except Exception as e:
                print(f"\n❌ Error in reasoned completion: {e}")
            
        elif choice == "5":
            # Demo 5: Vision Analysis
            print("\n" + "="*50)
            print("👁️ TEST 5: VISION ANALYSIS")
            print("="*50)
            try:
                image_url = prompt("Enter image URL for analysis", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1200px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
                custom_prompt = prompt("Enter analysis prompt (or use default)", "Describe this image and what it conveys in one paragraph.")
                
                vision_response = text_gen.vision_analysis(
                    image_url=image_url,
                    user_prompt=custom_prompt,
                    system_prompt="You are a concise image analyst. Limit response to 150 tokens in one paragraph.",
                    max_tokens=150
                )
                print("\n" + "-"*50)
                print("🎭 VISION ANALYSIS:")
                print("-"*50)
                print(f"\n{vision_response}\n")
            except Exception as e:
                print(f"\n❌ Error in vision analysis: {e}")
            
        elif choice == "6":
            # Memory Management
            print("\n" + "="*50)
            print("💾 TEST 6: MEMORY & HISTORY MANAGEMENT")
            print("="*50)
            
            # Check current memory state
            print("Current memory state:")
            memory_size = len(text_gen.memory.chat_history) if hasattr(text_gen, "memory") and hasattr(text_gen.memory, "chat_history") else 0
            print(f"Messages in memory: {memory_size}")
            
            # Option to view history
            if memory_size > 0 and prompt("View current chat history? (y/n)", "n").lower() == "y":
                for i, entry in enumerate(text_gen.memory.chat_history):
                    print(f"\nMessage {i+1}:")
                    print(f"Role: {entry.get('role', 'unknown')}")
                    content = entry.get('content', '')
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"Content: {content}")
            
            # Save memory
            if prompt("Save current memory to file? (y/n)", "y").lower() == "y":
                history_path = text_gen.save_history()
                print(f"📁 History saved to: {history_path}")
            
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE CLOSED 🏁")
    print("="*60 + "\n") 