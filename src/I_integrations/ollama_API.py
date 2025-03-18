import json
import numpy as np
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Union,
    Iterator,
    Sequence,
    Type,
)
from dotenv import load_dotenv
import base64
from pydantic import BaseModel, ValidationError
import logging
import requests

"""
Ollama_API.py
A standalone Ollama wrapper that mirrors OAI_API.py functionality while
properly implementing all Ollama-specific features.
"""

import ollama

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class OllamaWrapper:
    def __init__(
        self,
        model: str = "llama3.2-vision:11b",
        embedding_model: str = "nomic-embed-text",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_message: str = "You are a helpful assistant.",
        default_options: Optional[Dict[str, Any]] = None,
        auto_pull: bool = False,
    ):
        """Initialize the Ollama wrapper with parameters matching OAIWrapper."""
        load_dotenv(override=True)

        # Get available models first
        self.available_models = self.list_models()
        
        # Validate and potentially pull requested models
        self.model = self._validate_model(model, auto_pull)
        self.embedding_model = self._validate_model(embedding_model, auto_pull)
        
        # Rest of initialization
        self.system_message = system_message

        # Keep Ollama options but use OAI naming in our interface
        self.default_options = {
            'temperature': temperature,
            'num_predict': max_tokens,
            'top_k': 40,
            'top_p': 0.9,
            'repeat_penalty': 1.1,
            'stop': ['</s>', 'user:', 'assistant:'],
            **(default_options or {})
        }
        
        print(f"[OllamaWrapper] Initialized with model={self.model}, embedding_model={self.embedding_model}")

    def _create_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Create properly formatted message list for Ollama API."""
        messages = []
        
        # Add system message first if provided
        system_msg = system_prompt or self.system_message
        if system_msg:
            messages.append({
                "role": "system",
                "content": system_msg
            })
        
        # Add message history if provided
        if message_history:
            messages.extend(message_history)
        
        # Add current user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return messages

    def chat_completion(
        self, 
        user_prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        available_tools: Optional[Dict[str, callable]] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[str, Iterator[Dict[str, Any]]]:
        """Generate text completion using chat models with tool support."""
        try:
            print(f"[OllamaWrapper] Making request with model: {model or self.model}")
            
            # Create messages with history
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                message_history=message_history
            )

            # Prepare parameters
            params = {
                **self.default_options,
                "temperature": temperature or self.default_options['temperature'],
                "num_predict": max_tokens or self.default_options['num_predict'],
                **kwargs
            }
            
            # Handle Ollama-specific format parameter
            format_param = kwargs.pop('format', None)
            if format_param:
                params['format'] = format_param

            if tools and available_tools:
                # Format tool descriptions for the prompt
                tool_descriptions = []
                for tool in tools:
                    tool_desc = (
                        f"Tool: {tool['name']}\n"
                        f"Description: {tool['description']}\n"
                        f"Parameters: {json.dumps(tool['parameters'], indent=2)}"
                    )
                    tool_descriptions.append(tool_desc)
                
                tool_message = (
                    "IMPORTANT INSTRUCTIONS FOR TOOL USAGE:\n"
                    "1. You have access to tool functions to help with your responses. Call them when asked by the user.\n"
                    "2. Do NOT guess/simulate or make up the response if the user is expecting us to use the tool function.\n"
                    "3. IF THE USER DOESN'T ASK for the response the function would provide, respond normally without calling.\n"
                    "4. ONLY WHEN you are asked to use the tool function, response EXACTLY and ONLY the JSON response which calls the tool function and NOTHING ELSE. We have hard coded systems set in place to execute the function based on your response and return the Result back to you so is ESSENTIAL your initial response is EXACTLY the JSON response which calls the tool function and NOTHING ELSE.\n"
                    "5. After recieving the tool function Result, integrate the Result into your final reply without making up or changing the Result.\n"
                    "AVAILABLE TOOLS:\n\n" + 
                    "\n\n".join(tool_descriptions) +
                    "\n\nTO USE A TOOL, respond ONLY with a JSON object containing 'tool' and 'parameters' keys. DO NOT ADD IT IN A BLOCK OF TEXT, JUST THE JSON OBJECT."
                )

                # Use structured_output for guaranteed JSON response
                try:
                    tool_response = self.structured_output(
                        user_prompt=user_prompt,
                        system_prompt=tool_message,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=model,
                        message_history=message_history
                    )
                    
                    # Execute tool if valid response format
                    if isinstance(tool_response, dict) and "tool" in tool_response and "parameters" in tool_response:
                        tool_name = tool_response["tool"]
                        if tool_name in available_tools:
                            tool_func = available_tools[tool_name]
                            result = tool_func(**tool_response["parameters"])
                            
                            # Add tool result to messages and get final response
                            messages.extend([
                                {"role": "assistant", "content": f"Tool called:\n{json.dumps(tool_response)} \nResult of tool execution is:\n {result}"},
                                {"role": "user", "content": "Now respond naturally using ONLY the Result of the tool call. DO NOT make up or change the result."}
                            ])
                            
                            final_response = ollama.chat(
                                model=model or self.model,
                                messages=messages,
                                options=params,
                                stream=stream
                            )

                            if stream:
                                return self._process_stream(final_response)
                            return final_response["message"]["content"]
                except Exception as tool_error:
                    print(f"[OllamaWrapper] Error in tool execution: {tool_error}")
                    # Fall through to normal response if tool execution fails
            else:
                print("[OllamaWrapper] No tools or available_tools provided, skipping tool branch.")

            # Standard chat completion without tools
            response = ollama.chat(
                model=model or self.model,
                messages=messages,
                options=params,
                stream=stream
            )

            if stream:
                return self._process_stream(response)
            
            return response["message"]["content"]

        except Exception as e:
            print(f"Error in chat completion: {e}")
            raise

    def reasoned_completion(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = "deepseek-r1:8b",
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[str, Iterator[str], Dict[str, str]]:
        """Generate reasoned completion using chat_completion."""
        try:
            # Build a modified system prompt to encourage a thinking process with <think></think> tags
            thinking_prompt = (
                (system_prompt or self.system_message) +
                "\nShow your thinking process inside <think></think> tags."
                "\nIf there are no thinking steps, add the <think></think> tags but leave them empty."
                "\nWrite your final response outside of those tags."
            )

            # Build messages including message_history using _create_messages (if available)
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=thinking_prompt,
                message_history=message_history
            )

            if stream:
                response = self.chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    message_history=messages,
                    **kwargs
                )
                def process_stream():
                    buffer = ""
                    in_thinking = False
                    for chunk in response:
                        buffer += chunk
                        # Remove <think> when first encountered
                        if "<think>" in buffer and not in_thinking:
                            buffer = buffer.replace("<think>", "")
                            in_thinking = True
                        # When </think> is found, split out the thinking section
                        if "</think>" in buffer and in_thinking:
                            thinking, rest = buffer.split("</think>", 1)
                            buffer = rest
                            in_thinking = False
                        yield chunk
                    # Once done, check if the buffer contains the think tags and process them
                    if "<think>" in buffer and "</think>" in buffer:
                        parts = buffer.split("</think>", 1)
                        thinking = parts[0].replace("<think>", "").strip()
                        response_text = parts[1].strip()
                        return {"thinking": thinking, "response": response_text}
                    else:
                        return buffer
                return process_stream()
            else:
                response_obj = self.chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    message_history=messages,
                    **kwargs
                )
                full_response = response_obj
                if "<think>" in full_response and "</think>" in full_response:
                    parts = full_response.split("</think>", 1)
                    thinking = parts[0].replace("<think>", "").strip()
                    response_text = parts[1].strip()
                    return {"thinking": thinking, "response": response_text}
                else:
                    return full_response
            
        except Exception as e:
            print(f"Error in reasoned completion: {e}")
            raise

    def vision_analysis(
        self,
        image_path: str,
        user_prompt: str = "What's in this image?",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Analyze images using chat completion."""
        try:
            print(f"[OllamaWrapper] Making vision request with model: {model or self.model}")
            
            # Handle URL or local path
            if image_path.startswith(('http://', 'https://')):
                print(f"Fetching image from URL: {image_path}")
                response = requests.get(image_path, stream=True)
                response.raise_for_status()
                
                # Get image data directly from response
                image_data = base64.b64encode(response.content).decode('utf-8')
            else:
                # Read local file
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Create messages with image
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                message_history=message_history
            )
            
            # Add image to the last user message
            messages[-1]["images"] = [image_data]
            
            # Use chat_completion for consistency
            return self.chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                model=model or "llama3.2-vision:11b",  # Ensure vision model
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                message_history=messages,  # Use our modified messages
                **kwargs
            )

        except Exception as e:
            print(f"Error in vision analysis: {e}")
            raise
        
    def structured_output(
        self,
        user_prompt: str,
        output_class: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Any:
        """Generate structured outputs using Pydantic models or direct JSON."""
        try:
            print(f"[OllamaWrapper] Making structured output request with model: {model or self.model}")

            # Create a more explicit system prompt for JSON formatting
            json_system_prompt = (
                (system_prompt or "") + "\n\n"
                "IMPORTANT: Respond ONLY with a valid JSON object. "
                "No markdown, no explanations, just the JSON object. "
                "Ensure all JSON strings are properly escaped and terminated."
            )

            enhanced_prompt = user_prompt
            
            # Add schema information if output_class is provided
            if output_class:
                schema = output_class.schema()
                example = {}
                for field, details in schema["properties"].items():
                    field_type = details.get("type", "string")
                    if field_type == "string":
                        example[field] = "example string"
                    elif field_type == "array":
                        example[field] = ["example item 1", "example item 2"]
                    elif field_type == "integer":
                        example[field] = 25
                    else:
                        example[field] = "example value"

                enhanced_prompt = (
                    f"{user_prompt}\n\n"
                    "IMPORTANT: You must respond with ONLY valid JSON matching this exact schema:\n"
                    f"{json.dumps(schema, indent=2)}\n\n"
                    "Example format:\n"
                    f"{json.dumps(example, indent=2)}\n\n"
                    "Your JSON response:"
                )

            # Create messages and parameters
            messages = self._create_messages(
                user_prompt=enhanced_prompt,
                system_prompt=json_system_prompt,
                message_history=message_history
            )

            params = {
                **self.default_options,
                "temperature": temperature or 0.2,  # Lower temperature for structured output
                "num_predict": max_tokens or self.default_options['num_predict'],
                "format": "json",  # Important: Tell Ollama to expect JSON
                **kwargs
            }

            # Make direct API call
            response = ollama.chat(
                model=model or self.model,
                messages=messages,
                options=params
            )

            content = response["message"]["content"].strip()
            
            # Clean up response
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                json_response = json.loads(content)
                
                # Validate against output_class if provided
                if output_class:
                    return output_class(**json_response)
                return json_response
                
            except json.JSONDecodeError as je:
                print(f"JSON parse error at position {je.pos}: {je.msg}")
                print(f"Raw content: {content}")
                raise
            except ValidationError as ve:
                print(f"Validation error: {ve}")
                raise

        except Exception as e:
            print(f"Error in structured output: {e}")
            raise

    def create_embeddings(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """Generate embeddings for text (matches OAI_API.py)."""
        the_model = model or self.embedding_model
        texts = [text] if isinstance(text, str) else text
        
        try:
            # Print once at the start instead of for each chunk
            if len(texts) > 1:
                print(f"[OllamaWrapper] Creating embeddings for {len(texts)} chunks with model: {the_model}")
            else:
                print(f"[OllamaWrapper] Creating embedding with model: {the_model}")
            
            embeddings = []
            for t in texts:
                response = ollama.embeddings(
                    model=the_model,
                    prompt=t
                )
                embeddings.append(response.embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            print(f"[OllamaWrapper] Embedding error: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        """List available models, handling errors gracefully."""
        try:
            response = ollama.list()
            return [model.get("name", model.get("model", "unknown")) 
                   for model in response.models]
        except Exception as e:
            print(f"[OllamaWrapper] Error listing models: {str(e)}")
            return []

    def pull_model(self, model_name: str):
        """Pull a model from Ollama's registry."""
        try:
            print(f"[OllamaWrapper] Pulling model: {model_name}")
            
            for progress in ollama.pull(model_name):
                status = progress.status
                if status:
                    print(f"[OllamaWrapper] Pull status: {status}")
        except Exception as e:
            print(f"[OllamaWrapper] Pull error: {str(e)}")
            raise

    def _validate_model(self, model: str, auto_pull: bool = False) -> str:
        """
        Validate that a model is available, optionally pull it, or raise error.
        Returns the model name if valid, raises ValueError if not available.
        """
        if not model:
            raise ValueError("Model name cannot be empty")

        # Strip :latest if present for comparison
        base_model = model.replace(":latest", "")
        
        # Check if model or model:latest is available
        model_available = any(
            m.startswith(base_model) for m in self.available_models
        )

        if not model_available:
            if auto_pull:
                print(f"[OllamaWrapper] Model '{model}' not found. Attempting to pull...")
                try:
                    self.pull_model(model)
                    return model
                except Exception as e:
                    raise ValueError(
                        f"Failed to pull model '{model}'. Error: {str(e)}\n"
                        f"Available models: {self.available_models}"
                    )
            else:
                raise ValueError(
                    f"Model '{model}' not available locally and auto_pull=False.\n"
                    f"Available models: {self.available_models}\n"
                    "Either:\n"
                    f"1. Run: wrapper.pull_model('{model}')\n"
                    "2. Choose from available models\n"
                    "3. Set auto_pull=True in constructor"
                )
        
        return model

    def convert_function_to_schema(self, func) -> Dict:
        """Convert a Python function to Ollama's function schema format."""
        try:
            from pydantic import BaseModel, Field
            from typing import get_type_hints, Annotated
            import inspect

            # Get function signature and docstring
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)
            
            # Create schema for the function
            schema = {
                "name": func.__name__,
                "description": func.__doc__ or "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            def get_json_type(python_type: type) -> str:
                """Convert Python type to JSON schema type."""
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                return type_map.get(python_type, "string")
                    
            # Add parameters to schema
            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, str)
                description = ""
                
                # Extract parameter description from docstring
                if func.__doc__:
                    doc_lines = func.__doc__.split('\n')
                    for line in doc_lines:
                        if f':param {param_name}:' in line:
                            description = line.split(f':param {param_name}:')[1].strip()
                
                # Add parameter to schema
                schema["parameters"]["properties"][param_name] = {
                    "type": get_json_type(param_type),
                    "description": description
                }
                
                # Add to required if no default value
                if param.default == param.empty:
                    schema["parameters"]["required"].append(param_name)
            
            return schema
            
        except Exception as e:
            print(f"Error converting function to tool schema: {e}")
            raise

    def _process_stream(self, response):
        """Process streaming response from Ollama"""
        try:
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            print(f"[OllamaWrapper] Error in stream processing: {e}")

    def _execute_tool_call(self, tool_call: Dict, available_tools: Dict[str, callable]) -> Any:
        """Execute a tool call and return its result."""
        try:
            func_name = tool_call["name"]
            if func_name not in available_tools:
                raise ValueError(f"Unknown tool: {func_name}")
            
            args = tool_call["arguments"]
            return available_tools[func_name](**args)
        except Exception as e:
            return f"Error executing {func_name}: {str(e)}"

# Add the test suite below
if __name__ == "__main__":
    import tempfile
    import time
    import sys
    import requests
    
    def prompt(message, default=None):
        """Helper function to get user input with optional default value."""
        if default:
            result = input(f"{message} [{default}]: ")
            return result if result.strip() else default
        return input(f"{message}: ")

    def run_basic_response_test(api):
        """Test 1: Basic Response"""
        print("\n📝 Basic Response Test")
        result = api.chat_completion(
            user_prompt="What are three key elements for AI-generated music videos?",
            max_tokens=300
        )
        print(f"Response:\n{result}")

    def run_structured_response_test(api):
        """Test 2: Structured Output"""
        print("\n🧩 Structured Output Test")
        from pydantic import BaseModel
        
        class VideoStyle(BaseModel):
            elements: List[str]
            software: List[str]
            duration: str

        try:
            result = api.structured_output(
                user_prompt="List elements for a pop music video",
                output_class=VideoStyle,
                max_tokens=200
            )
            print(f"Structured Output: {json.dumps(result.dict(), indent=2)}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_streaming_test(api):
        """Test 3: Streaming"""
        print("\n🔄 Streaming Test")
        stream = api.chat_completion(
            user_prompt="Explain AI video transitions in 3 points",
            stream=True,
            max_tokens=150
        )
        print("Streaming output:")
        for chunk in stream:
            print(chunk, end="", flush=True)
        print("\nStream complete")

    def run_function_calling_test(api):
        """Test 4: Function Calling"""
        print("\n🔧 Function Test")
        def get_video_tools(style: str):
            return {"tools": ["AE", "Blender"]}
        
        try:
            result = api.chat_completion(
                user_prompt="Get tools for rock videos using get_video_tools",
                tools=[api.convert_function_to_schema(get_video_tools)],
                available_tools={"get_video_tools": get_video_tools},
                temperature=0.1,
                max_tokens=200
            )
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_reasoned_completion_test(api):
        """Test 5: Reasoned Response"""
        print("\n🧠 Reasoned Test")
        result = api.reasoned_completion(
            user_prompt="How to maintain video quality in long AI generations?",
            max_tokens=250
        )
        print(f"Response:\n{result}")

    def run_vision_test(api):
        """Test 6: Vision"""
        print("\n👁️ Vision Test")
        try:
            # Use valid public image URL
            image_url = "https://images.unsplash.com/photo-1511379938547-c1f69419868d"
            result = api.vision_analysis(
                image_path=image_url,
                user_prompt="Describe this music video frame",
                model="llama3.2-vision:11b",
                max_tokens=200
            )
            print(f"Analysis:\n{result}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_embedding_test(api):
        """Test 7: Embeddings"""
        print("\n🔢 Embeddings Test")
        embeds = api.create_embeddings(["AI video tools"])
        print(f"Embedding shape: {embeds.shape}")

    # Test order matching OpenAI structure
    test_functions = {
        "1": ("Basic Response", run_basic_response_test),
        "2": ("Structured Output", run_structured_response_test),
        "3": ("Streaming", run_streaming_test),
        "4": ("Function Calling", run_function_calling_test),
        "5": ("Reasoned Response", run_reasoned_completion_test),
        "6": ("Vision Analysis", run_vision_test),
        "7": ("Embeddings", run_embedding_test)
    }

    # Main test menu
    print("\n" + "="*50)
    print("🦙 OLLAMA API TEST SUITE")
    print("="*50)

    try:
        # Initialize API
        api = OllamaWrapper(
            model="llama3.2-vision:11b",
            auto_pull=True,
            system_message="You are a specialized AI assistant for music video creation."
        )

        # Test menu options
        while True:
            print("\nAvailable Tests:")
            for key, (name, _) in test_functions.items():
                print(f"{key}. {name}")
            print("0. Exit")

            choice = prompt("\nSelect a test to run", "0")

            if choice == "0":
                print("\nExiting test suite.")
                break
            elif choice in test_functions:
                try:
                    _, test_func = test_functions[choice]
                    test_func(api)
                except Exception as e:
                    print(f"\n❌ Error running test: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                input("\nPress Enter to continue...")
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"\n❌ Error initializing test suite: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("🏁 TEST SUITE COMPLETED")
    print("="*50)