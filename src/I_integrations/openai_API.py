import os
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any, Type, Iterator
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import sys
import time
import tempfile
import requests
import logging

# Disable noisy HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

class OpenAIAPI:
    """
    Streamlined wrapper for OpenAI's API services.
    Provides simplified access to key OpenAI capabilities.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        system_message: str = "You are a helpful assistant.",
        api_key: Optional[str] = None
    ):  
        # Load from .env file
        load_dotenv(override=True)
        
        # Use provided API key if available, otherwise use environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided or found in environment")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Settings
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        
        # Log initialization without showing API key
        logging.info(f"OpenAI API initialized with model: {self.model}")

    def _create_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Create properly formatted message list for OpenAI API."""
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
        if isinstance(user_prompt, str):
            messages.append({"role": "user", "content": user_prompt})
        else:
            # Handle case when user_prompt is a list (for vision analysis)
            messages.append({"role": "user", "content": user_prompt})
        
        return messages

    def chat_completion(
        self, 
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        message_history: List[Dict[str, str]] = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        tools: List[Dict] = None,
        available_tools: Dict[str, callable] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion with optional tools.
        
        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt
            message_history: History of previous messages
            model: Model to use
            temperature: Controls randomness
            max_tokens: Max tokens to generate
            tools: List of tool schemas
            available_tools: Dictionary mapping tool names to callable functions
            
        Returns:
            Generated response
        """
        try:
            # Prepare messages
            messages = message_history or []
            
            # Add system message if not already in history
            if not any(msg.get("role") == "system" for msg in messages):
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": user_prompt})
            
            # Prepare parameters
            params = {
                "model": model or self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
                **kwargs
            }
            
            # Add tools if provided
            if tools:
                params["tools"] = tools
            
            # Make API call
            response = self.client.chat.completions.create(**params)
            
            # Check for tool calls
            message = response.choices[0].message
            
            # If no tool calls, return content
            if not (hasattr(message, 'tool_calls') and message.tool_calls):
                return message.content
            
            # Process tool calls
            if tools and available_tools:
                # Store original assistant message
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id, 
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Process each tool call
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    
                    if func_name in available_tools:
                        # Parse arguments
                        try:
                            func_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            func_args = {}
                        
                        # Call the function
                        try:
                            result = available_tools[func_name](**func_args)
                        except Exception as e:
                            result = f"Error executing {func_name}: {str(e)}"
                        
                        # Convert result to string if needed
                        if not isinstance(result, str):
                            result = json.dumps(result)
                        
                        # Add tool response
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                
                # Make second API call with tool results
                second_response = self.client.chat.completions.create(
                    model=model or self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens
                )
                
                return second_response.choices[0].message.content
            
            return message.content
            
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return f"An error occurred: {str(e)}"

    def reasoned_completion(
        self,
        user_prompt: str,
        reasoning_effort: str = "low",
        message_history: List[Dict[str, str]] = None,
        model: str = "o3-mini",
        max_tokens: int = None,
        **kwargs
    ) -> str:
        """
        Generate a reasoned completion with explicit reasoning steps.
        
        Args:
            user_prompt: The user's prompt
            reasoning_effort: Level of reasoning detail (low, medium, high)
            message_history: Previous conversation history (non-system messages only)
            model: Model to use (default: o1-mini)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response with reasoning steps
        """
        try:
            # Prepare messages - IMPORTANT: o1-mini doesn't support system messages
            # Filter out any system messages and only keep user/assistant messages
            messages = []
            if message_history:
                messages = [msg for msg in message_history if msg.get("role") != "system"]
            
            # Add user message
            messages.append({
                "role": "user",
                "content": user_prompt
            })
            
            # Make API call with the proper parameters for o1-mini model
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_tokens or self.max_tokens
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"\n❌ Error in reasoned completion: {e}")
            return f"Error: {e}"

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
        detail: str = "auto",
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Analyze images with vision capabilities using OpenAI's API.
        
        Args:
            image_path: URL or local path to the image
            user_prompt: Text prompt to accompany the image
            system_prompt: Optional system message
            model: Model to use (defaults to gpt-4o)
            temperature: Controls randomness
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            message_history: Previous conversation history
            detail: Detail level for image processing ('auto', 'low', or 'high')
            
        Returns:
            Generated text description or analysis of the image
        """
        try:
            # Handle URL or local path for image
            if image_path.startswith(('http://', 'https://')):
                # For URLs, use the direct URL
                image_url = {
                    "url": image_path,
                    "detail": detail
                }
            else:
                # For local files, use base64 encoding
                import base64
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                    image_url = {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail
                    }
            
            # Create content array with text and image
            content = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": image_url}
            ]
            
            # Prepare messages
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add message history if provided
            if message_history:
                messages.extend(message_history)
            
            # Add user message with content
            messages.append({"role": "user", "content": content})
            
            # Make API call
            response = self.client.chat.completions.create(
                model=model or "gpt-4o",  # Default to gpt-4o for vision
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream,
                **kwargs
            )
            
            # Handle streaming response
            if stream:
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content

        except Exception as e:
            print(f"Error in vision analysis: {e}")
            return f"Error: {str(e)}"

    def structured_output(
        self,
        user_prompt: str,
        output_schema: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Any:
        """Generate structured JSON outputs."""
        try:
            # Enhance system prompt for JSON output
            json_system_prompt = (
                (system_prompt or self.system_message) + "\n\n"
                "IMPORTANT: You must respond with a valid JSON object. "
                "No other text or explanation should be included in your response."
            )
            
            # Create messages
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=json_system_prompt,
                message_history=message_history
            )
            
            # Add schema if provided
            response_format = {"type": "json_object"} 
            if output_schema:
                # Advanced: can include schema validation requirements
                response_format["schema"] = output_schema
            
            # Make API request
            response_obj = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or 0.2,  # Lower temperature for structured output
                max_tokens=max_tokens or self.max_tokens,
                response_format=response_format,
                **kwargs
            )
            
            # Extract and parse JSON
            response = response_obj.choices[0].message.content
            
            try:
                # Parse the JSON response
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback: Try to extract JSON from markdown or code blocks
                content = response.strip()
                
                # Extract from markdown code block
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        return json.loads(json_str)
                    except (IndexError, json.JSONDecodeError):
                        pass
                
                # Extract from generic code block
                if "```" in content:
                    try:
                        json_str = content.split("```")[1].strip()
                        return json.loads(json_str)
                    except (IndexError, json.JSONDecodeError):
                        pass
                
                # Return error if parsing fails
                print(f"Failed to parse JSON from response: {content}")
                return {"error": "JSON parsing failed", "raw_response": content}

        except Exception as e:
            print(f"Error in structured output: {e}")
            return {"error": str(e)}

    def create_embeddings(
        self, 
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """
        Generate embeddings for given text(s).
        
        Args:
            texts: Text or list of texts to embed
            model: Model to use for embeddings
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=model,
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings, dtype='float32')

    def text_to_speech(
        self, 
        text: str,
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """Convert text to speech using OpenAI's TTS."""
        try:
            # Create speech
            response = self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3"
            )
            
            # Save to file if path provided
            if output_path:
                response.stream_to_file(output_path)
                return output_path
            
            # Return audio content
            return response.content
            
        except Exception as e:
            print(f"Error in text-to-speech conversion: {e}")
            return f"Error: {str(e)}"

    def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "standard",
        style: str = "vivid",
        n: int = 1,
        model: str = "dall-e-3",
        **kwargs
    ) -> List[str]:
        """Generate images using DALL-E models."""
        try:
            # Generate images
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=n,
                **kwargs
            )
            
            # Return URLs of generated images
            return [img.url for img in response.data]
            
        except Exception as e:
            print(f"Error in image generation: {e}")
            return [f"Error: {str(e)}"]

    def convert_function_to_schema(self, func) -> Dict:
        """Convert a Python function to OpenAI's function schema format."""
        try:
            import inspect
            from typing import get_type_hints
            
            # Get function signature and docstring
            sig = inspect.signature(func)
            doc = func.__doc__ or ""
            type_hints = get_type_hints(func)
            
            # Create schema
            schema = {
                "name": func.__name__,
                "description": doc.split("\n")[0] if doc else "",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            
            # Type mapping
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object"
            }
            
            # Add parameters
            for param_name, param in sig.parameters.items():
                # Skip self parameter
                if param_name == "self":
                    continue
                    
                # Get type and description
                param_type = type_hints.get(param_name, str)
                param_schema = {
                    "type": type_map.get(param_type, "string"),
                    "description": ""
                }
                
                # Try to extract parameter description from docstring
                param_desc_marker = f":param {param_name}:"
                if param_desc_marker in doc:
                    param_desc = doc.split(param_desc_marker)[1]
                    param_desc = param_desc.split("\n")[0].split(":param")[0].strip()
                    param_schema["description"] = param_desc
                
                # Add to schema
                schema["parameters"]["properties"][param_name] = param_schema
                
                # Add to required list if no default value
                if param.default == param.empty:
                    schema["parameters"]["required"].append(param_name)
            
            # Return formatted schema for OpenAI
            return {
                "type": "function",
                "function": schema
            }
            
        except Exception as e:
            print(f"Error converting function to schema: {e}")
            return {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "parameters": {"type": "object", "properties": {}}
                }
            }

    def _execute_tool_call(self, tool_call, available_tools: Dict[str, callable]) -> Any:
        """Execute a tool call with error handling."""
        try:
            # Extract function name and arguments
            func_name = tool_call.function.name
            args_str = tool_call.function.arguments
            
            # Parse arguments
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                return f"Error: Invalid JSON in arguments: {args_str}"
            
            # Check if function exists
            if func_name not in available_tools:
                return f"Error: Unknown tool '{func_name}'"
            
            # Execute function with arguments
            return available_tools[func_name](**args)
            
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def _process_stream(self, response):
        """Process streaming response from OpenAI."""
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def transcribe_audio(
        self,
        audio_file_path: str,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "text",
        temperature: float = 0.0
    ) -> str:
        """Transcribe audio using OpenAI's Whisper model."""
        try:
            # Prepare parameters
            params = {
                "model": model,
                "file": open(audio_file_path, "rb"),
                "response_format": response_format,
                "temperature": temperature
            }
            
            # Add optional parameters if provided
            if language:
                params["language"] = language
            if prompt:
                params["prompt"] = prompt
            
            # Make API call
            response = self.client.audio.transcriptions.create(**params)
            
            # Return transcription text
            return response
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Error: {str(e)}"


# Example usage
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
        """Test 1: Basic Chat Completion"""
        print("\n📝 TEST: Basic Chat Completion")
        result = api.chat_completion(
            user_prompt="What are three key elements for AI-generated music videos?",
            max_tokens=300
        )
        print(f"Response:\n{result}")

    def run_structured_response_test(api):
        """Test 2: Structured Output"""
        print("\n🧩 TEST: Structured Output")
        
        try:
            # Use a simpler prompt that will generate a more compact response
            result = api.structured_output(
                user_prompt="List 3 essential elements for a pop music video",
                temperature=0.2,
                max_tokens=300  # Increase tokens to ensure complete response
            )
            
            # Check if we got an error response
            if isinstance(result, dict) and "error" in result:
                print(f"Error: {result['error']}")
                if "raw_response" in result:
                    # Try to clean and parse the raw response
                    try:
                        raw = result["raw_response"].strip()
                        # If it looks like JSON, try to manually complete it
                        if raw.startswith("{") and not raw.endswith("}"):
                            raw += "}"  # Add closing brace if missing
                        fixed_json = json.loads(raw)
                        print("Recovered structured output:")
                        print(json.dumps(fixed_json, indent=2))
                    except:
                        print("Could not recover structured output")
                        print(f"Raw response: {result['raw_response'][:100]}...")
            else:
                print(f"Structured Output: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_streaming_test(api):
        """Test 3: Streaming Response"""
        print("\n🔄 TEST: Streaming Response")
        print("Note: This test demonstrates how to process streaming responses")
        
        try:
            # Create a streaming response directly using the client
            stream = api.client.chat.completions.create(
                model=api.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Explain AI video transitions in 3 points"}
                ],
                max_tokens=150,
                stream=True  # Enable streaming
            )
            
            # Process the stream
            print("\nStreaming output:", flush=True)
            print("="*50, flush=True)
            
            # Use the _process_stream method to handle the streaming response
            for chunk in api._process_stream(stream):
                print(chunk, end="", flush=True)
                
            print("\n" + "="*50, flush=True)
            print("\nStreaming test complete!", flush=True)
            
        except Exception as e:
            print(f"Error in streaming: {str(e)}")

    def run_function_calling_test(api):
        """Test 4: Function Calling"""
        print("\n🔧 TEST: Function Calling")
        
        def get_video_style_info(style="pop"):
            """Get information about a specific music video style."""
            styles = {
                "pop": {
                    "software": ["Adobe After Effects", "Final Cut Pro"],
                    "visual_elements": ["Vibrant colors", "Quick cuts", "Choreography"],
                    "typical_length": "3-5 minutes"
                },
                "rock": {
                    "software": ["Final Cut Pro", "DaVinci Resolve"],
                    "visual_elements": ["Concert footage", "Film grain", "High contrast"],
                    "typical_length": "4-6 minutes"
                }
            }
            print(f"[Function executed] Getting style info for: {style}")
            return styles.get(style.lower(), styles["pop"])
        
        # Create function schema
        style_schema = api.convert_function_to_schema(get_video_style_info)
        
        try:
            result = api.chat_completion(
                user_prompt="I want to create a rock music video. Use get_video_style_info with style='rock'",
                tools=[style_schema],
                available_tools={"get_video_style_info": get_video_style_info},
                temperature=0.1
            )
            print(f"Result:\n{result}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_reasoned_completion_test(api):
        """Test 5: Reasoned Response"""
        print("\n🧠 TEST: Reasoned Response")
        try:
            # Remove the debug prints by using a simpler approach
            result = api.reasoned_completion(
                user_prompt="How to maintain video quality in long AI generations?",
                max_tokens=250
            )
            print(f"Response:\n{result}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Note: This test requires access to o1-mini or similar reasoning models")

    def run_vision_test(api):
        """Test 6: Vision Analysis"""
        print("\n👁️ TEST: Vision Analysis")
        try:
            # Use a reliable image URL from the OpenAI documentation
            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            
            print("Testing vision analysis with URL image...")
            result = api.vision_analysis(
                image_path=image_url,
                user_prompt="What's in this image? Describe it briefly.",
                model="gpt-4o",  # Explicitly use gpt-4o which has vision capabilities
                max_tokens=150,
                detail="low"  # Use low detail to save tokens
            )
            
            print(f"Analysis:\n{result}")
            
            # Try with a local image if available
            try:
                # Create a temporary image file for testing
                import tempfile
                import requests
                
                print("\nTesting vision analysis with local image file...")
                # Download the image to a temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                temp_file_path = temp_file.name
                temp_file.close()
                
                # Download the image
                img_response = requests.get(image_url)
                with open(temp_file_path, 'wb') as f:
                    f.write(img_response.content)
                
                print(f"Created temporary image file: {temp_file_path}")
                
                # Test with local file
                local_result = api.vision_analysis(
                    image_path=temp_file_path,
                    user_prompt="What's in this image? Describe it briefly.",
                    model="gpt-4o",
                    max_tokens=150,
                    detail="high"  # Test high detail with local file
                )
                
                print(f"Local file analysis:\n{local_result}")
                
                # Clean up
                import os
                os.unlink(temp_file_path)
                print(f"Removed temporary file: {temp_file_path}")
                
            except Exception as e:
                print(f"Local file test failed: {e}")
            
        except Exception as e:
            print(f"Error in vision analysis: {e}")
            
            # Try an alternative approach with more debugging
            try:
                print("\nTrying alternative approach with more debugging...")
                
                # Check if the image URL is accessible
                response = requests.head(image_url)
                print(f"Image URL status code: {response.status_code}")
                
                # Try with a different model
                print("Trying with gpt-4o-mini model...")
                result = api.vision_analysis(
                    image_path=image_url,
                    user_prompt="What's in this image?",
                    model="gpt-4o-mini",
                    max_tokens=100,
                    detail="low"
                )
                print(f"Analysis (gpt-4o-mini):\n{result}")
            except Exception as e2:
                print(f"Alternative approach also failed: {e2}")
                print(f"Error details: {str(e2)}")

    def run_embedding_test(api):
        """Test 7: Embeddings"""
        print("\n🔢 TEST: Embeddings")
        try:
            text = "AI video generation techniques"
            embeds = api.create_embeddings(text)
            print(f"Text: '{text}'")
            print(f"Embedding shape: {embeds.shape}")
            print(f"First 5 values: {embeds[0][:5]}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_tts_test(api):
        """Test 8: Text-to-Speech"""
        print("\n🔊 TEST: Text-to-Speech")
        try:
            temp_audio_path = os.path.join(tempfile.gettempdir(), "openai_api_test.mp3")
            result = api.text_to_speech(
                text="Welcome to the world of AI-generated music videos!",
                voice="alloy",
                output_path=temp_audio_path
            )
            print(f"Audio saved to: {temp_audio_path}")
            
            # Try to play the audio
            try:
                if sys.platform == "darwin":  # Mac
                    subprocess.run(["afplay", temp_audio_path], check=False)
                elif sys.platform == "win32":  # Windows
                    os.startfile(temp_audio_path)
                else:  # Linux
                    subprocess.run(["aplay", temp_audio_path], check=False)
            except Exception as e:
                print(f"Could not automatically play audio: {e}")
                
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_image_generation_test(api):
        """Test 9: Image Generation"""
        print("\n🎨 TEST: Image Generation")
        try:
            prompt = "A music studio with guitars and recording equipment, digital art style"
            urls = api.generate_image(
                prompt=prompt,
                size="1024x1024",
                n=1
            )
            print(f"Image prompt: {prompt}")
            print(f"Generated image URL: {urls[0] if urls else 'No URL returned'}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def run_transcription_test(api):
        """Test 10: Audio Transcription"""
        print("\n🎤 TEST: Audio Transcription")
        try:
            # First create a TTS file to transcribe
            temp_audio_path = os.path.join(tempfile.gettempdir(), "transcription_test.mp3")
            api.text_to_speech(
                text="This is a test of the OpenAI audio transcription API. It converts speech to text.",
                voice="alloy",
                output_path=temp_audio_path
            )
            
            # Now transcribe it
            transcription = api.transcribe_audio(
                audio_file_path=temp_audio_path
            )
            print(f"Created audio with text: 'This is a test of the OpenAI audio transcription API'")
            print(f"Transcription result: '{transcription}'")
        except Exception as e:
            print(f"Error: {str(e)}")

    # Test order matching other API wrappers with additional tests
    test_functions = {
        "1": ("Basic Response", run_basic_response_test),
        "2": ("Structured Output", run_structured_response_test),
        "3": ("Streaming", run_streaming_test),
        "4": ("Function Calling", run_function_calling_test),
        "5": ("Reasoned Response", run_reasoned_completion_test),
        "6": ("Vision Analysis", run_vision_test),
        "7": ("Embeddings", run_embedding_test),
        "8": ("Text-to-Speech", run_tts_test),
        "9": ("Image Generation", run_image_generation_test),
        "10": ("Audio Transcription", run_transcription_test)
    }

    # Main test menu
    print("\n" + "="*50)
    print("🔷 OPENAI API TEST SUITE")
    print("="*50)

    try:
        # Initialize API with appropriate model
        api = OpenAIAPI(
            model="gpt-4o-mini",
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