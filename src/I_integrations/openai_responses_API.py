import os
import json
import numpy as np
from typing import List, Dict, Union, Optional, Any, Type, Iterator
from openai import OpenAI
from dotenv import load_dotenv
import logging
import time
import sys

# Disable noisy HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("primp").setLevel(logging.WARNING)

class OpenAIWrapper:
    """
    Streamlined wrapper for OpenAI's Responses API services.
    Provides simplified access to key OpenAI capabilities.
    """
    def __init__(
        self,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_output_tokens: int = 4096,
        system_message: str = "You are a helpful assistant.",
        top_p: float = 1.0,
        truncation: str = "disabled",
        api_key: Optional[str] = None
    ):  
        # Load from .env file
        load_dotenv(override=True)
        
        # Use provided API key if available, otherwise use environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No OpenAI API key provided or found in environment. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key)
        
        # Settings
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_message = system_message
        self.previous_response_id = None  # Track conversation state
        self.top_p = top_p
        self.truncation = truncation
        
        # Log initialization without showing API key
        logging.info(f"OpenAI Responses API initialized with model: {self.model}")

    def _create_input(
        self,
        user_prompt: Union[str, List],
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Create properly formatted input list for OpenAI Responses API."""
        input_messages = []
        
        # Add system message first if provided
        if system_prompt:
            input_messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add message history if provided
        if message_history:
            input_messages.extend(message_history)
        
        # Add current user prompt
        if isinstance(user_prompt, str):
            input_messages.append({"role": "user", "content": user_prompt})
        else:
            # Handle case when user_prompt is more complex (e.g., for vision analysis)
            input_messages.append({"role": "user", "content": user_prompt})
        
        return input_messages

    def response(
        self, 
        user_prompt: str,
        system_prompt: str = None,
        message_history: List[Dict[str, Any]] = None,
        model: str = None,
        temperature: float = None,
        max_output_tokens: int = None,
        tools: List[Dict] = None,
        available_functions: Dict[str, callable] = None,
        stream: bool = False,
        store: bool = True,
        top_p: float = None,
        truncation: str = None,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """Generate a response using the Responses API."""
        try:
            # Create input messages
            input_messages = self._create_input(
                user_prompt=user_prompt,
                system_prompt=system_prompt if 'instructions' not in kwargs else None,
                message_history=message_history
            )

            # Setup parameters
            params = {
                "model": model or self.model,
                "input": input_messages,
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
                "previous_response_id": self.previous_response_id,
                "stream": stream,
                "store": store,
                "top_p": top_p or self.top_p,
                "truncation": truncation or self.truncation,
                **kwargs
            }

            # Add system prompt as instructions if needed
            if system_prompt and 'instructions' not in kwargs:
                params["instructions"] = system_prompt

            # Add tools if provided
            if tools:
                params["tools"] = tools

            # Make API call
            response = self.client.responses.create(**params)
            
            # Handle streaming
            if stream:
                return self._process_stream(response)
            
            # Store response ID for conversation continuity
            self.previous_response_id = response.id
            
            # Check for function calls
            if available_functions and hasattr(response, 'output'):
                function_calls = []
                
                # First collect all function calls in the response
                for item in response.output:
                    if hasattr(item, 'type') and item.type == 'function_call':
                        function_name = item.name
                        function_calls.append(item)
                
                # If we have function calls, handle them
                if function_calls:
                    # Create a copy of the input messages for the next request
                    updated_messages = list(input_messages)
                    
                    # Process each function call
                    for function_call in function_calls:
                        function_name = function_call.name
                        if function_name in available_functions:
                            try:
                                # Parse arguments
                                try:
                                    arguments = json.loads(function_call.arguments)
                                except json.JSONDecodeError:
                                    # If can't parse as JSON, use empty dict
                                    arguments = {}
                                
                                # Make sure arguments is a dict
                                if not isinstance(arguments, dict):
                                    arguments = {}
                                
                                # Execute the function
                                function_result = available_functions[function_name](**arguments)
                                
                                # Convert result to string if necessary
                                if not isinstance(function_result, str):
                                    function_result = json.dumps(function_result)
                                
                                # Add function call to conversation
                                updated_messages.append(function_call)
                                
                                # Add function output
                                updated_messages.append({
                                    "type": "function_call_output",
                                    "call_id": function_call.call_id,
                                    "output": function_result
                                })
                            except Exception as e:
                                # In case of error, add error message as function output
                                error_message = f"Error executing function: {str(e)}"
                                updated_messages.append(function_call)
                                updated_messages.append({
                                    "type": "function_call_output",
                                    "call_id": function_call.call_id,
                                    "output": error_message
                                })
                    
                    # Call API again with updated messages including function results
                    second_params = {**params, "input": updated_messages}
                    second_response = self.client.responses.create(**second_params)
                    
                    # Store response ID for conversation continuity
                    self.previous_response_id = second_response.id
                    
                    # Return the final response text
                    return second_response.output_text
            
            # If no function calls, return the text output
            return response.output_text if hasattr(response, 'output_text') else "No output text available"

        except Exception as e:
            return f"Error: {str(e)}"

    chat_completion = response
    
    def _process_stream(self, response):
        """Process streaming response from OpenAI Responses API."""
        try:
            for i, event in enumerate(response):
                if hasattr(event, "type") and event.type == "response.output_text.delta":
                    # Handle direct text delta
                    if hasattr(event, "delta"):
                        text = event.delta  # Get delta directly as text
                        print(text, end="", flush=True)
                        yield text
                elif hasattr(event, "output_text"):
                    # Handle full text output as fallback
                    text = event.output_text
                    print(text, end="", flush=True)
                    yield text
        except Exception as e:
            error_msg = f"\nStreaming error: {str(e)}\n"
            print(error_msg, file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            yield error_msg

    def structured_response(
        self,
        user_prompt: str,
        output_schema: Optional[Dict] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        model: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        truncation: Optional[str] = None,
        store: bool = True,
        schema_name: Optional[str] = "output_schema",
        strict: bool = True,
        **kwargs
    ) -> Any:
        """Generate structured JSON outputs using the Responses API.
        
        Args:
            user_prompt: The prompt to send to the model
            output_schema: JSON schema to validate the response against
            system_prompt: System instructions for the model
            temperature: Controls randomness (0-2)
            max_output_tokens: Maximum tokens to generate
            model: Model to use
            message_history: Previous conversation history
            top_p: Nucleus sampling parameter
            truncation: How to handle context overflow
            store: Whether to store the response
            schema_name: Name for the schema (optional)
            strict: Whether to enforce strict schema validation
        """
        try:
            # Create input messages
            input_messages = self._create_input(
                user_prompt=user_prompt,
                system_prompt=None if system_prompt and 'instructions' not in kwargs else system_prompt,
                message_history=message_history
            )

            # Setup parameters
            params = {
                "model": model or self.model,
                "input": input_messages,
                "temperature": temperature or 0.2,  # Lower temperature for structured output
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
                "top_p": top_p or self.top_p,
                "truncation": truncation or self.truncation,
                "store": store,
                **kwargs
            }
            
            # Add text format parameter with correct structure
            if output_schema:
                params["text"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": output_schema
                    }
                }
                
                # Add name if provided
                if schema_name:
                    params["text"]["format"]["name"] = schema_name
            else:
                # For simple JSON without schema, use json_object
                params["text"] = {
                    "format": {
                        "type": "json_object"
                    }
                }
                
            # Add instructions if system_prompt is provided and not used in input
            if system_prompt and 'instructions' not in kwargs and system_prompt not in input_messages:
                params["instructions"] = system_prompt

            # Call the API
            response = self.client.responses.create(**params)
            
            # Try to parse the JSON response
            try:
                # Handle refusals or incomplete responses
                if hasattr(response, 'status') and response.status == "incomplete":
                    return {"error": f"Incomplete response: {response.incomplete_details.reason if hasattr(response.incomplete_details, 'reason') else 'unknown reason'}"}
                
                return json.loads(response.output_text)
            except json.JSONDecodeError:
                # Try to clean the response if needed
                content = response.output_text.strip()
                
                # Extract from code blocks if needed
                if "```json" in content:
                    try:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                        return json.loads(json_str)
                    except (IndexError, json.JSONDecodeError):
                        pass
                        
                # Return error with raw response
                return {"error": "JSON parsing failed", "raw_response": content}

        except Exception as e:
            print(f"Error in structured response: {e}")
            return {"error": str(e)}
    
    # Alias for backward compatibility
    structured_output = structured_response

    def reasoned_response(
        self,
        user_prompt: str,
        reasoning_effort: str = "low",
        message_history: List[Dict[str, Any]] = None,
        model: str = None,
        max_output_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        system_prompt: str = None,
        store: bool = True,
        **kwargs
    ) -> str:
        """Generate a reasoned response using the reasoning parameter.
        
        Note: Full reasoning support requires an O-series model like o1.
        If these models aren't available, falls back to standard completion
        with a reasoning prompt modifier.
        """
        try:
            # Default to the instance default model
            if not model:
                model = self.model
            
            # Create input messages
            input_messages = self._create_input(
                user_prompt=user_prompt,
                system_prompt=None if system_prompt and 'instructions' not in kwargs else system_prompt,
                message_history=message_history
            )

            # Setup parameters
            params = {
                "model": model,
                "input": input_messages,
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "store": store,
                **kwargs
            }
            
            # Add reasoning parameter if model likely supports it
            if model.startswith('o1'):
                if reasoning_effort in ["low", "medium", "high"]:
                    params["reasoning"] = {"effort": reasoning_effort}
            else:
                # For non-O1 models, use a modified system prompt instead
                effort_descriptions = {
                    "low": "Provide a brief analysis with basic reasoning.",
                    "medium": "Provide a thoughtful analysis with clear step-by-step reasoning.",
                    "high": "Provide a comprehensive, in-depth analysis with detailed step-by-step reasoning and consideration of multiple perspectives."
                }
                
                reasoning_instruction = effort_descriptions.get(reasoning_effort, effort_descriptions["medium"])
                if system_prompt:
                    system_prompt = f"{system_prompt}\n\n{reasoning_instruction}"
                else:
                    system_prompt = f"You are a helpful assistant.\n\n{reasoning_instruction}"
            
            # Add instructions if needed
            if system_prompt and 'instructions' not in kwargs:
                params["instructions"] = system_prompt
                # If we have system messages in input and instructions parameter, remove from input
                input_messages = [msg for msg in input_messages if msg.get('role') != 'system']
                params["input"] = input_messages

            # Call the API
            response = self.client.responses.create(**params)
            self.previous_response_id = response.id
            return response.output_text

        except Exception as e:
            print(f"Error in reasoned response: {e}")
            return f"Error: {str(e)}"
    
    # Alias for backward compatibility
    reasoned_completion = reasoned_response

    def create_embeddings(
        self, 
        texts: Union[str, List[str]],
        model: str = "text-embedding-3-small"
    ) -> np.ndarray:
        """Generate embeddings for given text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=model,
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        return np.array(embeddings, dtype='float32')

    def vision_response(
        self,
        image_path: str,
        user_prompt: str = "What's in this image?",
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        truncation: Optional[str] = None,
        store: bool = True,
        **kwargs
    ) -> str:
        """Analyze images using the Responses API."""
        try:
            # Handle image data - direct URL string instead of object
            image_url = image_path
            if not image_path.startswith(('http://', 'https://')):
                # For local files, convert to base64 data URL
                import base64
                with open(image_path, "rb") as img_file:
                    base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{base64_image}"
            
            # Create content array with correct format for vision according to Responses API
            content = [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_image", "image_url": image_url}
            ]
            
            # Create input messages
            input_messages = [{"role": "user", "content": content}]
            
            # Setup parameters
            params = {
                "model": model or self.model,
                "input": input_messages,
                "temperature": temperature or self.temperature,
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
                "top_p": top_p or self.top_p,
                "truncation": truncation or self.truncation,
                "store": store,
                **kwargs
            }
            
            # Add system prompt as instructions
            if system_prompt:
                params["instructions"] = system_prompt
            
            # Use responses API
            response = self.client.responses.create(**params)
            
            # Store for conversation continuity
            self.previous_response_id = response.id
            
            return response.output_text

        except Exception as e:
            print(f"Error in vision response: {e}")
            return f"Error: {str(e)}"
    
    # Alias for backward compatibility
    vision_analysis = vision_response

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
            if output_path:
                # Use the non-deprecated streaming approach
                with self.client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                    speed=speed,
                    response_format="mp3"
                ) as response:
                    # Open the output file and write chunks
                    with open(output_path, "wb") as audio_file:
                        for chunk in response.iter_bytes():
                            audio_file.write(chunk)
                    return output_path
            else:
                # Simple non-streaming response for returning bytes
                response = self.client.audio.speech.create(
                    model=model,
                    voice=voice,
                    input=text,
                    speed=speed,
                    response_format="mp3"
                )
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
            response = self.client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style,
                n=n,
                **kwargs
            )
            
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
            
            # Extract main description from docstring
            main_desc = doc.split("\n")[0].strip() if doc else ""
            
            # Create schema with name at both levels as required by Responses API
            schema = {
                "type": "function",
                "name": func.__name__,  # Top-level name field required by Responses API
                "function": {
                    "name": func.__name__,
                    "description": main_desc,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
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
                    "type": type_map.get(param_type, "string")
                }
                
                # Try to extract parameter description from docstring
                param_desc_marker = f":param {param_name}:"
                if param_desc_marker in doc:
                    param_desc = doc.split(param_desc_marker)[1]
                    param_desc = param_desc.split("\n")[0].split(":param")[0].strip()
                    param_schema["description"] = param_desc
                
                # Add to schema
                schema["function"]["parameters"]["properties"][param_name] = param_schema
                
                # Add to required list if no default value
                if param.default == param.empty:
                    schema["function"]["parameters"]["required"].append(param_name)
            
            return schema
            
        except Exception as e:
            print(f"Error converting function to schema: {e}")
            return {
                "type": "function",
                "name": func.__name__,  # Top-level name field
                "function": {
                    "name": func.__name__,
                    "description": func.__doc__ or "",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

    def web_search_response(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        message_history: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        truncation: Optional[str] = None,
        store: bool = True,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Perform web search using the built-in web search tool.
        
        Args:
            user_prompt: User query to search for
            system_prompt: System message/instructions
            message_history: Previous conversation history
            model: Model to use
            max_output_tokens: Maximum tokens to generate
            temperature: Controls randomness (0-2)
            top_p: Nucleus sampling parameter
            truncation: How to handle context overflow
            store: Whether to store the response
            include: List of tool use parameters to include in the response
            
        Returns:
            Response text or object with search results if include is provided
        """
        try:
            # Create input messages
            input_messages = self._create_input(
                user_prompt=user_prompt,
                system_prompt=None if system_prompt and 'instructions' not in kwargs else system_prompt,
                message_history=message_history
            )
            
            # Setup parameters
            params = {
                "model": model or self.model,
                "input": input_messages,
                "tools": [{"type": "web_search"}],
                "max_output_tokens": max_output_tokens or self.max_output_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "truncation": truncation or self.truncation,
                "store": store,
                **kwargs
            }
            
            # Add instructions if needed
            if system_prompt and 'instructions' not in kwargs and system_prompt not in input_messages:
                params["instructions"] = system_prompt
                
            # Add include parameter if requested
            if include:
                params["include"] = include
            
            # Call the API
            response = self.client.responses.create(**params)
            
            # Store response ID for conversation continuity
            self.previous_response_id = response.id
            
            # Return full response object or just text depending on include
            if include:
                search_results = None
                # Extract search results from the response if available
                if hasattr(response, 'tool_use'):
                    search_results = response.tool_use
                
                return {
                    "text": response.output_text,
                    "response_id": response.id,
                    "search_results": search_results
                }
            else:
                return response.output_text
            
        except Exception as e:
            print(f"Error in web search response: {e}")
            return f"Error: {str(e)}"
    
    # Alias for backward compatibility
    web_search = web_search_response


# Example usage and testing
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
        """Test 1: Basic Response Method"""
        print("\n📝 TEST: Basic Response Method (response)")
        user_prompt = "What are three innovative approaches to create AI-generated music videos longer than 5 minutes while maintaining viewer engagement?"
        
        result = api.response(
            user_prompt=user_prompt,
            temperature=0.8,
            max_output_tokens=800
        )
        print(f"Response: {result}")

    def run_structured_response_test(api):
        """Test 2: Structured Response Method"""
        print("\n🧩 TEST: Structured Response Method (structured_response)")
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "video_components": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "required_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of required software tools"
                        },
                        "ai_models": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of AI models needed"
                        },
                        "estimated_processing_time": {
                            "type": "string",
                            "description": "Estimated time for processing"
                        },
                        "technical_requirements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Technical specifications needed"
                        }
                    },
                    "required": ["required_tools", "ai_models", "estimated_processing_time", "technical_requirements"]
                }
            },
            "required": ["video_components"]
        }
        
        json_result = api.structured_response(
            user_prompt="List the technical requirements and AI models needed for generating a 10-minute music video",
            output_schema=schema,
            max_output_tokens=600
        )
        print("Technical Requirements:")
        print(json.dumps(json_result, indent=2))

    def run_instructions_test(api):
        """Test 3: Response with Instructions"""
        print("\n📝 TEST: Instructions Parameter Method (response with instructions)")
        instructions = "You are a music AI expert. Focus on practical steps for generating cohesive background music for long videos."
        user_prompt = "How can we generate consistent background music that maintains thematic coherence for a 10-minute video?"
        
        music_result = api.response(
            user_prompt=user_prompt,
            system_prompt=instructions,
            temperature=0.7,
            max_output_tokens=600
        )
        print(f"Music Generation Strategy: {music_result}")

    def run_streaming_test(api):
        """Test 4: Streaming Response"""
        print("\n🔄 TEST: Streaming Response Method (response with stream=True)")
        print("Testing streaming capability - watch text appear word by word:")
        
        # Simple prompt from the OpenAI documentation
        prompt = "Say 'double bubble bath' ten times fast."
        
        print("\nStreaming response (should appear incrementally):", flush=True)
        print("="*50, flush=True)
        
        # Use the existing response method with streaming
        stream = api.response(
            user_prompt=prompt,
            stream=True,
            temperature=0.7,
            max_output_tokens=300
        )
        
        # Just iterate through the stream - printing happens inside _process_stream
        for _ in stream:
            pass
        
        print("\n" + "="*50, flush=True)
        print("\nStreaming test complete!", flush=True)

    def run_function_calling_test(api):
        """Test 5: Function Calling"""
        print("\n🔧 TEST: Function Calling Method (response with tools)")
        
        # Define very simple functions that are less likely to fail
        def get_video_style_info(style="pop"):
            """Get information about a specific music video style.
            
            :param style: The style of music video (pop, rock, electronic, acoustic)
            """
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
                },
                "electronic": {
                    "software": ["After Effects", "TouchDesigner"],
                    "visual_elements": ["Neon effects", "Particle systems", "Glitch effects"],
                    "typical_length": "3-5 minutes"
                },
                "acoustic": {
                    "software": ["Premiere Pro", "DaVinci Resolve"],
                    "visual_elements": ["Natural settings", "Soft lighting", "Minimalist"],
                    "typical_length": "3-4 minutes"
                }
            }
            
            # Default to pop if style not found
            return styles.get(style.lower(), styles["pop"])
        
        # Test function directly to verify
        print("\nTesting function directly:")
        direct_result = get_video_style_info("rock")
        print(f"Direct function result for 'rock' style: {json.dumps(direct_result, indent=2)}")
        
        # Create function schema
        style_schema = api.convert_function_to_schema(get_video_style_info)
        print("\nFunction schema:")
        print(json.dumps(style_schema, indent=2))
        
        # Test with extremely explicit prompt
        print("\nTesting function calling with explicit prompt...")
        explicit_prompt = """
I want to create a rock music video.
Please use the get_video_style_info function with style="rock" to tell me what software and visual elements I should use.
"""
        
        result = api.response(
            user_prompt=explicit_prompt,
            tools=[style_schema],
            available_functions={"get_video_style_info": get_video_style_info},
            temperature=0.1,  # Low temperature for more deterministic results
            system_prompt="You are a video production assistant. When asked about music video styles, ALWAYS use the get_video_style_info function with the specific style parameter."
        )
        
        print(f"\nResult: {result}")
        
        # Test with simple parameter-less function for comparison
        print("\nTesting parameter-less function...")
        
        def get_music_production_tips():
            """Get a list of tips for music video production."""
            return {
                "pre_production": [
                    "Create a detailed storyboard",
                    "Scout locations in advance",
                    "Prepare a shot list with timing"
                ],
                "production": [
                    "Use a stabilizer for smooth movements",
                    "Record multiple takes of key scenes",
                    "Ensure consistent lighting throughout"
                ],
                "post_production": [
                    "Color grade for visual consistency",
                    "Sync audio and visuals precisely",
                    "Add subtle visual effects to enhance mood"
                ]
            }
        
        # Create schema for the simple function
        tips_schema = api.convert_function_to_schema(get_music_production_tips)
        
        simple_result = api.response(
            user_prompt="What are some important tips for producing a high-quality music video? Use the get_music_production_tips function.",
            tools=[tips_schema],
            available_functions={"get_music_production_tips": get_music_production_tips},
            temperature=0.1,
            system_prompt="You're a music video production consultant. When asked for production tips, use the get_music_production_tips function."
        )
        
        print(f"\nSimple function result: {simple_result}")

    def run_reasoned_response_test(api):
        """Test 6: Reasoned Response"""
        print("\n🧠 TEST: Reasoned Response Method (reasoned_response)")
        style_prompt = "What's the most effective way to transition between AI-generated video segments while maintaining visual coherence?"
        
        style_result = api.reasoned_response(
            user_prompt=style_prompt,
            reasoning_effort="low",
            max_output_tokens=300,
            temperature=0.7
        )
        print(f"Reasoned Analysis (Low Effort): {style_result}")

    def run_web_search_test(api):
        """Test 7: Web Search"""
        print("\n🔍 TEST: Web Search Method (web_search_response)")
        search_result = api.web_search_response(
            user_prompt="What are the latest AI tools and techniques for long-form video generation in 2024?",
            max_output_tokens=600,
            include=["web_search_call.results"]
        )
        
        if isinstance(search_result, dict):
            print("\nAssistant's Response:")
            print(search_result["text"])
            print("\nSearch Results Used:")
            if search_result.get("search_results"):
                for result in search_result["search_results"]:
                    print(f"\n- {result['title']}")
                    print(f"  URL: {result['url']}")
                    print(f"  Snippet: {result['snippet']}")
        else:
            print(f"Search Results: {search_result}")

    def run_conversation_test(api):
        """Test 8: Conversation State"""
        print("\n🔄 TEST: Conversation State Method (response with previous_response_id)")
        
        first_message = api.response(
            user_prompt="What's the first step in planning a 10-minute AI-generated music video?",
            max_output_tokens=400
        )
        print(f"First Response: {first_message}")
        
        follow_up = api.response(
            user_prompt="What tools and timeline should we consider for this first step?",
            max_output_tokens=400
        )
        print(f"Follow-up Response: {follow_up}")

    def run_vision_test(api):
        """Test 9: Vision Analysis"""
        print("\n👁️ TEST: Vision Analysis Method (vision_response)")
        example_image_url = "https://images.unsplash.com/photo-1511379938547-c1f69419868d?q=80"
        
        vision_result = api.vision_response(
            image_path=example_image_url,
            user_prompt="How can we replicate this visual style using AI tools in our music video?",
            model="gpt-4o",
            max_output_tokens=500
        )
        print(f"Vision Analysis: {vision_result}")

    def run_tts_test(api):
        """Test 10: Text-to-Speech"""
        print("\n🔊 TEST: Text-to-Speech Method (text_to_speech)")
        temp_audio_path = os.path.join(tempfile.gettempdir(), "video_narration_test.mp3")
        
        tts_result = api.text_to_speech(
            text="Welcome to the future of AI-generated music videos, where creativity meets technology.",
            output_path=temp_audio_path
        )
        print(f"Audio Output: {temp_audio_path}")

    def run_image_generation_test(api):
        """Test 11: Image Generation"""
        print("\n🎨 TEST: Image Generation Method (generate_image)")
        scene_prompt = "A seamless transition between a digital waveform and a natural landscape, styled for a music video"
        
        image_urls = api.generate_image(
            prompt=scene_prompt,
            model="dall-e-3",
            size="1024x1024",
            quality="hd"
        )
        print(f"Generated Image: {image_urls[0] if image_urls else 'None'}")

    # Main test menu
    print("\n" + "="*50)
    print("🚀 OPENAI RESPONSES API TEST SUITE")
    print("="*50)

    try:
        # Initialize API with specific instructions for concise responses
        api = OpenAIWrapper(
            model="gpt-4o",
            system_message="You are a specialized AI assistant for music video creation. Provide clear, impactful answers focusing on practical steps and innovative approaches. Keep responses concise but informative."
        )

        # Test menu options
        test_functions = {
            "1": ("Basic Response Test", run_basic_response_test),
            "2": ("Structured Response Test", run_structured_response_test),
            "3": ("Instructions Test", run_instructions_test),
            "4": ("Streaming Test", run_streaming_test),
            "5": ("Function Calling Test", run_function_calling_test),
            "6": ("Reasoned Response Test", run_reasoned_response_test),
            "7": ("Web Search Test", run_web_search_test),
            "8": ("Conversation State Test", run_conversation_test),
            "9": ("Vision Analysis Test", run_vision_test),
            "10": ("Text-to-Speech Test", run_tts_test),
            "11": ("Image Generation Test", run_image_generation_test),
        }

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
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                
                input("\nPress Enter to continue...")
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"\n❌ Error initializing test suite: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("🏁 TEST SUITE COMPLETED")
    print("="*50)
