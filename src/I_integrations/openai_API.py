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

# Setup logger for this module
logger = logging.getLogger(__name__)

class OpenAIWrapper:
    """
    Streamlined wrapper for OpenAI's API services.
    Uses logging for operational messages.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        temperature: float = 0.7,
        max_tokens: int = 4096, # Default max_tokens for the API call itself
        system_message: str = "You are a helpful assistant.",
        api_key: Optional[str] = None
    ):  
        # Load from .env file
        load_dotenv(override=True)
        
        # Use provided API key if available, otherwise use environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning("No OpenAI API key provided or found in environment")
        
        # Initialize client
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Settings
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message
        
        # Log initialization 
        logger.info(f"Initialized with model={self.model}, embedding_model={self.embedding_model}")

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
        **kwargs
    ) -> Union[str, Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generate a chat completion with automatic tool call detection.

        Args:
            user_prompt: The user's prompt
            system_prompt: The system prompt
            message_history: History of previous messages
            model: Model to use
            temperature: Controls randomness
            max_tokens: Max tokens to generate
            tools: List of tool schemas (passed directly to API)

        Returns:
            - String content if no tool calls were made
            - Dict with tool call info if a single tool call was made
            - List of tool call dicts if multiple tool calls were made
        """
        try:
            # Select the correct model to use
            effective_model = model or self.model
            logger.info(f"Making chat completion request with model: {effective_model}")

            # Prepare messages
            messages = []
            if system_prompt:
                 messages.append({"role": "system", "content": system_prompt})
            if message_history:
                 messages.extend(message_history)
            # Handle user prompt (could be string or list for vision)
            if isinstance(user_prompt, str):
                messages.append({"role": "user", "content": user_prompt})
            elif isinstance(user_prompt, list): # Handle vision content format
                 messages.append({"role": "user", "content": user_prompt})
            # Handle cases where user_prompt might be None (e.g., tool synthesis call)
            elif user_prompt is None and messages:
                 pass # Continue if history provides context and user_prompt is None
            else:
                 raise ValueError("Invalid user_prompt format or missing content.")

            # Prepare parameters
            params = {
                "model": effective_model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                # Use default max_tokens if not provided or if it's None
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                **kwargs
            }

            # Add tools if provided
            if tools:
                # Basic validation: ensure tools is a list of dicts
                if isinstance(tools, list) and all(isinstance(t, dict) for t in tools):
                    params["tools"] = tools
                    # Optionally set tool_choice if needed, e.g., "auto"
                    # params["tool_choice"] = "auto"
                else:
                    logger.warning("Invalid format for 'tools' parameter. Expected list of dicts.")

            # Handle streaming
            if kwargs.get("stream", False):
                # For streaming, just return the processed stream
                logger.info("Returning processed stream")
                response = self.client.chat.completions.create(**params)
                return self._process_stream(response)

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Check for tool calls
            if response.choices and response.choices[0].message:
                message = response.choices[0].message
                
                # Format tool calls if present
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Standardize tool call format
                    tool_calls = []
                    for tc in message.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}  # Use empty dict if parsing fails
                            
                        tool_calls.append({
                            "name": tc.function.name,
                            "arguments": args
                        })
                    
                    # Return tool calls directly
                    if len(tool_calls) == 1:
                        # If only one tool call, return it as a single dict
                        return tool_calls[0]
                    else:
                        # If multiple tool calls, return the list
                        return tool_calls
                else:
                    # Regular response - just return the content
                    return message.content
            else:
                logger.warning("No content found in response.")
                return ""

        except Exception as e:
            logger.error(f"Error in chat completion: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            # Return error as string for robustness
            return f"An error occurred: {str(e)}"

    def reasoned_completion(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        message_history: List[Dict[str, str]] = None,
        model: str = "o3-mini",
        temperature: Optional[float] = None,
        max_tokens: int = None,
        reasoning_effort: str = "low",  # OpenAI-specific parameter, kept as an option
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Generate a reasoned completion with explicit reasoning steps.
        
        This method is designed for OpenAI's reasoning models which have specific requirements:
        - No support for system messages in the API call
        - Uses reasoning_effort parameter
        - Uses max_completion_tokens instead of max_tokens
        
        Args:
            user_prompt: The user's prompt
            system_prompt: System prompt (will be properly handled for OpenAI)
            message_history: Previous conversation history
            model: Model to use (default: o3-mini)
            temperature: Temperature (not used by OpenAI reasoning models)
            max_tokens: Maximum tokens to generate
            reasoning_effort: Level of reasoning detail (low, medium, high)
            stream: Whether to stream the response
            
        Returns:
            Generated response with reasoning steps (or iterator if streaming)
        """
        try:
            logger.info(f"Making reasoned completion request with model: {model}")
            
            # For OpenAI reasoning models, we need special handling:
            
            # 1. Filter out system messages - OpenAI reasoning models don't support them
            filtered_messages = []
            if message_history:
                filtered_messages = [msg for msg in message_history if msg.get("role") != "system"]
            
            # 2. If system_prompt is provided, inject it into the user prompt for reasoning models
            enriched_prompt = user_prompt
            if system_prompt:
                enriched_prompt = f"System instruction: {system_prompt}\n\nUser request: {user_prompt}"
            
            # 3. Add user message
            filtered_messages.append({
                "role": "user",
                "content": enriched_prompt
            })
            
            # 4. Make API call with the proper parameters for reasoning models
            response = self.client.chat.completions.create(
                model=model,
                messages=filtered_messages,
                reasoning_effort=reasoning_effort,
                max_completion_tokens=max_tokens or self.max_tokens,
                stream=stream
                # Note: temperature is not supported by reasoning models
            )
            
            # Handle streaming response
            if stream:
                logger.info("Reasoning model streaming response started")
                def stream_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in reasoned completion: {e}", exc_info=True)
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
            logger.info(f"Making vision analysis request with model: {model or 'gpt-4o'}")
            
            # Handle URL or local path for image
            if image_path.startswith(('http://', 'https://')):
                # For URLs, use the direct URL
                logger.info(f"Using image URL: {image_path}")
                image_url = {
                    "url": image_path,
                    "detail": detail
                }
            else:
                # For local files, use base64 encoding
                logger.info(f"Loading local image: {image_path}")
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
                logger.info("Vision analysis streaming response started")
                def stream_generator():
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return stream_generator()
            else:
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in vision analysis: {e}", exc_info=True)
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
        """Generate structured JSON outputs using JSON mode or a specific JSON schema."""
        try:
            logger.info(f"Making structured output request with model: {model or self.model}")
            
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
            
            # Set response_format based on whether a specific schema is provided
            if output_schema:
                # Use the new json_schema type if a specific schema is given
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": kwargs.get("schema_name", "custom_structured_output"), # Allow overriding name
                        "description": kwargs.get("schema_description", "User-defined JSON schema."), # Allow overriding description
                        "schema": output_schema,
                        "strict": kwargs.get("strict_schema", True) # Default to strict, allow override
                    }
                }
                logger.info(f"Using json_schema mode with schema: {kwargs.get('schema_name', 'custom_structured_output')}")
            else:
                # Default to json_object mode if no specific schema is provided
                response_format = {"type": "json_object"}
                logger.info("Using json_object mode.")

            # Make API request - use higher max_tokens for structured output
            structured_max_tokens = max_tokens or 4000
            
            response_obj = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temperature or 0.2,
                max_tokens=structured_max_tokens,
                response_format=response_format, # Use the determined response format
                **kwargs
            )
            
            # Extract and parse JSON
            response_content = response_obj.choices[0].message.content
            finish_reason = response_obj.choices[0].finish_reason
            
            # Check finish reason before parsing, as per docs
            if finish_reason == "length":
                 logger.warning(f"Response truncated due to max_tokens ({structured_max_tokens}). JSON may be incomplete.")
                 # Return an error indicating truncation
                 return {"error": "JSON incomplete due to max_tokens limit", "raw_response": response_content}
            elif finish_reason == "stop":
                try:
                    # Parse the JSON response
                    return json.loads(response_content)
                except json.JSONDecodeError as e:
                    # Fallback: Try to extract JSON from markdown (less likely with new modes but safe) 
                    logger.warning(f"Failed to parse direct JSON ({e}), attempting extraction from response")
                    content = response_content.strip()
                    
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
                    logger.error(f"Failed to parse JSON from response: {content}")
                    return {"error": "JSON parsing failed", "raw_response": content}
            else:
                # Handle other finish reasons like 'content_filter' or potentially 'tool_calls'
                logger.warning(f"Response finished with reason: {finish_reason}")
                # For refusals or content filters, the actual content might not be JSON
                # Safest to return an error or the raw content depending on use case.
                # Here, we'll return an error object.
                return {"error": f"Response terminated unexpectedly: {finish_reason}", "raw_response": response_content}

        except Exception as e:
            # Catch potential API errors or other issues
            import traceback
            logger.error(f"Error in structured output: {e}", exc_info=True)
            traceback.print_exc() # Print full traceback for debugging
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
        
        # Print once at the start instead of for each chunk
        if len(texts) > 1:
            logger.info(f"Creating embeddings for {len(texts)} chunks with model: {model}")
        else:
            logger.info(f"Creating embedding with model: {model}")
        
        try:
            # Process texts to ensure they're valid input format
            # - Convert any non-string objects to strings
            # - Ensure no None values or empty strings
            # - Limit length to avoid token limits
            processed_texts = []
            for text in texts:
                if text is None:
                    # Skip None values
                    continue
                if not isinstance(text, str):
                    # Convert non-string objects to string
                    text = str(text)
                if not text.strip():
                    # Skip empty strings
                    continue
                # Limit extremely long texts (OpenAI has token limits)
                if len(text) > 8000:  # Approximate token limit for embedding
                    text = text[:8000]
                processed_texts.append(text)
            
            # Handle empty input case
            if not processed_texts:
                logger.warning("No valid texts to embed")
                # Return zero vector of appropriate size
                # Typical embedding size for text-embedding-3-small
                return np.zeros((0, 1536), dtype='float32')
            
            # Create embeddings in batches if needed (to avoid API limits)
            batch_size = 16  # Adjusted batch size to avoid OpenAI limits
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i+batch_size]
                try:
                    # Call OpenAI API with proper formatting
                    response = self.client.embeddings.create(
                        model=model,
                        input=batch  # Now properly formatted
                    )
                    
                    # Extract embeddings
                    batch_embeddings = [data.embedding for data in response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as batch_error:
                    logger.error(f"Error in embedding batch {i//batch_size+1}: {str(batch_error)}")
                    # Fill with zeros for the failed batch
                    if all_embeddings:
                        # Get dims from first successful batch
                        dims = len(all_embeddings[0])
                        for _ in range(len(batch)):
                            all_embeddings.append([0.0] * dims)
                    else:
                        # No successful batches yet, use default dim
                        for _ in range(len(batch)):
                            all_embeddings.append([0.0] * 1536)  # Default embedding size
            
            # Convert all to numpy array
            return np.array(all_embeddings, dtype='float32')
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}", exc_info=True)
            raise

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
            logger.info(f"Converting text to speech using model: {model}, voice: {voice}")
            
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
                logger.info(f"Speech saved to: {output_path}")
                return output_path
            
            # Return audio content
            return response.content
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}", exc_info=True)
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
            logger.info(f"Generating image with model: {model}, prompt: '{prompt[:30]}...'")
            
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
            
            urls = [img.url for img in response.data]
            logger.info(f"Successfully generated {len(urls)} image(s)")
            # Return URLs of generated images
            return urls
            
        except Exception as e:
            logger.error(f"Error in image generation: {e}", exc_info=True)
            return [f"Error: {str(e)}"]

    def _process_stream(self, response):
        """Process streaming response from OpenAI."""
        try:
            for chunk in response:
                # Check structure for content in streaming chunks
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Error in stream processing: {e}", exc_info=True)
            # Optionally yield an error message or handle differently
            yield f" Stream Error: {str(e)} "

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
            logger.info(f"Transcribing audio using model: {model}")
            
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
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return f"Error: {str(e)}"


# Example usage / Test Suite
if __name__ == "__main__":
    import tempfile
    import time
    import sys
    import requests

    # Configure logging for the test suite
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting OpenAIWrapper Test Suite")

    def prompt(message, default=None):
        """Helper function to get user input with optional default value."""
        if default:
            result = input(f"{message} [{default}]: ")
            return result if result.strip() else default
        return input(f"{message}: ")

    def run_basic_response_test(api):
        """Test 1: Basic Chat Completion"""
        logger.info("\n📝 TEST: Basic Chat Completion")
        result = api.chat_completion(
            user_prompt="What are three key elements for AI-generated music videos?",
            max_tokens=300
        )
        logger.info(f"Response:\n{result}")

    def run_structured_response_test(api):
        """Test 2: Structured Output"""
        logger.info("\n🧩 TEST: Structured Output")
        try:
            result = api.structured_output(
                user_prompt="List 3 essential elements for a pop music video as a JSON object with a key 'elements'.",
                system_prompt="You are a helpful assistant designed to output JSON.",
                temperature=0.2,
                max_tokens=300
            )
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Error generating structured output: {result.get('error')}")
                logger.error(f"Raw response snippet: {result.get('raw_response', '')[:200]}...")
            else:
                logger.info(f"Structured Output: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Error during structured output test: {e}", exc_info=True)

    def run_streaming_test(api):
        """Test 3: Streaming Response"""
        logger.info("\n🔄 TEST: Streaming Response")
        try:
            stream = api.chat_completion(
                user_prompt="Explain AI video transitions in 3 points",
                stream=True,
                max_tokens=150,
            )
            logger.info("\nStreaming output:", flush=True)
            logger.info("="*50, flush=True)
            full_response = ""
            for chunk in stream:
                logger.info(chunk, end="", flush=True)
                full_response += chunk
            logger.info("\n" + "="*50, flush=True)
            logger.info("\nStreaming test complete!", flush=True)
        except Exception as e:
            logger.error(f"\nError in streaming: {str(e)}")

    def run_function_calling_test(api):
        """Test 4: Function Calling with Dummy Schema"""
        logger.info("\n🔧 TEST: Function Calling")
        
        # Create a dummy tool schema directly
        dummy_tool_schema = {
            "type": "function",
            "function": {
                "name": "get_video_style_info",
                "description": "Get information about a specific music video style",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "style": {
                            "type": "string", 
                            "description": "The style of video (e.g. pop, rock, etc.)",
                            "default": "pop"
                        }
                    },
                    "required": []
                }
            }
        }
        
        try:
            # Request with tools
            result = api.chat_completion(
                user_prompt="I want to create a rock music video.",
                tools=[dummy_tool_schema],
                temperature=0.1
            )
            logger.info(f"Response type: {type(result).__name__}")
            
            # Check if the result is a tool call (dict with name/arguments)
            if isinstance(result, dict) and "name" in result and "arguments" in result:
                logger.info("\n✅ Tool call detected!")
                logger.info(f"Function: {result['name']}")
                logger.info(f"Arguments: {json.dumps(result['arguments'], indent=2)}")
                logger.info("\nThe model chose to use a function call instead of a text response.")
            # Check if it's a list of tool calls
            elif isinstance(result, list) and all(isinstance(tc, dict) and "name" in tc for tc in result):
                logger.info("\n✅ Multiple tool calls detected!")
                for i, tool_call in enumerate(result):
                    logger.info(f"\nTool call #{i+1}:")
                    logger.info(f"  Function: {tool_call['name']}")
                    logger.info(f"  Arguments: {json.dumps(tool_call['arguments'], indent=2)}")
                logger.info("\nThe model made multiple function calls.")
            # Otherwise it's a regular text response
            else:
                logger.warning("\n❌ No tool calls detected. Regular text response:")
                logger.warning(f"\n{result[:200]}...")
                logger.warning("\nThe model chose to respond with text instead of calling a function.")
        except Exception as e:
            logger.error(f"Error during function calling test: {e}", exc_info=True)
            import traceback
            traceback.print_exc()

    def run_reasoned_completion_test(api):
        """Test 5: Reasoned Response"""
        logger.info("\n🧠 TEST: Reasoned Response")
        try:
            result = api.reasoned_completion(
                user_prompt="How to maintain video quality in long AI generations?",
                max_tokens=250
            )
            logger.info(f"Response:\n{result}")
        except Exception as e:
            logger.error(f"Error during reasoned completion test: {e}", exc_info=True)
            logger.warning("Note: This test requires access to o3-mini or similar reasoning models")

    def run_vision_test(api):
        """Test 6: Vision Analysis"""
        logger.info("\n👁️ TEST: Vision Analysis")
        try:
            image_url = "https://images.unsplash.com/photo-1511379938547-c1f69419868d" # Example music related image
            logger.info("Testing vision analysis with URL image...")
            result = api.vision_analysis(
                image_path=image_url,
                user_prompt="What's in this image? Describe it briefly.",
                model="gpt-4o",
                max_tokens=150,
                detail="low"
            )
            logger.info(f"Analysis:\n{result}")
        except Exception as e:
            logger.error(f"Error in vision analysis: {e}", exc_info=True)


    def run_embedding_test(api):
        """Test 7: Embeddings"""
        logger.info("\n🔢 TEST: Embeddings")
        try:
            text = "AI video generation techniques"
            embeds = api.create_embeddings(text)
            logger.info(f"Text: '{text}'")
            logger.info(f"Embedding shape: {embeds.shape}")
            if embeds.size > 0:
                logger.info(f"First 5 values: {embeds[0][:5]}")
        except Exception as e:
            logger.error(f"Error during embedding test: {e}", exc_info=True)

    def run_tts_test(api):
        """Test 8: Text-to-Speech"""
        logger.info("\n🔊 TEST: Text-to-Speech")
        try:
            temp_audio_path = os.path.join(tempfile.gettempdir(), "openai_api_test.mp3")
            result = api.text_to_speech(
                text="Welcome to the world of AI-generated music videos!",
                voice="alloy",
                output_path=temp_audio_path
            )
            if os.path.exists(temp_audio_path):
                logger.info(f"Audio saved to: {temp_audio_path}")
                # Try to play the audio (platform specific)
                try:
                    if sys.platform == "darwin": subprocess.run(["afplay", temp_audio_path], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
                    elif sys.platform == "win32": os.startfile(temp_audio_path)
                    else: subprocess.run(["aplay", temp_audio_path], check=False, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) # Requires aplay
                except Exception as play_e:
                    logger.warning(f"Could not automatically play audio: {play_e}")
            else:
                 logger.error(f"Audio file not created at {temp_audio_path}. API response: {result}")

        except Exception as e:
            logger.error(f"Error during TTS test: {e}", exc_info=True)

    def run_image_generation_test(api):
        """Test 9: Image Generation"""
        logger.info("\n🎨 TEST: Image Generation")
        try:
            prompt_text = "A music studio with guitars and recording equipment, digital art style"
            urls = api.generate_image(
                prompt=prompt_text,
                size="1024x1024",
                n=1
            )
            logger.info(f"Image prompt: {prompt_text}")
            if isinstance(urls, list) and urls and not urls[0].startswith("Error"):
                 logger.info(f"Generated image URL: {urls[0]}")
            else:
                 logger.error(f"Image generation failed or returned error: {urls}")
        except Exception as e:
            logger.error(f"Error during image generation test: {e}", exc_info=True)

    def run_transcription_test(api):
        """Test 10: Audio Transcription"""
        logger.info("\n🎤 TEST: Audio Transcription")
        temp_audio_path = None # Initialize
        try:
            # First create a TTS file to transcribe
            temp_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_dir, "transcription_test.mp3")
            tts_result = api.text_to_speech(
                text="This is a test of the OpenAI audio transcription API. It converts speech to text.",
                voice="alloy",
                output_path=temp_audio_path
            )

            if not os.path.exists(temp_audio_path):
                 logger.error(f"Failed to create TTS file for transcription test at {temp_audio_path}. TTS API response: {tts_result}")
                 return # Exit test if audio file wasn't created

            # Now transcribe it
            transcription = api.transcribe_audio(
                audio_file_path=temp_audio_path
            )
            logger.info(f"Created audio with text: 'This is a test of the OpenAI audio transcription API'")
            logger.info(f"Transcription result: '{transcription}'")

        except Exception as e:
            logger.error(f"Error during transcription test: {e}", exc_info=True)
        finally:
             # Clean up the temporary audio file
             if temp_audio_path and os.path.exists(temp_audio_path):
                 try:
                     os.remove(temp_audio_path)
                     logger.info(f"Cleaned up temporary file: {temp_audio_path}")
                 except OSError as e:
                     logger.error(f"Error removing temporary file {temp_audio_path}: {e}")


    # Test order matching other API wrappers with additional tests
    test_functions = {
        "1": ("Basic Response", run_basic_response_test),
        "2": ("Structured Output", run_structured_response_test),
        "3": ("Streaming", run_streaming_test),
        "4": ("Function Calling (Dummy Schema)", run_function_calling_test),
        "5": ("Reasoned Response", run_reasoned_completion_test),
        "6": ("Vision Analysis", run_vision_test),
        "7": ("Embeddings", run_embedding_test),
        "8": ("Text-to-Speech", run_tts_test),
        "9": ("Image Generation", run_image_generation_test),
        "10": ("Audio Transcription", run_transcription_test)
    }

    # Main test menu
    logger.info("\n" + "="*50)
    logger.info("🔷 OPENAI API WRAPPER TEST SUITE (Post-Refactor)")
    logger.info("="*50)

    try:
        # Initialize API with appropriate model
        # Ensure API key is available in .env or passed directly
        api = OpenAIWrapper(
            model="gpt-4o-mini",
            system_message="You are a specialized AI assistant for music video creation."
        )
        if not api.client:
             logger.critical("OpenAI client not initialized. Check API key.")
             exit()

        # Test menu options
        while True:
            logger.info("\nAvailable Tests:")
            for key, (name, _) in test_functions.items():
                logger.info(f"{key}. {name}")
            logger.info("0. Exit")

            choice = prompt("\nSelect a test to run", "0")

            if choice == "0":
                logger.info("\nExiting test suite.")
                break
            elif choice in test_functions:
                logger.info(f"\n--- Running Test {choice}: {test_functions[choice][0]} ---")
                try:
                    _, test_func = test_functions[choice]
                    test_func(api)
                except Exception as e:
                    logger.critical(f"\n❌ Unhandled Error running test: {str(e)}")
                    import traceback
                    traceback.print_exc()

                input("\nPress Enter to continue...")
            else:
                logger.warning("\nInvalid choice. Please try again.")

    except Exception as e:
        logger.critical(f"\n❌ Error initializing test suite: {str(e)}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "="*50)
    logger.info("🏁 TEST SUITE COMPLETED")
    logger.info("="*50) 