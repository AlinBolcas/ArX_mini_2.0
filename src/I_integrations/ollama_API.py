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
        
        logger.info(f"[OllamaWrapper] Initialized with model={self.model}, embedding_model={self.embedding_model}")

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

    def _extract_tool_call_from_text(self, text_response: str, tools: Optional[List[Dict]] = None) -> Union[Dict, List[Dict], str]:
        """
        Extract tool calls from text responses for compatibility with OpenAI format.
        
        Args:
            text_response: Text response from Ollama
            tools: List of tools that were made available
            
        Returns:
            - Dict with tool call info if a single tool call is detected
            - List of dict with tool calls if multiple tool calls are detected
            - Original string if no tool calls are detected
        """
        # If no tools were provided or response doesn't look JSON-like, return as is
        if not tools or not (
            text_response.strip().startswith("{") or 
            "```json" in text_response or 
            "```" in text_response or
            "```tool_code" in text_response or
            "tool_code:" in text_response
        ):
            return text_response
        
        try:
            # Step 1: Extract JSON from various formats
            json_objects = []
            
            # Check for tool_code blocks specifically - models often use this format
            if "```tool_code" in text_response or "tool_code:" in text_response:
                logger.debug("[OllamaWrapper] Detected tool_code format, attempting to parse")
                
                # Match tool_code blocks with ```tool_code``` format
                import re
                tool_code_blocks = re.findall(r'```tool_code\n(.*?)\n```', text_response, re.DOTALL)
                if not tool_code_blocks:
                    # Try alternate format: ```tool_code ... ```
                    tool_code_blocks = re.findall(r'```tool_code\s*(.*?)\s*```', text_response, re.DOTALL)
                
                # Process each tool_code block
                for block in tool_code_blocks:
                    # First try to directly parse it as JSON
                    try:
                        if block.strip().startswith("{"):
                            json_obj = json.loads(block.strip())
                            json_objects.append(json_obj)
                            continue
                    except json.JSONDecodeError:
                        pass
                    
                    # If not JSON, it might be in the format: tool_code: function_name, param1="value1", param2="value2"
                    if "tool_code:" in block or "," in block:
                        try:
                            # Extract function name and params
                            block = block.strip()
                            if block.startswith("tool_code:"):
                                block = block[len("tool_code:"):].strip()
                            
                            # Split by first comma to separate function name and params
                            parts = block.split(",", 1)
                            function_name = parts[0].strip()
                            
                            arguments = {}
                            if len(parts) > 1:
                                # Parse param=value pairs
                                params_text = parts[1].strip()
                                param_pattern = r'(\w+)=(?:"([^"]*)"|\{([^}]*)\}|(\S+))'
                                param_matches = re.findall(param_pattern, params_text)
                                
                                for match in param_matches:
                                    param_name = match[0]
                                    # Find the first non-empty value among the capture groups
                                    param_value = next((v for v in match[1:] if v), "")
                                    arguments[param_name] = param_value
                            
                            json_objects.append({
                                "function": function_name,
                                "arguments": arguments
                            })
                            logger.debug(f"[OllamaWrapper] Extracted tool call from non-JSON format: {function_name}")
                        except Exception as e:
                            logger.error(f"[OllamaWrapper] Error parsing non-JSON tool_code: {e}")
            
            # Try direct JSON parse first for a single complete object
            if not json_objects and text_response.strip().startswith("{"):
                try:
                    # First try to parse the entire response as a single JSON object
                    json_obj = json.loads(text_response.strip())
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    # If that fails, it might be multiple JSON objects in sequence
                    # Look for patterns like "} {" or "}\n{" that indicate multiple objects
                    import re
                    # Find all potential JSON objects using regex
                    potential_jsons = re.findall(r'({[^{}]*(?:{[^{}]*})*[^{}]*})', text_response)
                    for json_str in potential_jsons:
                        try:
                            json_obj = json.loads(json_str)
                            json_objects.append(json_obj)
                        except json.JSONDecodeError:
                            continue
            
            # Try code blocks with JSON
            if not json_objects and "```json" in text_response:
                json_blocks = text_response.split("```json")
                for block in json_blocks[1:]:
                    if "```" in block:
                        json_text = block.split("```")[0].strip()
                        # Check for multiple JSON objects in this block too
                        try:
                            # Try as single object first
                            json_obj = json.loads(json_text)
                            json_objects.append(json_obj)
                        except json.JSONDecodeError:
                            # Try to find multiple objects
                            import re
                            potential_jsons = re.findall(r'({[^{}]*(?:{[^{}]*})*[^{}]*})', json_text)
                            for json_str in potential_jsons:
                                try:
                                    json_obj = json.loads(json_str)
                                    json_objects.append(json_obj)
                                except json.JSONDecodeError:
                                    continue
            
            # Try generic code blocks if still not found
            if not json_objects and "```" in text_response:
                code_blocks = text_response.split("```")
                for i in range(1, len(code_blocks), 2):
                    if i < len(code_blocks):
                        json_text = code_blocks[i].strip()
                        if json_text.startswith("{"):
                            try:
                                # Try as single object first
                                json_obj = json.loads(json_text)
                                json_objects.append(json_obj)
                            except json.JSONDecodeError:
                                # Try to find multiple objects
                                import re
                                potential_jsons = re.findall(r'({[^{}]*(?:{[^{}]*})*[^{}]*})', json_text)
                                for json_str in potential_jsons:
                                    try:
                                        json_obj = json.loads(json_str)
                                        json_objects.append(json_obj)
                                    except json.JSONDecodeError:
                                        continue
            
            # Step 2: Process extracted JSON objects
            tool_calls = []
            for json_obj in json_objects:
                # Check if it looks like a tool call
                if (
                    ("function" in json_obj and "arguments" in json_obj) or
                    ("name" in json_obj and "arguments" in json_obj) or
                    ("tool" in json_obj and "parameters" in json_obj)
                ):
                    # Standardize format to match OpenAI API
                    # Extract either "function", "name", or "tool" as the function name
                    function_name = json_obj.get("function", 
                                             json_obj.get("name", 
                                              json_obj.get("tool")))
                    
                    # Extract either "arguments" or "parameters" as the arguments
                    arguments = json_obj.get("arguments", json_obj.get("parameters", {}))
                    
                    # Ensure arguments is a dict
                    if not isinstance(arguments, dict):
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {"raw_input": arguments}
                        else:
                            arguments = {"value": arguments}
                    
                    # Add standardized tool call
                    tool_calls.append({
                        "name": function_name,
                        "arguments": arguments
                    })
            
            # Step 3: Return in OpenAI-compatible format
            if len(tool_calls) == 1:
                logger.debug(f"[OllamaWrapper] Extracted single tool call: {tool_calls[0]['name']}")
                return tool_calls[0]  # Single tool call as dict
            elif len(tool_calls) > 1:
                logger.debug(f"[OllamaWrapper] Extracted multiple tool calls: {[tc['name'] for tc in tool_calls]}")
                return tool_calls  # Multiple tool calls as list
            
            # No valid tool calls found
            return text_response
            
        except Exception as e:
            logger.error(f"[OllamaWrapper] Error parsing tool calls: {e}")
            import traceback
            traceback.print_exc()
            # Return original response if parsing fails
            return text_response

    def chat_completion(
        self, 
        user_prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        message_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> Union[str, Dict, List[Dict], Iterator[str]]:
        """
        Generate text completion using chat models with tool support.
        
        For tool/function calling, Ollama models don't have native function calling
        like OpenAI, but we describe available tools in the system prompt, extract
        tool calls if present, and return in the same format as OpenAI.
        
        Returns:
            - String content if no tool calls were made
            - Dict with tool call info if a single tool call was made
            - List of tool call dicts if multiple tool calls were made
            - For streaming: An iterator yielding content chunks
        """
        try:
            logger.info(f"[OllamaWrapper] Making request with model: {model or self.model}")
            
            # If tools are provided, enhance the system prompt to describe them
            enhanced_system_prompt = system_prompt
            if tools and not stream:
                tool_descriptions = []
                for tool in tools:
                    # Handle both OpenAI format and simple format
                    if isinstance(tool, dict):
                        if "function" in tool:  # OpenAI format
                            name = tool["function"].get("name", "unknown")
                            desc = tool["function"].get("description", "")
                            params = tool["function"].get("parameters", {})
                        else:  # Direct schema
                            name = tool.get("name", "unknown")
                            desc = tool.get("description", "")
                            params = tool.get("parameters", {})
                        
                        # Format a more compelling tool description
                        param_desc = []
                        if "properties" in params:
                            for param_name, param_info in params.get("properties", {}).items():
                                param_type = param_info.get("type", "string")
                                param_desc_text = param_info.get("description", "")
                                
                                # Add enum values if available
                                enum_values = param_info.get("enum", [])
                                if enum_values:
                                    param_desc_text += f" Options: {', '.join(enum_values)}"
                                    
                                param_desc.append(f"- {param_name} ({param_type}): {param_desc_text}")
                        
                        tool_desc = (
                            f"Function: {name}\n"
                            f"Description: {desc}\n"
                            f"Parameters:\n" + 
                            "\n".join(param_desc) +
                            f"\nRequired parameters: {', '.join(params.get('required', []))}"
                        )
                        tool_descriptions.append(tool_desc)
                
                if tool_descriptions:
                    # Create a more forceful instruction to use tools
                    enhanced_system_prompt = (system_prompt or self.system_message) + "\n\n" + (
                        "CRITICALLY IMPORTANT INSTRUCTIONS FOR TOOL USAGE:\n\n"
                        "You have access to the following tools/functions:\n\n" + 
                        "\n\n".join(tool_descriptions) + 
                        "\n\n"
                        "STRICTLY FOLLOW THESE RULES:\n"
                        "1. IF the user's request requires ANY of these tools, output ONLY a JSON object and NOTHING ELSE.\n"
                        "2. The JSON MUST use this EXACT format:\n"
                        '{"function": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}\n\n'
                        "3. DO NOT include explanations, markdown formatting, or code blocks around the JSON.\n"
                        "4. DO NOT use text before or after the JSON object.\n"
                        "5. DO NOT write 'I'll use the tool' or explain what you're doing.\n"
                        "6. ONLY return pure JSON when using a tool.\n\n"
                        "EXAMPLES:\n"
                        "For get_weather: {\"function\": \"get_weather\", \"arguments\": {\"location\": \"New York\"}}\n"
                        "For generate_image: {\"function\": \"generate_image\", \"arguments\": {\"prompt\": \"sunset over mountains\"}}\n\n"
                        "If NO tools are needed, respond normally with regular text."
                    )
            
            # Create messages with history
            messages = self._create_messages(
                user_prompt=user_prompt,
                system_prompt=enhanced_system_prompt,
                message_history=message_history
            )

            # Prepare parameters
            params = {
                **self.default_options,
                "temperature": temperature or self.default_options['temperature'],
                "num_predict": max_tokens or self.default_options['num_predict'],
                **kwargs
            }
            
            # Handle Ollama-specific format parameter (only if explicitly passed)
            format_param = kwargs.pop('format', None)
            if format_param:
                params['format'] = format_param

            # elif tools:
            #     params['format'] = 'json'

            # Make API call
            response = ollama.chat(
                model=model or self.model,
                messages=messages,
                options=params,
                stream=stream
            )

            # For streaming, just yield the content chunks
            if stream:
                return self._process_stream(response)
            
            # For normal mode, get the content as a string
            content = response["message"]["content"].strip()
            
            # If tools were provided, attempt to extract tool calls to match OpenAI format
            if tools:
                # Pre-process content to try to find JSON in noisy text
                if not content.startswith('{"function":') and not content.startswith('{"name":'):
                    # Look for patterns that indicate a JSON object is embedded in text
                    import re
                    json_pattern = r'({[\s\S]*?})'  # Non-greedy match for JSON-like structure
                    matches = re.findall(json_pattern, content)
                    if matches:
                        for potential_json in matches:
                            try:
                                # If this parses as JSON and has function/name/arguments, replace content with just this JSON
                                parsed = json.loads(potential_json)
                                if ('function' in parsed and 'arguments' in parsed) or ('name' in parsed and 'arguments' in parsed):
                                    logger.debug(f"[OllamaWrapper] Found likely tool call JSON in text, extracting: {potential_json[:100]}...")
                                    content = potential_json
                                    break
                            except json.JSONDecodeError:
                                continue
                return self._extract_tool_call_from_text(content, tools)
            
            # Otherwise return the raw content
            return content

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            import traceback
            traceback.print_exc()
            # Return error as text for robustness
            return f"An error occurred: {str(e)}"

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
                full_response = self.chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model,
                    stream=False,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    message_history=messages,
                    **kwargs
                )
                
                # Now full_response will be a string, not a dict
                if "<think>" in full_response and "</think>" in full_response:
                    parts = full_response.split("</think>", 1)
                    thinking = parts[0].replace("<think>", "").strip()
                    response_text = parts[1].strip()
                    return {"thinking": thinking, "response": response_text}
                else:
                    return full_response
            
        except Exception as e:
            logger.error(f"Error in reasoned completion: {e}")
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
            logger.info(f"[OllamaWrapper] Making vision request with model: {model or self.model}")
            
            # Handle URL or local path
            if image_path.startswith(('http://', 'https://')):
                logger.info(f"Fetching image from URL: {image_path}")
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
            
            # Use chat_completion for consistency - now returns raw text string
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
            logger.error(f"Error in vision analysis: {e}")
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
            logger.info(f"[OllamaWrapper] Making structured output request with model: {model or self.model}")

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
                # Use model_json_schema() instead of schema() (Pydantic V2)
                schema = output_class.model_json_schema()
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
                logger.error(f"JSON parse error at position {je.pos}: {je.msg}")
                logger.debug(f"Raw content: {content}")
                raise
            except ValidationError as ve:
                logger.error(f"Validation error: {ve}")
                raise

        except Exception as e:
            logger.error(f"Error in structured output: {e}")
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
                logger.info(f"[OllamaWrapper] Creating embeddings for {len(texts)} chunks with model: {the_model}")
            else:
                logger.info(f"[OllamaWrapper] Creating embedding with model: {the_model}")
            
            embeddings = []
            for t in texts:
                response = ollama.embeddings(
                    model=the_model,
                    prompt=t
                )
                embeddings.append(response.embedding)
            
            return np.array(embeddings, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"[OllamaWrapper] Embedding error: {str(e)}")
            raise

    def list_models(self) -> List[str]:
        """List available models, handling errors gracefully."""
        try:
            response = ollama.list()
            return [model.get("name", model.get("model", "unknown")) 
                   for model in response.models]
        except Exception as e:
            logger.error(f"[OllamaWrapper] Error listing models: {str(e)}")
            return []

    def pull_model(self, model_name: str):
        """Pull a model from Ollama's registry."""
        try:
            logger.info(f"[OllamaWrapper] Pulling model: {model_name}")
            
            for progress in ollama.pull(model_name):
                status = progress.status
                if status:
                    logger.info(f"[OllamaWrapper] Pull status: {status}")
        except Exception as e:
            logger.error(f"[OllamaWrapper] Pull error: {str(e)}")
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
                logger.warning(f"[OllamaWrapper] Model '{model}' not found. Attempting to pull...")
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

    def _process_stream(self, response):
        """Process streaming response from Ollama"""
        try:
            for chunk in response:
                if "message" in chunk and "content" in chunk["message"]:
                    yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"[OllamaWrapper] Error in stream processing: {e}")

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
            # Use model_dump() instead of dict() (Pydantic V2)
            print(f"Structured Output: {json.dumps(result.model_dump(), indent=2)}")
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
        """Test 4: Function Calling (Modified)"""
        print("\n🔧 Function Test (OpenAI-Compatible Format)")
        print("Testing tool call extraction from Ollama response to match OpenAI format")
        
        # Define two simple tool schemas for testing
        weather_tool_schema = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name and optional country code"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
        
        image_tool_schema = {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a text description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "Detailed description of the desired image"
                        },
                        "style": {
                            "type": "string",
                            "enum": ["realistic", "cartoon", "abstract", "digital-art"],
                            "description": "Visual style of the generated image"
                        },
                        "size": {
                            "type": "string",
                            "enum": ["small", "medium", "large"],
                            "description": "Size of the generated image"
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
        
        tools = [weather_tool_schema, image_tool_schema]
        
        try:
            # Test 1: Single tool call
            print("\n--- TESTING SINGLE TOOL CALL ---")
            print("Query: 'What's the weather like in Paris?'")
            
            # This should trigger a tool call for weather
            result1 = api.chat_completion(
                user_prompt="What's the weather like in Paris?",
                tools=tools,
                temperature=0.1,
                max_tokens=500,
            )
            
            print("\nResult type:", type(result1).__name__)
            
            # Check if we got an OpenAI-style tool call dict
            if isinstance(result1, dict) and "name" in result1 and "arguments" in result1:
                print("\n✅ Successfully extracted tool call:")
                print(f"Tool name: {result1['name']}")
                print(f"Arguments: {json.dumps(result1['arguments'], indent=2)}")
            else:
                print("\n❌ No tool call extracted, got regular response:")
                print(f"{str(result1)[:200]}...")
                
            # Test 2: Try to get multiple tool calls (harder to trigger reliably)
            print("\n--- TESTING POTENTIAL MULTIPLE TOOL CALLS ---")
            print("Query: 'I'm traveling to New York. What's the weather there, and can you generate an image of the skyline?'")
            
            # This might trigger multiple tool calls, but it's model-dependent
            result2 = api.chat_completion(
                user_prompt="I'm traveling to New York. What's the weather there, and can you generate an image of the skyline?",
                tools=tools,
                temperature=0.1,
                max_tokens=500,
            )
            
            print("\nResult type:", type(result2).__name__)
            
            if isinstance(result2, list) and all(isinstance(tc, dict) and "name" in tc for tc in result2):
                print("\n✅ Successfully extracted multiple tool calls:")
                for i, tc in enumerate(result2):
                    print(f"\nTool call #{i+1}:")
                    print(f"Tool name: {tc['name']}")
                    print(f"Arguments: {json.dumps(tc['arguments'], indent=2)}")
            elif isinstance(result2, dict) and "name" in result2 and "arguments" in result2:
                print("\n✅ Extracted single tool call (model chose one tool):")
                print(f"Tool name: {result2['name']}")
                print(f"Arguments: {json.dumps(result2['arguments'], indent=2)}")
            else:
                print("\n❌ No tool calls extracted, got regular response:")
                print(f"{str(result2)[:200]}...")
            
            # Test 3: Test with direct JSON response (simulate Ollama output)
            print("\n--- TESTING EXTRACTION FROM DIRECT JSON ---")
            
            # Create a mock JSON response that our extraction should handle
            mock_json_response = """
            ```json
            {
              "function": "get_weather",
              "arguments": {
                "location": "Tokyo",
                "unit": "celsius"
              }
            }
            ```

            Here's the weather information for Tokyo.
            """
            
            # Use the internal extraction method directly
            extracted = api._extract_tool_call_from_text(mock_json_response, tools)
            
            if isinstance(extracted, dict) and "name" in extracted and "arguments" in extracted:
                print("\n✅ Successfully extracted tool call from JSON block:")
                print(f"Tool name: {extracted['name']}")
                print(f"Arguments: {json.dumps(extracted['arguments'], indent=2)}")
            else:
                print("\n❌ Failed to extract from mock JSON response")
                
            # Test 4: Test with multiple sequential JSON objects (common Ollama response pattern)
            print("\n--- TESTING EXTRACTION FROM MULTIPLE SEQUENTIAL JSON OBJECTS ---")
            
            # Create a mock response with multiple sequential JSON objects
            mock_multiple_json = """
            {"function": "get_weather", "arguments": {"location": "London", "unit": "celsius"}}
            {"function": "generate_image", "arguments": {"prompt": "London skyline with Tower Bridge", "style": "realistic", "size": "large"}}
            """
            
            # Use the internal extraction method directly
            extracted_multiple = api._extract_tool_call_from_text(mock_multiple_json, tools)
            
            if isinstance(extracted_multiple, list) and len(extracted_multiple) == 2:
                print("\n✅ Successfully extracted multiple tool calls from sequential JSONs:")
                for i, tc in enumerate(extracted_multiple):
                    print(f"\nTool call #{i+1}:")
                    print(f"Tool name: {tc['name']}")
                    print(f"Arguments: {json.dumps(tc['arguments'], indent=2)}")
            else:
                print("\n❌ Failed to extract multiple tool calls from sequential JSONs:")
                print(f"Result type: {type(extracted_multiple).__name__}")
                print(f"Content: {extracted_multiple}")
                
            # Test 5: Test with tool_code format (Ollama's another common pattern)
            print("\n--- TESTING EXTRACTION FROM TOOL_CODE FORMAT ---")
            
            # Create a mock response with the tool_code format
            mock_tool_code = """
            I'll use the web_crawl and generate_image tools to fulfill your request.
            
            ```tool_code
            tool_code: web_crawl, query="latest news on AI technology", sources="DuckDuckGo", num_results=3
            ```
            
            After finding information, I'll generate an image:
            
            ```tool_code
            {"function": "generate_image", "arguments": {"prompt": "Advanced AI robot with futuristic background"}}
            ```
            """
            
            # Use the internal extraction method directly
            extracted_tool_code = api._extract_tool_call_from_text(mock_tool_code, tools)
            
            if isinstance(extracted_tool_code, list) and len(extracted_tool_code) >= 1:
                print("\n✅ Successfully extracted tool calls from tool_code format:")
                for i, tc in enumerate(extracted_tool_code):
                    print(f"\nTool call #{i+1}:")
                    print(f"Tool name: {tc['name']}")
                    print(f"Arguments: {json.dumps(tc['arguments'], indent=2)}")
            elif isinstance(extracted_tool_code, dict) and "name" in extracted_tool_code:
                print("\n✅ Successfully extracted single tool call from tool_code format:")
                print(f"Tool name: {extracted_tool_code['name']}")
                print(f"Arguments: {json.dumps(extracted_tool_code['arguments'], indent=2)}")
            else:
                print("\n❌ Failed to extract tool calls from tool_code format:")
                print(f"Result type: {type(extracted_tool_code).__name__}")
                print(f"Content: {extracted_tool_code}")
                
        except Exception as e:
            print(f"\n❌ Error during function call testing: {e}")
            import traceback
            traceback.print_exc()

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
                max_tokens=1000
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
        "4": ("Function Calling (Modified)", run_function_calling_test),
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
            model="gemma3:4b",
            auto_pull=True,
            system_message="You are a specialized AI assistant."
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