"""
textGen.py

Unified TextGen interface integrating OpenAI API, memory management, 
retrieval augmented generation (RAG), and tools functionality.

This module serves as the central integration point for AI text generation,
providing a simplified interface for accessing all functionality with proper context
and memory management.

---
Implement Ollama.

Tool usage (do not repeat double usage of tools)


"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Union, Any, Iterator
from pathlib import Path
import subprocess
import time
import inspect
from typing import get_type_hints, Type
import traceback # Ensure traceback is imported

# Configure logging - MAKE IT DO NOTHING
logging.basicConfig(level=logging.CRITICAL)

# Create dummy logger functions that do nothing
def noop(*args, **kwargs):
    pass

# Override logger methods to do nothing
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)
logger.info = noop
logger.warning = noop
logger.error = noop
logger.debug = noop
logger.critical = noop

# Disable all existing loggers
for name, _logger in logging.Logger.manager.loggerDict.items():
    if isinstance(_logger, logging.Logger):
        _logger.setLevel(logging.CRITICAL)
        _logger.info = noop
        _logger.warning = noop
        _logger.error = noop
        _logger.debug = noop
        _logger.critical = noop

# Add path for importing modules from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import required modules
try:
    # Try both import methods silently
    try:
        from II_textGen.tools import Tools
        from II_textGen.memory import Memory
        from II_textGen.rag import RAG
    except ImportError:
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
                 provider: str = "openai",  # New parameter to select provider
                 openai_api_key: Optional[str] = None, 
                 replicate_api_token: Optional[str] = None, 
                 default_model: Optional[str] = None,  # Make optional so we can set based on provider
                 short_term_limit: int = 8000, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 agent_name: Optional[str] = None):  # New parameter for agent-specific memory
        """
        Initialize TextGen with all required components.
        
        Args:
            provider: Which LLM provider to use ("openai" or "ollama")
            openai_api_key: Optional OpenAI API key
            replicate_api_token: Optional Replicate API token
            default_model: Default model to use (provider-specific if not specified)
            short_term_limit: Token limit for short-term memory
            chunk_size: Size of chunks for RAG
            chunk_overlap: Overlap between chunks for RAG
            agent_name: Name of the agent this TextGen instance belongs to (for memory isolation)
        """
        # Initialize tools with API keys
        self.tools = Tools(
            openai_api_key=openai_api_key,
            replicate_api_token=replicate_api_token
        )
        
        # Store agent name for logging and reference
        self.agent_name = agent_name
        
        # Set the provider
        self.provider = provider.lower()
        
        # Set default models based on provider
        if default_model is None:
            if self.provider == "openai":
                default_model = "gpt-4o-mini"
            elif self.provider == "ollama": 
                default_model = "gemma3:4b"
            else:
                raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'.")
        
        # Store default model
        self.default_model = default_model
        
        # Initialize the appropriate LLM wrapper
        if self.provider == "openai":
            # Import at runtime to avoid circular imports
            from src.I_integrations.openai_API import OpenAIWrapper
            self.llm = OpenAIWrapper(
                model=default_model,
                api_key=openai_api_key
            )
            logger.info(f"Initialized OpenAI provider with model: {default_model}" + (f" for agent: {agent_name}" if agent_name else ""))
        elif self.provider == "ollama":
            # Import at runtime to avoid circular imports
            from src.I_integrations.ollama_API import OllamaWrapper
            self.llm = OllamaWrapper(
                model=default_model,
                auto_pull=True  # Auto-pull models if not available
            )
            logger.info(f"Initialized Ollama provider with model: {default_model}" + (f" for agent: {agent_name}" if agent_name else ""))
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'openai' or 'ollama'.")
        
        # Initialize memory, passing the initialized LLM API instance and agent_name for memory isolation
        self.memory = Memory(llm_api=self.llm, short_term_limit=short_term_limit, agent_name=agent_name)

        # Initialize RAG instances for different contexts
        self.ltm_rag = RAG(llm_api=self.llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.session_rag = RAG(llm_api=self.llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Cache state trackers
        self._indexed_ltm_hash = None
        self._indexed_session_context_hash = None

        # Embedding function
        self.embedding = lambda text: self.llm.create_embeddings(text)
        
        # Register available tools
        self._register_tools()
        
        # Initial indexing of LTM (if memory loaded previously or has initial state)
        self._conditionally_index_ltm()

    def _conditionally_index_ltm(self):
        """Checks if LTM needs indexing and performs it."""
        insights = self.memory.retrieve_long_term()
        current_ltm_hash = hash(tuple(sorted(insights))) if insights else None

        if current_ltm_hash != self._indexed_ltm_hash:
            logger.info(f"Long-term memory changed (hash: {self._indexed_ltm_hash} -> {current_ltm_hash}). Re-indexing LTM RAG.")
            self.ltm_rag.clear()
            if insights:
                try:
                    # Assuming index_context handles embeddings internally or uses a pre-set function
                    self.ltm_rag.index_context(
                        documents=insights, 
                        document_ids=[f"ltm_{i}" for i in range(len(insights))]
                    )
                    self._indexed_ltm_hash = current_ltm_hash
                    logger.info(f"Successfully indexed {len(insights)} LTM insights.")
                except Exception as e:
                    logger.error(f"Failed to index long-term memory: {e}")
                    self._indexed_ltm_hash = None # Ensure re-indexing attempt next time
            else:
                 self._indexed_ltm_hash = None # No insights, hash is None
        else:
            logger.info("Long-term memory unchanged. LTM RAG index is up-to-date.")

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
    
    def _prepare_prompts(self, user_prompt: str, system_context: str = None, context: str = None, context_char_limit: int = 2500, top_k: int = 3) -> tuple:
        """
        Prepare final system and user messages with context, using separate RAG instances 
        for long-term memory and transient session context with caching.
        
        Args:
            user_prompt: The main user prompt
            system_context: Additional context to add to system message
            context: Additional context to add to user message
            context_char_limit: Character limit above which context is processed via RAG
            top_k: Number of relevant chunks to retrieve via RAG for each source
            
        Returns:
            Tuple of (final_system, final_user) with all context included
        """
        final_system_parts = []
        final_user_parts = []

        # --- 1. Long-Term Memory Context --- 
        self._conditionally_index_ltm() # Ensures ltm_rag is up-to-date
        ltm_results = []
        if self._indexed_ltm_hash: # Only search if LTM is indexed
            try:
                logger.info("Searching LTM RAG for relevant context.")
                ltm_results = self.ltm_rag.search_index(query=user_prompt, k=top_k)
            except Exception as e:
                 logger.error(f"Error searching LTM RAG: {e}")

        if ltm_results:
            ltm_context_str = "\n---\n".join([result["text"] for result in ltm_results])
            final_system_parts.append("Relevant Long-Term Memory Context:\n" + ltm_context_str)
            logger.info(f"Retrieved {len(ltm_results)} relevant LTM chunks.")
        else:
            logger.info("No relevant context found in Long-Term Memory.")

        # --- 2. Session Context (System + User) --- 
        large_sys_ctx = system_context if system_context and len(system_context) > context_char_limit else None
        large_usr_ctx = context if context and len(context) > context_char_limit else None
        
        # Calculate hash for current large session contexts
        current_session_hash = hash((large_sys_ctx, large_usr_ctx))

        session_docs = []
        session_doc_ids = []
        if large_sys_ctx:
            session_docs.append(large_sys_ctx)
            session_doc_ids.append("sys_ctx_0")
        if large_usr_ctx:
            session_docs.append(large_usr_ctx)
            session_doc_ids.append("usr_ctx_0")

        # Conditionally index session context
        if current_session_hash != self._indexed_session_context_hash:
            logger.info(f"Session context changed (hash: {self._indexed_session_context_hash} -> {current_session_hash}). Re-indexing Session RAG.")
            self.session_rag.clear()
            if session_docs:
                try:
                    # Assuming index_context handles embeddings internally or uses a pre-set function
                    self.session_rag.index_context(
                        documents=session_docs,
                        document_ids=session_doc_ids
                    )
                    self._indexed_session_context_hash = current_session_hash
                    logger.info(f"Successfully indexed {len(session_docs)} large session context document(s).")
                except Exception as e:
                    logger.error(f"Failed to index session context: {e}")
                    self._indexed_session_context_hash = None # Ensure re-indexing attempt next time
            else:
                self._indexed_session_context_hash = None # No large session context
        else:
             logger.info("Session context unchanged. Session RAG index is up-to-date.")

        # Search session context RAG
        session_results = []
        if self._indexed_session_context_hash: # Only search if session RAG is indexed
            try:
                logger.info("Searching Session RAG for relevant context.")
                # Fetch slightly more results to allow filtering by source
                session_results = self.session_rag.search_index(query=user_prompt, k=top_k * 2) 
            except Exception as e:
                logger.error(f"Error searching Session RAG: {e}")
        
        # --- 3. Assemble Final Prompts --- 
        
        # Add small system context directly
        if system_context and not large_sys_ctx:
            logger.info("Adding small system context directly.")
            final_system_parts.append(system_context)
        
        # Add relevant large system context from RAG results
        sys_ctx_rag_results = [res["text"] for res in session_results if res["metadata"].get("document_id", "").startswith("sys_ctx")]
        if sys_ctx_rag_results:
            final_system_parts.append("Relevant System Context:\n" + "\n---\n".join(sys_ctx_rag_results[:top_k]))
            logger.info(f"Adding {len(sys_ctx_rag_results[:top_k])} relevant system context chunks from Session RAG.")

        # Add small user context directly
        if context and not large_usr_ctx:
            logger.info("Adding small user context directly.")
            final_user_parts.append(context)

        # Add relevant large user context from RAG results
        usr_ctx_rag_results = [res["text"] for res in session_results if res["metadata"].get("document_id", "").startswith("usr_ctx")]
        if usr_ctx_rag_results:
            final_user_parts.append("Relevant User Context:\n" + "\n---\n".join(usr_ctx_rag_results[:top_k]))
            logger.info(f"Adding {len(usr_ctx_rag_results[:top_k])} relevant user context chunks from Session RAG.")

        # Add the main user prompt
        final_user_parts.append(user_prompt)

        # Combine parts
        final_system = "\n\n".join(final_system_parts).strip()
        final_user = "\n\n".join(final_user_parts).strip()
        
        return final_system, final_user
    
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
                    tool_schema = self.convert_function_to_schema(tool_func)
                    tools.append(tool_schema)
            logger.info(f"Prepared {len(tools)} tools for potential use: {', '.join(tool_names)}")
        
        # If streaming is requested, handle normally as tools can't be used with streaming
        if kwargs.get('stream', False):
            logger.info("Streaming requested, bypassing tool processing")
            response = self.llm.chat_completion(
                user_prompt=final_user,
                system_prompt=combined_system,
                message_history=message_history,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,  # Include tools in case the API supports streaming with tools
                **kwargs
            )
            return response
        
        # Make initial API call with tools
        initial_response = self.llm.chat_completion(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            **kwargs
        )
        
        # Check if the response is a tool call
        if isinstance(initial_response, dict) and "name" in initial_response and "arguments" in initial_response:
            logger.info(f"Tool call detected: {initial_response['name']}")
            
            # Execute the tool
            tool_result = self._execute_tool_call(initial_response["name"], initial_response["arguments"])
            
            # Add original prompt, tool call, and result to message history for final completion
            tool_messages = [
                {"role": "user", "content": final_user},
                {"role": "assistant", "content": f"I'll use the {initial_response['name']} tool to help answer that."},
                {"role": "system", "content": f"Tool '{initial_response['name']}' was called with arguments: {json.dumps(initial_response['arguments'])}. Result: {tool_result}"}
            ]
            
            # Make a follow-up call with the tool result
            follow_up_prompt = "Please provide your response based on the tool result."
            final_response = self.llm.chat_completion(
                user_prompt=follow_up_prompt,
                system_prompt=combined_system,
                message_history=message_history + tool_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Optionally include tool call information in the stored interaction
            if store_interaction:
                tool_info = f"Tool '{initial_response['name']}' was used with arguments: {json.dumps(initial_response['arguments'])}\nResult: {tool_result}\n\n"
                self.memory.save_short_term(system_prompt, user_prompt, tool_info + str(final_response))
            
            return final_response
        
        # Check if response is a list of tool calls
        elif isinstance(initial_response, list) and all(isinstance(tc, dict) and "name" in tc for tc in initial_response):
            logger.info(f"Multiple tool calls detected: {', '.join(tc['name'] for tc in initial_response)}")
            
            tool_messages = [{"role": "user", "content": final_user}]
            tool_info = []
            
            # Execute each tool in sequence
            for i, tool_call in enumerate(initial_response):
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]
                
                logger.info(f"Executing tool call {i+1}/{len(initial_response)}: {tool_name}")
                tool_result = self._execute_tool_call(tool_name, tool_args)
                
                # Add each tool call and result to messages
                tool_messages.append({
                    "role": "assistant", 
                    "content": f"I'll use the {tool_name} tool to help answer part of your question."
                })
                tool_messages.append({
                    "role": "system", 
                    "content": f"Tool '{tool_name}' was called with arguments: {json.dumps(tool_args)}. Result: {tool_result}"
                })
                
                # Add to tool info for storing in memory
                tool_info.append(f"Tool '{tool_name}' was used with arguments: {json.dumps(tool_args)}\nResult: {tool_result}")
            
            # Make a final call with all tool results
            follow_up_prompt = "Please provide your final response based on all the tool results."
            final_response = self.llm.chat_completion(
                user_prompt=follow_up_prompt,
                system_prompt=combined_system,
                message_history=message_history + tool_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Store the interaction with tool info
            if store_interaction:
                self.memory.save_short_term(system_prompt, user_prompt, 
                                           "\n\n".join(tool_info) + "\n\n" + str(final_response))
            
            return final_response
        
        # Regular response without tool calls
        if store_interaction:
            self.memory.save_short_term(system_prompt, user_prompt, initial_response)
        
        return initial_response

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
            
        # Get structured output, using higher max_tokens for structured output to avoid truncation
        # Default to 1000 tokens if not specified, which should be enough for most structured responses
        structured_max_tokens = max_tokens or 4000
        
        response = self.llm.structured_output(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=structured_max_tokens,
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
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
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
            stream: Whether to stream the response
            
        Returns:
            Generated response text or stream iterator
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
            
        # Use vision_analysis with stream parameter
        response = self.llm.vision_analysis(
            image_path=image_url,
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        # Store the interaction if requested and not streaming
        if store_interaction and not stream:
            self.memory.save_short_term(system_prompt, user_prompt, response)
        elif stream and store_interaction:
            # We'll need to handle storing the interaction after streaming completes
            # by having the caller collect the streamed content
            logger.info("Note: When streaming, interaction must be stored by the caller")
        
        return response

    def reasoned_completion(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant that shows logical reasoning before answering.",
        temperature: float = None,
        max_tokens: int = None,
        context: str = None,
        system_context: str = None,
        store_interaction: bool = True,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Iterator[str]]:
        """
        Generate a completion with explicit reasoning steps.
        
        Args:
            user_prompt: The main user prompt
            system_prompt: System prompt for the LLM
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            context: Additional context for user message
            system_context: Additional context for system message
            store_interaction: Whether to store the interaction in memory
            stream: Whether to stream the response
            
        Returns:
            Generated response with reasoning included (or iterator if streaming)
        """
        # Get message history
        message_history = self.memory.retrieve_short_term_formatted()
            
        # Prepare context-enriched prompts
        final_system, final_user = self._prepare_prompts(user_prompt, system_context, context)
        
        # Combine the base system_prompt with additional context
        combined_system = system_prompt
        if final_system:
            combined_system += "\n" + final_system
        
        # Call appropriate LLM API's reasoned_completion method
        response = self.llm.reasoned_completion(
            user_prompt=final_user,
            system_prompt=combined_system,
            message_history=message_history,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            **kwargs
        )
        
        # Store the interaction if requested and not streaming
        if store_interaction and not stream:
            self.memory.save_short_term(system_prompt, user_prompt, response)
        elif stream and store_interaction:
            # We'll need to handle storing the interaction after streaming completes
            # by having the caller collect the streamed content
            logger.info("Note: When streaming, interaction must be stored by the caller")
        
        return response
        
    def clear_memory(self) -> None:
        """Clear all conversation history and RAG caches."""
        self.memory.clear()
        self.ltm_rag.clear()
        self.session_rag.clear()
        self._indexed_ltm_hash = None
        self._indexed_session_context_hash = None
        logger.info("Conversation history and RAG indexes cleared")
        
    def save_memory(self, filename: Optional[str] = None) -> str:
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
        
    def load_LTM(self, filename: str) -> bool:
        """
        Load conversation history from a file and re-index LTM.
        
        Args:
            filename: Name of the file to load
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not filename.lower().endswith('.json'):
                filename += '.json'
            
            self.memory.load(filename)
            logger.info(f"Loaded memory from: {filename}")
            
            # Re-index LTM after loading
            logger.info("Triggering LTM re-indexing after load.")
            self._conditionally_index_ltm()
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            return False

    def convert_function_to_schema(self, func) -> Dict:
        """
        Convert a Python function to OpenAI's function schema format.
        This schema is widely compatible.

        Args:
            func: The function to convert.

        Returns:
            A dictionary representing the function schema.
        """
        try:
            # Get function signature and docstring
            sig = inspect.signature(func)
            # Use inspect.getdoc for cleaner docstring handling
            doc = inspect.getdoc(func) or ""
            type_hints = get_type_hints(func)

            # Create schema structure (OpenAI format)
            schema = {
                "name": func.__name__,
                "description": doc.split("\n\n")[0], # Use first paragraph for description
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }

            # Basic Python type to JSON schema type mapping
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
                # Add mappings for Any and None if needed, defaulting to string
                Any: "string",
                type(None): "string"
            }

            param_docs = {}
            if doc:
                # Simple parsing for ':param <name>: <description>'
                for line in doc.split('\n'):
                    if line.strip().startswith(':param'):
                        parts = line.strip().split(':', 2)
                        if len(parts) == 3:
                             param_name = parts[1].replace('param', '').strip()
                             param_desc = parts[2].strip()
                             param_docs[param_name] = param_desc

            # Add parameters to schema
            for param_name, param in sig.parameters.items():
                # Skip self/cls parameters if they exist
                if param_name in ['self', 'cls']:
                    continue

                param_type_hint = type_hints.get(param_name, str)
                # Handle Optional types (e.g., Optional[str])
                origin_type = getattr(param_type_hint, "__origin__", None)
                if origin_type is Union:
                     # Get the first type argument that isn't NoneType
                     non_none_type = next((t for t in getattr(param_type_hint, "__args__", ()) if t is not type(None)), str)
                     param_type = non_none_type
                elif origin_type:
                     param_type = origin_type # Handle List[str], Dict[str, int] etc. - map the container
                else:
                     param_type = param_type_hint

                json_type = type_map.get(param_type, "string") # Default to string if type not mapped

                param_schema = {
                    "type": json_type,
                    # Use parsed docstring description or empty string
                    "description": param_docs.get(param_name, "")
                }

                # Handle enums for list types if specified in type hints (conceptual)
                # if json_type == "array" and hasattr(param_type_hint, "__args__"):
                #    item_type = getattr(param_type_hint, "__args__")[0]
                #    param_schema["items"] = {"type": type_map.get(item_type, "string")}

                # Add parameter details to the schema
                schema["parameters"]["properties"][param_name] = param_schema

                # Add to required list if no default value
                if param.default == param.empty:
                    schema["parameters"]["required"].append(param_name)

            # Return the schema in the format expected by OpenAI tools
            # (Other LLMs might adapt to this common format)
            return {
                "type": "function",
                "function": schema
            }

        except Exception as e:
            logger.error(f"Error converting function '{func.__name__}' to schema: {e}\n{traceback.format_exc()}")
            # Return a basic schema as fallback
            return {
                "type": "function",
                "function": {
                    "name": func.__name__,
                    "description": f"Error retrieving schema for {func.__name__}.",
                    "parameters": {"type": "object", "properties": {}}
                }
            }

    def _execute_tool_call(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """
        Execute a specified tool with given arguments.

        Args:
            tool_name: The name of the tool function to execute.
            tool_args: A dictionary of arguments for the tool function.

        Returns:
            The result of the tool execution, or an error message string.
        """
        if tool_name in self.available_tools:
            tool_func = self.available_tools[tool_name]
            try:
                # Execute the tool function with the provided arguments
                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")
                result = tool_func(**tool_args)
                logger.info(f"Successfully executed tool '{tool_name}'. Raw result type: {type(result)}")

                # Convert result to string if needed for message history consistency
                # Handle common non-serializable types carefully
                if isinstance(result, (bytes,)): # Example: Handle bytes directly
                    result = result.decode('utf-8', errors='ignore')
                elif not isinstance(result, (str, int, float, bool, list, dict, type(None))):
                    try:
                        # Attempt standard JSON serialization first
                        result_str = json.dumps(result)
                    except TypeError:
                         # Fallback to string representation if JSON fails
                         logger.warning(f"Result for tool '{tool_name}' is not JSON serializable, falling back to str(). Type: {type(result)}")
                         result_str = str(result)
                    result = result_str # Assign the serialized/stringified result

                logger.debug(f"Formatted result for tool '{tool_name}': {result}")
                return result
            except Exception as e:
                logger.error(f"Error executing tool '{tool_name}' with args {tool_args}: {e}\n{traceback.format_exc()}")
                # Provide a more detailed error message back to the LLM
                error_details = f"Error executing tool '{tool_name}': {str(e)}. Check logs for details."
                return error_details
        else:
            logger.warning(f"Attempted to call unknown tool: {tool_name}")
            return f"Error: Tool '{tool_name}' not found in available tools: {list(self.available_tools.keys())}"


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
        print("1. Basic Chat Completion (with Streaming)")
        print("2. Multi-Tool Completion")
        print("3. Structured Output")
        print("4. Reasoned Completion")
        print("5. Vision Analysis")
        print("6. Memory & History Management")
        print("7. RAG Caching Efficiency Test")
        print("P. Change Provider (OpenAI/Ollama)")
        print("0. Exit")
        
        choice = prompt("Select a test to run", "0")
        
        if choice == "0":
            print("Exiting test suite.")
            break
            
        elif choice == "1":
            # Demo 1: Basic Chat Completion (with Streaming)
            print("\n" + "="*50)
            print("💬 TEST 1: BASIC CHAT COMPLETION (STREAMING)")
            print("="*50)
            custom_prompt = prompt("Enter your prompt (or use default)", "Explain the concept of API integration in one short paragraph.")
            
            print("\n" + "-"*50)
            print("📋 STREAMING RESPONSE:")
            print("-"*50)
            try:
                stream = text_gen.chat_completion(
                    user_prompt=custom_prompt,
                    system_prompt="You are a concise technical writer. Keep all responses under 150 tokens in exactly one paragraph.",
                    max_tokens=200,
                    stream=True  # Enable streaming
                )
                
                # Process the stream
                full_response = ""
                for chunk in stream:
                    print(chunk, end="", flush=True)
                    full_response += chunk # Accumulate the full response if needed later
                print() # Newline after stream ends
                
            except Exception as e:
                 print(f"\n❌ Error during streaming chat completion: {e}")
                 full_response = "Error occurred during streaming."
            
            print("\n" + "-"*50)
            print("Streaming complete.")
            print("-"*50)
            
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
            print("- get_current_datetime: Get current date and time")
            print("- get_weather: Get current weather for a location")
            print("- get_forecast: Get weather forecast for a location")
            print("- text_to_speech: Convert text to spoken audio")
            print("- get_news: Get news articles related to a topic")
            
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
                    tool_names=["web_crawl", "generate_image", "get_current_datetime", "get_weather", "get_forecast", "text_to_speech", "get_news"],
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
            topic = prompt("Enter a topic for structured analysis", "whatever json")
            
            # Define a simple output schema for consistent JSON structure
            # IMPORTANT: Must include additionalProperties: False for json_schema mode
            output_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "components": {"type": "array", "items": {"type": "string"}},
                    "inputs": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "challenges": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "components", "inputs", "outputs", "challenges"], # Also specify required fields
                "additionalProperties": False # REQUIRED by the API for json_schema type
            }
            
            structured = text_gen.structured_output(
                user_prompt=f"Create a simple JSON output describing a {topic}. Include name, components, inputs, outputs, and challenges as arrays of strings.",
                system_prompt="You are a systems analyst. Return only a valid JSON object with the requested fields.",
                max_tokens=500, # Adjusted max_tokens for potentially larger structured output
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
                    system_prompt="You are a helpful assistant that shows logical reasoning before answering.",
                    temperature=0.7,
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
                use_streaming = prompt("Stream the response? (y/n)", "n").lower() == "y"
                
                if use_streaming:
                    print("\n" + "-"*50)
                    print("📋 STREAMING VISION ANALYSIS:")
                    print("-"*50)
                    
                    stream = text_gen.vision_analysis(
                        image_url=image_url,
                        user_prompt=custom_prompt,
                        system_prompt="You are a concise image analyst. Limit response to 150 tokens in one paragraph.",
                        max_tokens=150,
                        stream=True
                    )
                    
                    # Process stream
                    full_response = ""
                    for chunk in stream:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                    print("\n")
                    
                    print("-"*50)
                    print("Streaming vision analysis complete.")
                    print("-"*50)
                else:
                    # Regular non-streaming vision analysis
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
            
            # Check current memory state using the retrieval method
            print("Current memory state:")
            # Use the public method to get formatted history
            history_data = text_gen.memory.retrieve_short_term_formatted() 
            memory_size = len(history_data) 
            print(f"Messages in memory: {memory_size}")
            
            # Option to view history
            if memory_size > 0 and prompt("View current chat history? (y/n)", "n").lower() == "y":
                for i, entry in enumerate(history_data): # Iterate through the retrieved data
                    print(f"\nMessage {i+1}:")
                    print(f"Role: {entry.get('role', 'unknown')}")
                    content = entry.get('content', '')
                    # Handle potential list content (e.g., from vision)
                    if isinstance(content, list): 
                        content_str = json.dumps(content) 
                    else:
                        content_str = str(content)
                        
                    if len(content_str) > 100:
                        content_str = content_str[:100] + "..."
                    print(f"Content: {content_str}") # Print the potentially truncated string
            
            # Save memory
            if prompt("Save current memory to file? (y/n)", "y").lower() == "y":
                memory_path = text_gen.save_memory()
                print(f"📁 History saved to: {memory_path}")
            
        elif choice == "7":
            # Demo 7: RAG Caching Efficiency Test
            print("\n" + "="*50)
            print("⚡ TEST 7: RAG CACHING EFFICIENCY TEST")
            print("="*50)
            
            # Create a large test context > context_char_limit
            print("Setting up test contexts...")
            large_system_context = """
            # Comprehensive AI System Documentation
            
            This document provides an overview of our AI system architecture, components, 
            and integration methods. It serves as a reference for developers, system architects,
            and technical leaders.
            
            ## System Architecture
            
            The architecture follows a modular design with the following key components:
            
            1. Data Ingestion Layer - Handles raw input processing
            2. Feature Engineering Pipeline - Transforms raw data into ML-ready features
            3. Model Training Infrastructure - Manages model development lifecycle
            4. Inference Engine - Optimized for low-latency predictions
            5. Monitoring & Feedback Loop - Tracks model performance and drift
            
            ### Data Ingestion
            
            The data ingestion layer supports multiple input formats including JSON, CSV, 
            Parquet, and unstructured text. It performs initial validation, schema enforcement,
            and basic cleaning operations. Data is then stored in a staging area before proceeding
            to feature engineering.
            
            ### Feature Engineering
            
            The feature engineering pipeline includes:
            
            - Numerical feature processing (scaling, normalization, outlier handling)
            - Categorical encoding (one-hot, target, frequency encoding)
            - Text processing (tokenization, embedding generation)
            - Time-series feature extraction (lags, windows, Fourier features)
            
            All transformations are versioned and reproducible, with configuration managed
            through a central feature registry.
            
            ### Model Training
            
            The training infrastructure supports:
            
            - Distributed training across GPU clusters
            - Hyperparameter optimization via Bayesian search
            - Curriculum learning for complex tasks
            - Transfer learning from foundation models
            - Ensemble methods and model distillation
            
            Models are evaluated using a comprehensive suite of metrics appropriate for each
            task type, with automatic A/B testing against baseline models.
            
            ### Inference Engine
            
            The inference engine is designed for:
            
            - Horizontal scaling to handle traffic spikes
            - Model versioning and canary deployments
            - Optimized inference with ONNX Runtime and TensorRT
            - Batch and real-time prediction endpoints
            - Caching for frequently requested predictions
            
            ### Monitoring
            
            The monitoring system tracks:
            
            - Model performance metrics (accuracy, latency, throughput)
            - Data drift and concept drift indicators
            - Resource utilization (CPU, GPU, memory)
            - Prediction explanations and feature importance
            - Anomalous inputs and outputs
            
            ## Integration Methods
            
            The system supports integration via:
            
            1. REST APIs with OpenAPI specification
            2. gRPC endpoints for high-throughput applications
            3. Message queue consumers (Kafka, RabbitMQ)
            4. Batch processing through scheduled jobs
            5. Embedded deployment for edge devices
            
            ## Deployment Options
            
            The system can be deployed as:
            
            - Containerized microservices on Kubernetes
            - Serverless functions for lightweight components
            - Virtual machines for resource-intensive workloads
            - On-premise appliance with reduced external dependencies
            
            ## Security Considerations
            
            The system implements:
            
            - Role-based access control for all endpoints
            - Encryption of sensitive data at rest and in transit
            - Audit logging of all prediction requests and model updates
            - Regular vulnerability scanning and dependency updates
            - Privacy-preserving techniques for sensitive datasets
            
            ## Compliance Features
            
            Built-in support for:
            
            - Explainable predictions with SHAP and LIME
            - Data lineage tracking through the entire pipeline
            - Bias detection and mitigation tools
            - Automated documentation generation for audits
            - Model cards for transparency
            """
            
            large_user_context = """
            # Technical Interview Questions on AI Systems
            
            ## Machine Learning Fundamentals
            
            1. Explain the bias-variance tradeoff and how it impacts model selection.
            2. Compare and contrast L1 and L2 regularization and their effects on model parameters.
            3. Describe the advantages and limitations of ensemble methods like Random Forests.
            4. How do you handle class imbalance in classification problems?
            5. Explain the concept of gradient descent and its variants.
            
            ## Deep Learning
            
            1. Describe the architecture and applications of Transformer models.
            2. What are attention mechanisms and how do they improve model performance?
            3. Explain techniques to prevent overfitting in deep neural networks.
            4. Compare CNN, RNN, and Transformer architectures for different tasks.
            5. How do you debug a neural network that's not learning properly?
            
            ## Natural Language Processing
            
            1. Explain how word embeddings capture semantic relationships.
            2. Describe the evolution from RNN-based to Transformer-based language models.
            3. What techniques are effective for few-shot learning in NLP?
            4. How do you evaluate the quality of a text generation model?
            5. Explain approaches to handle multilingual NLP tasks.
            
            ## Computer Vision
            
            1. Explain the architecture of a modern object detection system.
            2. How do GANs work and what are their applications in computer vision?
            3. Describe techniques for semantic segmentation.
            4. What approaches help deep learning models generalize to varied lighting conditions?
            5. Explain self-supervised learning methods in computer vision.
            
            ## MLOps and System Design
            
            1. Design a real-time recommendation system architecture.
            2. How would you implement a model monitoring system to detect drift?
            3. Explain approaches to versioning for data, models, and features.
            4. Describe a system for secure deployment of models with PHI/PII data.
            5. How would you design an A/B testing framework for ML models?
            
            ## Reinforcement Learning
            
            1. Explain the difference between policy-based and value-based methods.
            2. How does the exploration-exploitation tradeoff impact RL system design?
            3. Describe approaches to handle sparse rewards in RL.
            4. What are model-based RL methods and when are they preferred?
            5. Explain how multi-agent RL differs from single-agent approaches.
            
            ## Ethics and Responsible AI
            
            1. How would you detect and mitigate bias in an ML system?
            2. Describe approaches to make model predictions more interpretable.
            3. What techniques help ensure privacy in ML systems using sensitive data?
            4. How would you design a governance framework for responsible AI deployment?
            5. Explain the concept of AI safety and methods to ensure safe model behavior.
            """
            
            # First query WITHOUT context to establish baseline
            print("\n--- BASELINE QUERY (NO LARGE CONTEXTS) ---")
            start_time = time.time()
            response = text_gen.chat_completion(
                user_prompt="What are the key components of a robust AI system?",
                system_prompt="You are a helpful technical assistant.",
                max_tokens=100
            )
            print(f"Response time without contexts: {time.time() - start_time:.2f} seconds")
            print(f"Response: {response[:100]}...")
            
            # First query WITH both large contexts - will need indexing
            print("\n--- FIRST QUERY WITH LARGE CONTEXTS (REQUIRES INDEXING) ---")
            start_time = time.time()
            response1 = text_gen.chat_completion(
                user_prompt="What are the key components of a robust AI system?",
                system_prompt="You are a helpful technical assistant.",
                system_context=large_system_context,
                context=large_user_context,
                max_tokens=100
            )
            first_time = time.time() - start_time
            print(f"Response time (first query with indexing): {first_time:.2f} seconds")
            print(f"Response: {response1[:100]}...")
            
            # Second query with SAME contexts - should use cache
            print("\n--- SECOND QUERY WITH IDENTICAL CONTEXTS (SHOULD USE CACHE) ---")
            start_time = time.time()
            response2 = text_gen.chat_completion(
                user_prompt="Explain methods for monitoring ML models in production.",
                system_prompt="You are a helpful technical assistant.",
                system_context=large_system_context,
                context=large_user_context,
                max_tokens=100
            )
            second_time = time.time() - start_time
            print(f"Response time (second query using cache): {second_time:.2f} seconds")
            print(f"Response: {response2[:100]}...")
            
            # Calculate and display efficiency improvement
            if first_time > 0:
                improvement = (first_time - second_time) / first_time * 100
                print(f"\nPerformance improvement: {improvement:.1f}% faster with cached RAG indexing")
                if improvement > 10:
                    print("✅ Significant improvement detected! RAG caching is working effectively.")
                else:
                    print("⚠️ Limited improvement - caching may not be working optimally.")
            
            # Try query with MODIFIED system context
            print("\n--- THIRD QUERY WITH MODIFIED SYSTEM CONTEXT (REQUIRES PARTIAL RE-INDEXING) ---")
            # Modify system context slightly to invalidate cache
            modified_system_context = large_system_context + "\n\n## NEW SECTION\nThis is a new section that wasn't in the original document."
            start_time = time.time()
            response3 = text_gen.chat_completion(
                user_prompt="What are best practices for AI model deployment?",
                system_prompt="You are a helpful technical assistant.",
                system_context=modified_system_context,
                context=large_user_context, # Same user context
                max_tokens=100
            )
            third_time = time.time() - start_time
            print(f"Response time (modified system context): {third_time:.2f} seconds")
            print(f"Response: {response3[:100]}...")
            
            # Show summary statistics
            print("\n--- SUMMARY ---")
            print(f"Baseline (no context):       {time.time() - start_time:.2f} sec")
            print(f"Initial indexing:            {first_time:.2f} sec")
            print(f"With cache:                  {second_time:.2f} sec")
            print(f"Re-indexing (modified ctx):  {third_time:.2f} sec")
            
        elif choice.lower() == "p":
            print("\n" + "="*50)
            print("🔄 CHANGE PROVIDER")
            print("="*50)
            
            # Display current provider
            current_provider = getattr(text_gen, "provider", "openai")
            print(f"Current provider: {current_provider}")
            
            # List available providers
            print("\nAvailable providers:")
            print("1. OpenAI - Cloud-based, requires API key")
            print("2. Ollama - Local running, no API key needed")
            
            provider_choice = prompt("Select provider (1/2)", "1" if current_provider == "openai" else "2")
            
            # Reinitialize with new provider
            if provider_choice == "1":
                new_provider = "openai"
                new_model = prompt("Enter OpenAI model name", "gpt-4o-mini")
            else:
                new_provider = "ollama"
                new_model = prompt("Enter Ollama model name", "gemma3:4b")
            
            # Recreate TextGen instance with new provider
            text_gen = TextGen(
                provider=new_provider,
                default_model=new_model
            )
            print(f"\n✅ Provider changed to: {new_provider} with model: {new_model}")
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE CLOSED 🏁")
    print("="*60 + "\n") 