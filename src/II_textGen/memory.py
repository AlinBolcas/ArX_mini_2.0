import os
import json
import sys
from typing import List, Dict, Optional, Union, Any
from datetime import datetime
from pathlib import Path
import logging

# Configure logging - set default level for this module
# This ensures messages are emitted if not configured by root
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO) # Default to INFO if not set elsewhere

# --- LLM Provider Configuration ---
# Change this to "openai" to use OpenAIWrapper
LLM_PROVIDER = "openai" 
# --- End Configuration ---

# Add paths for importing other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the chosen LLM API wrapper
LLMAPIWrapper = None

try:
    if LLM_PROVIDER == "ollama":
        from I_integrations.ollama_API import OllamaWrapper
        LLMAPIWrapper = OllamaWrapper
        logger.info("Selected LLM Provider for Memory (default): Ollama")
    elif LLM_PROVIDER == "openai":
        from I_integrations.openai_API import OpenAIWrapper
        LLMAPIWrapper = OpenAIWrapper
        logger.info("Selected LLM Provider for Memory (default): OpenAI")
    else:
        logger.warning(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in memory.py. No default LLM API wrapper loaded.")
except ImportError as e:
    logger.warning(f"Could not import {LLM_PROVIDER.upper()} API for Memory default: {e}")
    LLMAPIWrapper = None

class Memory:
    """
    Memory system with short-term and long-term storage capabilities.
    Short-term memory manages recent conversation history with a token limit.
    Long-term memory stores essential insights extracted from conversations.
    Uses logging for operational messages.
    """
    
    def __init__(
        self, 
        llm_api: Optional[Any] = None,
        short_term_limit: int = 20000,
        long_term_interval: int = 3,
        memory_dir: Optional[str] = None,
        agent_name: Optional[str] = None
    ):
        """
        Initialize the memory system.
        
        Args:
            llm_api: An instance of the LLM API wrapper (e.g., OllamaWrapper or OpenAIWrapper)
            short_term_limit: Maximum token count for short-term memory
            long_term_interval: Number of interactions before automatic long-term storage
            memory_dir: Custom directory for storing memory files
            agent_name: Name of the agent this memory belongs to (for isolation)
        """
        # Initialize LLM API if not provided
        if llm_api is None:
            # This assumes LLMAPIWrapper is imported at the top level based on choice
            if LLMAPIWrapper is not None:
                try:
                    # Attempt to initialize the chosen wrapper
                    # Add specific init args if needed, e.g., auto_pull for Ollama
                    if LLM_PROVIDER == "ollama":
                        self.llm_api = LLMAPIWrapper(auto_pull=True)
                    else:
                         self.llm_api = LLMAPIWrapper()
                    logger.info(f"Initialized default {LLM_PROVIDER} API wrapper inside Memory.")
                except Exception as e:
                     # Use logger for the error
                     logger.error(f"Failed to auto-initialize {LLM_PROVIDER} API in Memory: {e}. Ensure it's configured correctly.", exc_info=True)
                     raise ImportError(f"Failed to auto-initialize {LLM_PROVIDER} API: {e}. Ensure it's configured correctly.") from e
            else:
                logger.critical(f"{LLM_PROVIDER.upper()} API module is required but could not be imported or initialized.")
                raise ImportError(f"{LLM_PROVIDER.upper()} API module is required but could not be imported or initialized.")
        else:
            self.llm_api = llm_api
            
        # Store the agent name for memory isolation
        self.agent_name = agent_name
        
        # Set up memory directories and files
        if memory_dir:
            self.memory_dir = Path(memory_dir)
        else:
            # Default to project's data/output/memory directory
            project_root = Path(__file__).parent.parent.parent
            self.memory_dir = project_root / "data" / "output" / "memory"
            
        # Create directory if it doesn't exist
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Create agent-specific subfolder if agent_name is provided
        if self.agent_name:
            self.agent_memory_dir = self.memory_dir / self.agent_name
            self.agent_memory_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using agent-specific memory directory: {self.agent_memory_dir}")
            
            # Set up agent-specific file paths
            self.short_term_file = self.agent_memory_dir / "short_term_memory.json"
            self.long_term_file = self.agent_memory_dir / "long_term_memory.json"
            self.short_term_markdown = self.agent_memory_dir / "short_term_memory.md"
            self.long_term_markdown = self.agent_memory_dir / "long_term_memory.md"
        else:
            # Set up file paths with default names (shared memory)
            self.short_term_file = self.memory_dir / "short_term_memory.json"
            self.long_term_file = self.memory_dir / "long_term_memory.json"
            self.short_term_markdown = self.memory_dir / "short_term_memory.md"
            self.long_term_markdown = self.memory_dir / "long_term_memory.md"
        
        # Configuration parameters
        self.short_term_limit = short_term_limit
        self.long_term_interval = long_term_interval
        self._short_term_counter = 0
        
        # Initialize empty memory files if they don't exist
        self._initialize_memory_files()
        
        logger.info(f"Memory initialized{' for agent '+self.agent_name if self.agent_name else ''}. Short-term limit: {short_term_limit} tokens, LTM interval: {long_term_interval} interactions.")
        
    def _initialize_memory_files(self):
        """Create empty memory files if they don't exist."""
        if not self.short_term_file.exists():
            with open(self.short_term_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)
            logger.info(f"Initialized empty short-term memory file: {self.short_term_file}")
                
        if not self.long_term_file.exists():
            with open(self.long_term_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=4)
            logger.info(f"Initialized empty long-term memory file: {self.long_term_file}")
                
    def save_short_term(self, system_prompt: str, user_prompt: str, assistant_response: str):
        """
        Save an interaction to short-term memory.
        
        Args:
            system_prompt: The system message
            user_prompt: The user's message
            assistant_response: The assistant's response
        """
        try:
            # Load current short-term memory
            with open(self.short_term_file, "r", encoding="utf-8") as f:
                short_term_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning(f"Short-term memory file not found or invalid, starting new list.")
            short_term_data = []
            
        # Create new memory entry
        new_entry = {
            "user": user_prompt,
            "assistant": assistant_response,
            "system": system_prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to memory
        short_term_data.append(new_entry)
        
        # Trim if exceeding token limit
        trimmed_count = self._trim_short_term_memory(short_term_data)
        if trimmed_count > 0:
             logger.info(f"Trimmed {trimmed_count} oldest entries from short-term memory.")
        
        # Save updated memory
        with open(self.short_term_file, "w", encoding="utf-8") as f:
            json.dump(short_term_data, f, indent=4)
            
        # Update markdown for readability
        self._update_markdown(self.short_term_file, self.short_term_markdown)
        
        # Increment counter and check for automatic long-term storage
        self._short_term_counter += 1
        if self._short_term_counter >= self.long_term_interval:
            logger.info(f"Reached long-term interval ({self.long_term_interval}), attempting insight extraction.")
            self.extract_to_long_term()
            self._short_term_counter = 0

    def retrieve_short_term_formatted(self) -> List[Dict[str, str]]:
        """
        Retrieve short-term memory formatted for LLM API calls.
        
        Returns:
            List of messages in the format expected by OpenAI API
        """
        try:
            with open(self.short_term_file, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Attempted to retrieve short-term memory, but file not found or invalid.")
            return []
            
        # Format for OpenAI API
        formatted_data = []
        for entry in memory_data:
            if "system" in entry:
                formatted_data.append({"role": "system", "content": entry["system"]})
            if "user" in entry:
                formatted_data.append({"role": "user", "content": entry["user"]})
            if "assistant" in entry:
                formatted_data.append({"role": "assistant", "content": entry["assistant"]})
                
        return formatted_data

    def retrieve_long_term(self) -> List[str]:
        """
        Retrieve insights from long-term memory.
        
        Returns:
            List of insight strings
        """
        try:
            with open(self.long_term_file, "r", encoding="utf-8") as f:
                insights = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.warning("Attempted to retrieve long-term memory, but file not found or invalid.")
            return []
            
        return [entry["insight"] for entry in insights if entry.get("insight")]

    def clear(self):
        """
        Clear all memory by resetting files.
        """
        # Delete and recreate the memory files
        files_to_clear = [
            self.short_term_file, self.long_term_file,
            self.short_term_markdown, self.long_term_markdown
        ]
        cleared_count = 0
        for file_path in files_to_clear:
            if file_path.exists():
                try:
                    file_path.unlink()
                    cleared_count += 1
                except OSError as e:
                    logger.error(f"Error removing memory file {file_path}: {e}")

        # Reset counter
        self._short_term_counter = 0
        
        # Initialize empty files
        self._initialize_memory_files()
        
        logger.info(f"Memory reset complete. Removed {cleared_count} files and re-initialized.")

    def save(self, filename: Optional[str] = None) -> str:
        """
        Save memory to a specific file.
        
        Args:
            filename: Optional filename to save to
            
        Returns:
            Path to the saved file
        """
        # First make sure all memory files are in sync (no need for separate backup files)
        # Now we'll just save directly to the memory files
        
        try:
            # Get formatted memory data
            formatted_short_term = self.retrieve_short_term_formatted()
            long_term_insights = self.retrieve_long_term()
            
            # Ensure memory directory exists
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            
            # Update markdown files to make sure they're current
            self._update_markdown(self.short_term_file, self.short_term_markdown)
            self._update_markdown(self.long_term_file, self.long_term_markdown)
            
            logger.info(f"Memory synced to files in: {self.memory_dir}")
            return str(self.memory_dir)
            
        except Exception as e:
            logger.error(f"Error saving memory: {e}", exc_info=True)
            raise
        
    def load(self, filename: str):
        """
        Load memory from a file.
        
        Args:
            filename: Name of the file to load (with or without extension)
        """
        # Ensure it has .json extension
        if not filename.lower().endswith('.json'):
            filename += '.json'
            
        # Find the file in the memory directory
        load_path = self.memory_dir / filename
        
        if not load_path.exists():
            logger.error(f"Memory file not found: {load_path}")
            raise FileNotFoundError(f"Memory file not found: {load_path}")
            
        try:
            # Load the memory data
            with open(load_path, "r", encoding="utf-8") as f:
                memory_data = json.load(f)
                
            # Reset current memory
            self.clear()
            
            # Load short term memory if present
            if "short_term" in memory_data and isinstance(memory_data["short_term"], list):
                # Convert to the expected format for short_term_file
                short_term_entries = []
                for i in range(0, len(memory_data["short_term"]), 3):
                    if i+2 < len(memory_data["short_term"]):
                        entry = {
                            "system": memory_data["short_term"][i]["content"],
                            "user": memory_data["short_term"][i+1]["content"],
                            "assistant": memory_data["short_term"][i+2]["content"],
                            "timestamp": datetime.now().isoformat()
                        }
                        short_term_entries.append(entry)
                
                with open(self.short_term_file, "w", encoding="utf-8") as f:
                    json.dump(short_term_entries, f, indent=4)
            
            # Load long term memory if present
            if "long_term" in memory_data and isinstance(memory_data["long_term"], list):
                # Convert to the expected format for long_term_file
                long_term_entries = []
                for insight in memory_data["long_term"]:
                    entry = {
                        "insight": insight,
                        "timestamp": datetime.now().isoformat()
                    }
                    long_term_entries.append(entry)
                    
                with open(self.long_term_file, "w", encoding="utf-8") as f:
                    json.dump(long_term_entries, f, indent=4)
                    
            # Update markdown files
            self._update_markdown(self.short_term_file, self.short_term_markdown)
            self._update_markdown(self.long_term_file, self.long_term_markdown)
                
            logger.info(f"Memory loaded from: {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading memory: {e}", exc_info=True)
            raise

    def extract_to_long_term(self, num_interactions: Optional[int] = None, force_insight: Optional[str] = None):
        """
        Extract insights from recent interactions and save to long-term memory.
        
        Args:
            num_interactions: Number of recent interactions to analyze
            force_insight: Optional insight to save directly (bypasses OpenAI call)
        """
        # If a direct insight is provided, save it immediately
        if force_insight:
            self._save_long_term_insight(force_insight)
            logger.info(f"Saved forced insight to long-term memory: '{force_insight[:50]}...'")
            return
        
        # Normal extraction process with API call
        # Default to the long_term_interval if not specified
        if num_interactions is None:
            num_interactions = self.long_term_interval
        
        # Get recent interactions from the full memory
        try:
            with open(self.short_term_file, "r", encoding="utf-8") as f:
                short_term_data = json.load(f)
                
            if not short_term_data:
                logger.info("No recent memories found for extraction.")
                return
                
            # Take the most recent n interactions
            recent_memories = short_term_data[-num_interactions:]
            
            # First determine if the conversation contains valuable insights
            decision_system = (
                "You are an AI librarian responsible for knowledge preservation. "
                "Your task is to identify if the following conversation contains any "
                "valuable, factual insights or important information worth preserving. "
                "Respond with only 'Yes' or 'No'."
            )
            
            decision_prompt = "Recent Conversation:\n\n"
            for i, memory in enumerate(recent_memories, 1):
                decision_prompt += f"Interaction {i}:\n"
                decision_prompt += f"User: {memory.get('user', '')}\n"
                decision_prompt += f"Assistant: {memory.get('assistant', '')}\n\n"
                
            decision_prompt += "Does this conversation contain valuable information worth preserving?"
            
            # Make API call to decide
            decision = self.llm_api.chat_completion(
                user_prompt=decision_prompt,
                system_prompt=decision_system,
                temperature=0.2,
                max_tokens=10
            ).strip().lower()
            
            # Extract insight if conversation is valuable
            if "yes" in decision:
                # Extract the insight
                extraction_system = (
                    "You are an AI librarian responsible for knowledge preservation. "
                    "Your task is to extract key insights from the following conversation. "
                    "Create a clear, concise statement that captures this knowledge in a way that "
                    "will be useful for future reference."
                )
                
                extraction_prompt = "Extract a key insight from this conversation:\n\n"
                for i, memory in enumerate(recent_memories, 1):
                    extraction_prompt += f"Interaction {i}:\n"
                    extraction_prompt += f"User: {memory.get('user', '')}\n"
                    extraction_prompt += f"Assistant: {memory.get('assistant', '')}\n\n"
                    
                # Extract the insight
                insight = self.llm_api.chat_completion(
                    user_prompt=extraction_prompt,
                    system_prompt=extraction_system,
                    temperature=0.3,
                    max_tokens=100
                ).strip()
                
                # Save the insight
                self._save_long_term_insight(insight)
                
                logger.info(f"Long-term memory insight extracted: {insight[:70]}...")
            else:
                logger.info("No valuable insights found in recent conversations.")
                
        except Exception as e:
            logger.error(f"Error during insight extraction: {e}", exc_info=True)
        
    def _save_long_term_insight(self, insight: str):
        """
        Save an insight to long-term memory.
        
        Args:
            insight: The insight to save
        """
        if not insight: return # Don't save empty insights
        try:
            with open(self.long_term_file, "r", encoding="utf-8") as f:
                long_term_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            long_term_data = []
            
        # Add new insight
        long_term_data.append({
            "insight": insight,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save updated memory
        with open(self.long_term_file, "w", encoding="utf-8") as f:
            json.dump(long_term_data, f, indent=4)
            
        # Update markdown - make sure this works
        self._update_markdown(self.long_term_file, self.long_term_markdown)
        
        logger.info(f"Insight saved to long-term memory and markdown updated.")

    def get_formatted_context(
        self, 
        include_short_term: bool = True,
        include_long_term: bool = True,
        max_short_term_entries: Optional[int] = None
    ) -> str:
        """
        Get a formatted context string combining short and long-term memory.
        Useful for providing context to an LLM.
        
        Args:
            include_short_term: Whether to include short-term memory
            include_long_term: Whether to include long-term memory
            max_short_term_entries: Max number of short-term entries to include
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add long-term insights if requested
        if include_long_term:
            insights = self.retrieve_long_term()
            if insights:
                context_parts.append("# LONG-TERM MEMORY (Key Insights)")
                for i, entry in enumerate(insights, 1):
                    context_parts.append(f"{i}. {entry}")
                context_parts.append("")  # Empty line for separation
        
        # Add short-term conversation if requested
        if include_short_term:
            short_term = self.retrieve_short_term_formatted()
            if short_term:
                # Apply limit if specified
                if max_short_term_entries is not None and max_short_term_entries > 0:
                    short_term = short_term[-max_short_term_entries:]
                    
                context_parts.append("# RECENT CONVERSATION HISTORY")
                for i, entry in enumerate(short_term, 1):
                    context_parts.append(f"## Interaction {i}")
                    if "system" in entry:
                        context_parts.append(f"System: {entry['system']}")
                    context_parts.append(f"User: {entry['user']}")
                    context_parts.append(f"Assistant: {entry['assistant']}")
                    context_parts.append("")  # Empty line
        
        return "\n".join(context_parts)
        
    def _update_markdown(self, json_file: Path, md_file: Path):
        """
        Convert a JSON memory file to Markdown for easier reading.
        
        Args:
            json_file: Path to the JSON file
            md_file: Path to output Markdown file
        """
        try:
            if not json_file.exists(): return # Don't try to update if source doesn't exist
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            markdown = f"# Memory Log ({json_file.stem})\n\n"
            
            for entry in reversed(data):
                # Get timestamp
                timestamp = entry.get('timestamp', 'Unknown time')
                markdown += f"## {timestamp}\n\n"
                
                # Format based on entry type
                if "insight" in entry:  # Long-term memory
                    markdown += f"> {entry['insight']}\n\n"
                else:  # Short-term memory
                    if "system" in entry:
                        markdown += f"**System:** {entry['system']}\n\n"
                    markdown += f"**User:** {entry['user']}\n\n"
                    markdown += f"**Assistant:** {entry['assistant']}\n\n"
                    
                markdown += "---\n\n"
                
            # Write markdown file
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(markdown)
                
        except Exception as e:
            logger.error(f"Error updating markdown file {md_file} from {json_file}: {e}", exc_info=True)

    def _trim_short_term_memory(self, memory_data: List[Dict[str, Any]]) -> int:
        """
        Trim short-term memory to stay under the token limit.
        Removes oldest entries first.
        
        Args:
            memory_data: The memory data to trim
        """
        # Simple token estimation: character count is a reasonable approximation
        total_tokens = sum(
            len(entry.get("user", "")) + 
            len(entry.get("assistant", "")) + 
            len(entry.get("system", "")) 
            for entry in memory_data
        )
        
        removed_count = 0
        while total_tokens > self.short_term_limit and memory_data:
            removed_entry = memory_data.pop(0)
            total_tokens -= (
                len(removed_entry.get("user", "")) + 
                len(removed_entry.get("assistant", "")) + 
                len(removed_entry.get("system", ""))
            )
            removed_count += 1
        return removed_count

# Example usage with a practical function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if LLMAPIWrapper is None:
        logger.critical(f"Error: Could not import or initialize the {LLM_PROVIDER.upper()} API wrapper.")
        exit(1)
    
    # Initialize the chosen API Wrapper instance for the example
    llm_api = None # Initialize as None
    try:
        if LLM_PROVIDER == "ollama":
             # Ollama specific init, auto_pull checks/pulls the default model if not present
             llm_api = LLMAPIWrapper(auto_pull=True) 
        else:
             # Assumes OpenAI/other init doesn't need specific args here
             llm_api = LLMAPIWrapper() 
        logger.info(f"Successfully initialized {LLM_PROVIDER.upper()} API wrapper for example usage.")
    except Exception as e:
        logger.critical(f"Error initializing {LLM_PROVIDER.upper()} API wrapper: {e}", exc_info=True)
        exit(1)

    # Initialize memory system, passing the specific API instance
    # Note: Memory class can also auto-initialize if llm_api=None is passed, 
    # but we explicitly pass the instance created above for clarity in this example.
    memory = Memory(llm_api=llm_api, short_term_limit=20000) 
    memory.clear()
    
    logger.info(f"\n===== MEMORY SYSTEM PRACTICAL DEMO (using {LLM_PROVIDER.upper()}) =====")
    
    def chatbot_with_memory(user_input: str, system_prompt: str = "You are a helpful AI assistant."):
        """
        A simple chatbot function that uses memory to maintain conversation context.
        
        Args:
            user_input: User's message
            system_prompt: System message for the LLM
            
        Returns:
            Assistant's response
        """
        # Get formatted conversation history from memory object
        message_history = memory.retrieve_short_term_formatted()
        
        # Generate response using the chosen LLM API instance (`llm_api` passed to Memory)
        # The Memory class's internal llm_api attribute is used within its methods like extract_to_long_term
        # Here in the example function, we use the llm_api instance directly for the chat completion.
        response = llm_api.chat_completion( 
            user_prompt=user_input,
            system_prompt=system_prompt,
            message_history=message_history
        )
        
        # Save the interaction to memory (Memory uses its internal llm_api for insight extraction later)
        memory.save_short_term(system_prompt, user_input, response)
        
        # After 3 interactions, trigger insight extraction (uses Memory's internal llm_api)
        if memory._short_term_counter >= 3:
            logger.info("\nChecking for insights to add to long-term memory...")
            memory.extract_to_long_term(num_interactions=4)  # Use the last 4 interactions
        
        return response
    
    # Demonstrate a multi-turn conversation brainstorming the AI Art & Music project
    conversations = [
        "Let's brainstorm ideas for an AI system that generates art from music. What are the key components we need?",
        "How can we effectively analyze and extract meaningful attributes from music to generate good image prompts?",
        "What kind of memory system would work best for storing and retrieving past music-to-image generations?",
        "How can we implement an effective feedback loop for iteratively refining the generated images?",
        "What are some creative ways we could use multiple AI agents to collaborate on the art generation process?",
        "How should we structure the pipeline to ensure coherent visual representations of musical elements?"
    ]
    
    logger.info(f"\n===== MULTI-TURN CONVERSATION (using {LLM_PROVIDER.upper()}) =====")
    for i, user_message in enumerate(conversations):
        logger.info(f"\nUser [{i+1}]: {user_message}")
        response = chatbot_with_memory(user_message)
        logger.info(f"AI: {response[:150]}...")
    
    # Force adding key insights for demonstration
    logger.info("\n> Adding important insights to long-term memory...")
    key_insights = [
        "AI art generation requires a multi-stage pipeline including music analysis, prompt engineering, image generation and iterative refinement to create coherent visual representations of musical elements.",
        "Long-term memory and agent collaboration are essential for storing past music-to-image generations and enabling sophisticated iteration cycles that improve output quality over time."
    ]
    
    for insight in key_insights:
        memory.extract_to_long_term(force_insight=insight)
    
    # Show memory contents
    logger.info("\n===== CONVERSATION MEMORY =====")
    logger.info("\nShort-term memory (recent messages):")
    formatted = memory.retrieve_short_term_formatted()
    messages_by_role = {"system": [], "user": [], "assistant": []}
    for msg in formatted:
        # Ensure roles are handled correctly, default to 'content' if role missing
        role = msg.get("role", "unknown") 
        content = msg.get("content", "")
        if role not in messages_by_role: messages_by_role[role] = []
        messages_by_role[role].append(content)
    
    # Display conversation turns using user messages as anchors
    num_user_msgs = len(messages_by_role.get("user", []))
    for i in range(num_user_msgs):
        logger.info(f"\nConversation Turn {i+1}:")
        user_msg = messages_by_role["user"][i]
        logger.info(f"User: {user_msg[:50]}..." if len(user_msg) > 50 else f"User: {user_msg}")
        
        # Check if corresponding assistant message exists
        if i < len(messages_by_role.get("assistant", [])):
            ai_msg = messages_by_role["assistant"][i]
            logger.info(f"AI: {ai_msg[:50]}..." if len(ai_msg) > 50 else f"AI: {ai_msg}")
        else:
             logger.info("AI: [No corresponding assistant message found]") # Handle potential mismatch
    
    logger.info("\nLong-term memory (key insights):")
    insights = memory.retrieve_long_term()
    if insights:
        for i, insight in enumerate(insights, 1):
            logger.info(f"{i}. {insight}")
    else:
        logger.info("No long-term insights stored yet.")
    
    logger.info(f"\nMemory files stored in: {memory.memory_dir}")
    logger.info(f"\n===== DEMO COMPLETE (using {LLM_PROVIDER.upper()}) ======") 