import os
import sys
import json
import threading # Import threading
import traceback # Ensure traceback is imported
import logging # Import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Any, Callable # Added Optional, Any, Callable
from pydantic import BaseModel, Field

# Configure logging FIRST - Set root logger to suppress almost everything by default
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s:%(name)s:%(message)s')

# Set up module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Module default log level

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
# Navigate up to the project root (2 levels up from src/III_agents/)
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our API wrappers and utils
from src.II_textGen.textGen import TextGen
from src.VI_utils.utils import printColoured # Import printColoured

# Explicitly set levels for noisy loggers AFTER they are potentially imported/configured by TextGen etc.
logging.getLogger("src.II_textGen.textGen").setLevel(logging.WARNING)
logging.getLogger("src.II_textGen.tools").setLevel(logging.WARNING)
logging.getLogger("src.II_textGen.rag").setLevel(logging.WARNING) # Add RAG logger control
logging.getLogger("faiss").setLevel(logging.CRITICAL) # Suppress FAISS info

# ---- Agent Configuration ----
class AgentConfig(BaseModel):
    """Stores configuration for a specific agent persona."""
    name: str
    system_prompt: Optional[str] = "You are a helpful assistant. Be concise."
    temperature: Optional[float] = None
    max_tokens: Optional[int] = 100 # Default short max_tokens
    description: Optional[str] = "A general-purpose agent."
    log_color: Optional[str] = "white" # Color for logging output
    context: Optional[str] = None # Context documents for RAG
    system_context: Optional[str] = None # System context documents for RAG
    tool_names: Optional[List[str]] = None # Predefined tools the agent can use

class AgentGen(TextGen):
    """
    AgentGen: A lightweight yet powerful agent framework built on TextGen.
    Includes agent configuration management and basic parallel execution via Workers.
    """

    def __init__(self,
                 provider: str = "openai",
                 openai_api_key: Optional[str] = None,
                 replicate_api_token: Optional[str] = None,
                 default_model: Optional[str] = None,
                 short_term_limit: int = 8000,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        super().__init__(
            provider=provider,
            openai_api_key=openai_api_key,
            replicate_api_token=replicate_api_token,
            default_model=default_model,
            short_term_limit=short_term_limit,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.agent_configs: Dict[str, AgentConfig] = {}
        logger.info(f"🤖 AgentGen initialized with provider '{self.provider}' and model '{self.default_model}'.")

    def create_agent(self,
                     name: str,
                     system_prompt: Optional[str] = None,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     description: Optional[str] = None,
                     log_color: Optional[str] = "white",
                     context: Optional[str] = None,
                     system_context: Optional[str] = None,
                     tool_names: Optional[List[str]] = None):
        """
        Creates and stores a configuration for a named agent.
        Args:
            name (str): Name of the agent
            system_prompt (Optional[str]): Base system prompt for the agent
            temperature (Optional[float]): Temperature setting for generation
            max_tokens (Optional[int]): Maximum tokens to generate
            description (Optional[str]): Description of the agent's capabilities
            log_color (Optional[str]): Color for this agent's logs ('red', 'green', 'blue', etc.)
            context (Optional[str]): Context document for RAG
            system_context (Optional[str]): System context document for RAG
            tool_names (Optional[List[str]]): Predefined tools the agent can use
        """
        if name in self.agent_configs:
            logger.warning(f"⚠️ Agent configuration '{name}' already exists. Overwriting.")
            
        # Validate tool names if provided
        if tool_names:
            available_tools = self.get_available_tools()
            available_tool_names = [tool['name'] for tool in available_tools]
            valid_tool_names = [name for name in tool_names if name in available_tool_names]
            
            if len(valid_tool_names) < len(tool_names):
                invalid_tools = [name for name in tool_names if name not in available_tool_names]
                logger.warning(f"⚠️ Some tools for agent '{name}' are invalid and will be ignored: {invalid_tools}")
                tool_names = valid_tool_names

        config = AgentConfig(
            name=name,
            # Default to concise system prompt if none provided
            system_prompt=system_prompt or "You are a helpful assistant. Keep your response concise and under 100 tokens.",
            temperature=temperature,
            # Default to 100 tokens if not specified
            max_tokens=max_tokens if max_tokens is not None else 100, 
            description=description or f"Agent specializing in {name}.",
            log_color=log_color,
            context=context,
            system_context=system_context,
            tool_names=tool_names
        )
        self.agent_configs[name] = config
        
        # Print in the agent's color when creating it
        tool_info = f" with {len(tool_names)} TOOLS" if tool_names else ""
        context_info = ", with CONTEXT." if context or system_context else ""
        printColoured(f"✅ Agent configuration '{name}' created{tool_info}{context_info} (Color: {log_color}).", log_color)

    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """Retrieves a stored agent configuration by name."""
        return self.agent_configs.get(name)

    def select_best_tools(self, user_prompt: str, top_k: int = 3) -> list:
        """
        Dynamically selects the most relevant tools based on the user prompt.
        """
        # Fix: Get tools in a format that matches what we need
        available_tools = self.get_available_tools()
        available_tools_info = [f"- {tool['name']}: {tool['description']}"
                                for tool in available_tools]
        
        if not available_tools_info:
            return []

        # More explicit prompt to get the exact format we need
        tool_selection_prompt = f"""
Task: "{user_prompt}"

Available tools:
{', '.join(available_tools_info)}

Select up to {top_k} tools that would be most useful for this task.
FORMAT INSTRUCTIONS: Return ONLY a JSON array of tool names as strings. No other text or explanation.
Example: ["tool1", "tool2"]
If no tools are useful, return empty array: []
"""

        try:
            selected_tools = self.structured_output(
                user_prompt=tool_selection_prompt,
                system_prompt="You are a tool selector that returns ONLY a JSON array of tool names. No other format is accepted.",
                store_interaction=False,
                max_tokens=100  # Increased slightly to handle longer array
            )

            # Handle various possible response formats consistently
            if isinstance(selected_tools, dict):
                # Case 1: {'tools': ['tool1', 'tool2']}
                if 'tools' in selected_tools and isinstance(selected_tools['tools'], list):
                    selected_tools = selected_tools['tools']
                # Case 2: Look for any other list values that might contain our tools
                else:
                    for key, value in selected_tools.items():
                        if isinstance(value, list) and all(isinstance(item, str) for item in value):
                            selected_tools = value
                            break
                    else:
                        # If we couldn't find a valid list, use an empty list
                        selected_tools = []

            # Ensure we have a proper list
            if not isinstance(selected_tools, list):
                selected_tools = []
            
            # Filter to only valid tool names
            valid_tool_names = [tool['name'] for tool in available_tools]
            final_selection = [name for name in selected_tools if isinstance(name, str) and name in valid_tool_names]
            
            if final_selection:  # Only print if tools were actually selected
                logger.info(f"🛠️ Selected Tools for '{user_prompt[:30]}...': {final_selection}")
            
            return final_selection

        except Exception as e:
            logger.error(f"❌ Error during tool selection: {e}")
            return []

    def loop_simple(self,
                    user_prompt: str,
                    agent_name: Optional[str] = None,
                    system_prompt: str = None,
                    temperature: float = None,
                    max_tokens: int = None, # Default will be applied from config or class default
                    context: str = None,
                    system_context: str = None,
                    tool_names: Optional[List[str]] = None,
                    use_dynamic_tools: bool = False,
                    max_depth: int = 3, # Reduced default depth for brevity
                    verbose: bool = True) -> str:
        """
        Iterative loop using tools, aiming for concise responses.
        
        Args:
            user_prompt: The main user prompt
            agent_name: Name of the agent to use (if any)
            system_prompt: Optional override for the agent's system prompt
            temperature: Optional override for the agent's temperature
            max_tokens: Optional override for the agent's max_tokens
            context: Optional override for the agent's context
            system_context: Optional override for the agent's system_context
            tool_names: Optional override for the agent's tool_names
            use_dynamic_tools: Whether to use dynamic tool selection (default: False)
            max_depth: Maximum number of iterations
            verbose: Whether to print verbose output
        """
        config = None
        agent_log_color = "white"
        final_system_prompt = system_prompt
        final_temperature = temperature
        final_max_tokens = max_tokens if max_tokens is not None else 100
        final_context = context
        final_system_context = system_context
        final_tool_names = tool_names
        
        # Initialize from agent config if provided
        if agent_name:
            config = self.get_agent_config(agent_name)
            if config:
                agent_log_color = config.log_color
                if verbose: printColoured(f"⚙️ Using config for '{agent_name}' (Color: {agent_log_color}) in loop_simple.", "magenta")
                
                # Use config values unless explicitly overridden
                final_system_prompt = system_prompt or config.system_prompt
                final_temperature = temperature if temperature is not None else config.temperature
                final_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
                final_context = context or config.context
                final_system_context = system_context or config.system_context
                
                # Use provided tool_names, or agent's configured tools, or dynamic selection if requested
                if tool_names is not None:
                    final_tool_names = tool_names
                elif config.tool_names and not use_dynamic_tools:
                    final_tool_names = config.tool_names
                    if verbose: printColoured(f"🛠️ Using agent's predefined tools: {final_tool_names}", agent_log_color)
            else:
                 if verbose: printColoured(f"⚠️ Agent '{agent_name}' not found. Using defaults.", "yellow")

        response = ""
        agent_id = agent_name or 'Default'
        finished = False  # Flag to track if we've seen FINAL RESPONSE
        
        for i in range(max_depth):
            # Use dynamic tool selection only if requested and no predefined tools
            if use_dynamic_tools or final_tool_names is None:
                iteration_tools = self.select_best_tools(user_prompt)
            else:
                iteration_tools = final_tool_names

            iterative_user_prompt = f"Task: {user_prompt}\n"
            if response: 
                 iterative_user_prompt += f"Current Answer: {response}\n"
            # Simplified prompt for brevity
            iterative_user_prompt += "Refine the answer concisely using tools if needed. If complete, state 'FINAL RESPONSE:'."

            response = self.chat_completion(
                user_prompt=iterative_user_prompt,
                system_prompt=final_system_prompt or "Be concise. Use tools if needed. State 'FINAL RESPONSE:' when done.",
                temperature=final_temperature,
                max_tokens=final_max_tokens, # Apply default/config max_tokens
                context=final_context,
                system_context=final_system_context,
                tool_names=iteration_tools
            )

            if verbose:
                printColoured(f"🔄 Simple Loop {i+1} [{agent_id}]: {response}", agent_log_color)
                
            if "FINAL RESPONSE:" in response:
                response = response.replace("FINAL RESPONSE:", "").strip()
                if verbose: printColoured(f"🏁 Simple Loop [{agent_id}] finished.", "green")
                finished = True
                break

        # Only show max depth warning if we didn't finish successfully
        if not finished and verbose:
            printColoured(f"⚠️ Simple Loop [{agent_id}] max depth ({max_depth}) reached.", "yellow")

        return response.replace("FINAL RESPONSE:", "").strip()

    def loop_react(self,
                   user_prompt: str,
                   agent_name: Optional[str] = None,
                   system_prompt: str = None,
                   temperature: float = None,
                   max_tokens: int = None,
                   context: str = None,
                   system_context: str = None,
                   tool_names: Optional[List[str]] = None,
                   use_dynamic_tools: bool = False,
                   max_depth: int = 3,
                   verbose: bool = True) -> str:
        """
        ReAct loop, aiming for concise steps.
        
        Args:
            user_prompt: The main user prompt
            agent_name: Name of the agent to use (if any)
            system_prompt: Optional override for the agent's system prompt
            temperature: Optional override for the agent's temperature
            max_tokens: Optional override for the agent's max_tokens
            context: Optional override for the agent's context
            system_context: Optional override for the agent's system_context
            tool_names: Optional override for the agent's tool_names
            use_dynamic_tools: Whether to use dynamic tool selection (default: False)
            max_depth: Maximum number of iterations
            verbose: Whether to print verbose output
        """
        config = None
        agent_log_color = "white"
        final_system_prompt = system_prompt
        final_temperature = temperature
        final_max_tokens = max_tokens if max_tokens is not None else 100
        final_context = context
        final_system_context = system_context
        final_tool_names = tool_names
        agent_role = agent_name or 'Default'
        finished = False

        # Initialize from agent config if provided
        if agent_name:
            config = self.get_agent_config(agent_name)
            if config:
                agent_log_color = config.log_color
                if verbose: printColoured(f"⚙️ Using config for '{agent_name}' (Color: {agent_log_color}) in loop_react.", "magenta")
                
                # Use config values unless explicitly overridden
                final_system_prompt = system_prompt or config.system_prompt
                final_temperature = temperature if temperature is not None else config.temperature
                final_max_tokens = max_tokens if max_tokens is not None else config.max_tokens
                final_context = context or config.context
                final_system_context = system_context or config.system_context
                
                # Use provided tool_names, or agent's configured tools, or dynamic selection if requested
                if tool_names is not None:
                    final_tool_names = tool_names
                elif config.tool_names and not use_dynamic_tools:
                    final_tool_names = config.tool_names
                    if verbose: printColoured(f"🛠️ Using agent's predefined tools: {final_tool_names}", agent_log_color)
            else:
                if verbose: printColoured(f"⚠️ Agent '{agent_name}' not found. Using defaults.", "yellow")

        response = ""
        current_context = final_context or ""

        for i in range(max_depth):
            # Use dynamic tool selection only if requested and no predefined tools
            if use_dynamic_tools or final_tool_names is None:
                # Use shorter user prompt for tool selection if response exists
                tool_select_prompt = f"{user_prompt} Current state: {response}" if response else user_prompt
                iteration_tools = self.select_best_tools(tool_select_prompt)
            else:
                iteration_tools = final_tool_names

            # --- Step 1: OBSERVATION (Concise) ---
            obs_prompt = f"Task: {user_prompt}\n"
            if response: obs_prompt += f"Current State: {response}\n"
            obs_prompt += "Observe the current state briefly. What are the key facts?"

            observation = self.chat_completion(
                user_prompt=obs_prompt,
                system_prompt=final_system_prompt or "Observe concisely.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            if verbose: printColoured(f"🔄 React {i+1} [{agent_role}] OBSERVE: {observation}", agent_log_color)

            # --- Step 2: REFLECTION (Concise) ---
            refl_prompt = f"Task: {user_prompt}\nState: {response}\nObservation: {observation}\nReflect briefly: What is the next logical step?"

            reflection = self.chat_completion(
                user_prompt=refl_prompt,
                system_prompt=final_system_prompt or "Reflect concisely on the next step.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            if verbose: printColoured(f"🔄 React {i+1} [{agent_role}] REFLECT: {reflection}", agent_log_color)

            # --- Step 3: ACTION (Concise) ---
            # Improved action prompt to encourage complete sentences and proper termination
            act_prompt = f"""Task: {user_prompt}
Observation: {observation}
Plan: {reflection}

Execute the next step concisely using tools if needed. 
State 'FINAL RESPONSE:' only if task is complete.

IMPORTANT: Ensure your response is complete and doesn't cut off mid-sentence. 
If you're writing a story or creative content, craft complete sentences that fit within the token limit.
"""

            response = self.chat_completion(
                user_prompt=act_prompt,
                system_prompt=final_system_prompt or "Act concisely with complete sentences. Use tools if needed. State 'FINAL RESPONSE:' when done.",
                temperature=final_temperature, 
                max_tokens=final_max_tokens,
                context=current_context, 
                system_context=final_system_context,
                tool_names=iteration_tools
            )
            
            if verbose: printColoured(f"🔄 React {i+1} [{agent_role}] ACTION: {response}", agent_log_color)

            # Check if response appears to be cut off mid-sentence
            if response.strip().endswith(('as if the', 'as if', 'such as', 'like a', 'like the')) or response.endswith(('...', '…')):
                if verbose: printColoured(f"⚠️ React response may be incomplete. Adding final period.", "yellow")
                response = response.rstrip('.…') + "."  # Ensure it ends properly
            
            if "FINAL RESPONSE:" in response:
                response = response.replace("FINAL RESPONSE:", "").strip()
                if verbose: printColoured(f"🏁 React Loop [{agent_role}] finished.", "green")
                finished = True
                break

        # Only show max depth warning if we didn't finish successfully
        if not finished and verbose:
            printColoured(f"⚠️ React Loop [{agent_role}] max depth ({max_depth}) reached.", "yellow")

        return response.replace("FINAL RESPONSE:", "").strip()

    def triage_agent(self,
                     user_prompt: str,
                     handoff_agents: List[Dict[str, str]],
                     system_prompt: str = None,
                     temperature: float = None,
                     max_tokens: int = 1000,
                     context: str = None,
                     system_context: str = None,
                     max_depth: int = 5,
                     verbose: bool = True) -> Union[List[str], str]:
        """
        Uses the LLM to decide which agent(s) to hand off a task to.
        """
        if not handoff_agents:
            logger.warning("⚠️ No handoff agents provided for triage.")
            return "Error: No handoff agents specified."

        # Create a simplified list of agent names for the prompt
        agent_names = [agent['name'] for agent in handoff_agents]
        agent_options_text = "\n".join([f"- {agent['name']}: {agent['description']}" for agent in handoff_agents])

        # Extremely simplified and explicit prompt
        triage_decision_prompt = f"""
User request: '{user_prompt}'

Available Agents:
{agent_options_text}

FORMAT INSTRUCTIONS:
1. Select the best agent(s) for this request.
2. Return ONLY a JSON array of agent names as strings.
3. Example correct format: ["AgentName1", "AgentName2"]
4. If no agents fit, return: []
5. DO NOT use any other format or include any explanations.

Available agent names: {', '.join(agent_names)}
"""

        triage_system_prompt = "You are a triage coordinator that returns ONLY a JSON array of agent names. Your entire response must be a valid JSON array of strings."

        if verbose:
            printColoured(f"🚦 Triage Agent: Determining best agent for: '{user_prompt[:50]}...'", "magenta")

        try:
            # Use a higher max_tokens to avoid truncation issues with JSON
            selected_agent_names = self.structured_output(
                user_prompt=triage_decision_prompt,
                system_prompt=triage_system_prompt,
                temperature=temperature,
                max_tokens=4000,
                context=context,
                system_context=system_context,
                store_interaction=False
            )

            # Handle response format consistently
            if isinstance(selected_agent_names, dict):
                # Case 1: {'agents': ['Agent1', 'Agent2']}
                if 'agents' in selected_agent_names and isinstance(selected_agent_names['agents'], list):
                    selected_agent_names = selected_agent_names['agents']
                # Case 2: Look for agent names in dictionary keys that match our valid agent names
                else:
                    potential_agents = [k for k in selected_agent_names.keys() if k in agent_names]
                    if potential_agents:
                        selected_agent_names = potential_agents
                    # Case 3: Look for any list values that might contain agent names
                    else:
                        for key, value in selected_agent_names.items():
                            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                                selected_agent_names = value
                                break
                        else:
                            # If we couldn't find a valid list, use an empty list
                            selected_agent_names = []

            # Ensure we have a proper list
            if not isinstance(selected_agent_names, list):
                selected_agent_names = []
            
            # Filter to only valid agent names
            valid_agent_names = [agent['name'] for agent in handoff_agents]
            chosen_valid_names = [name for name in selected_agent_names if isinstance(name, str) and name in valid_agent_names]

            if verbose:
                if chosen_valid_names:
                    printColoured(f"🚦 Triage Decision: Hand off to -> {', '.join(chosen_valid_names)}", "green")
                else:
                    printColoured("🚦 Triage Decision: No specific agent selected.", "yellow")
            
            return chosen_valid_names

        except Exception as e:
            logger.error(f"❌ Triage Error: {e}")
            return f"Error: Triage failed due to an exception: {e}"

# ---- Worker Class ----
class Worker:
    """Runs an agent function in a separate thread."""
    def __init__(self, target_func: Callable, args: tuple = (), kwargs: dict = {}):
        self.target_func = target_func
        self.args = args
        self.kwargs = kwargs
        self.thread: Optional[threading.Thread] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        # Get agent color from kwargs if possible
        self.agent_name = kwargs.get('agent_name', 'Unknown')
        self.log_color = "grey" # Default worker log color
        agent_gen_instance = kwargs.get('agent_gen_instance') # Need instance to get color
        if agent_gen_instance and isinstance(agent_gen_instance, AgentGen):
             config = agent_gen_instance.get_agent_config(self.agent_name)
             if config: self.log_color = config.log_color

    def _run_target(self):
        try:
            self.kwargs['verbose'] = self.kwargs.get('verbose', True)
            # Note: Removing agent_gen_instance before calling target
            clean_kwargs = {k: v for k, v in self.kwargs.items() if k != 'agent_gen_instance'}
            printColoured(f"🧵 Worker starting [{self.agent_name}]...", self.log_color)
            self.result = self.target_func(*self.args, **clean_kwargs)
            printColoured(f"🧵 Worker finished [{self.agent_name}]. Result: {str(self.result)[:50]}...", self.log_color)
        except Exception as e:
            printColoured(f"🧵❌ Worker [{self.agent_name}] error: {e}", "red")
            self.error = e
            self.result = None
            # traceback.print_exc()

    def start(self):
        if self.thread is None or not self.thread.is_alive():
            self.result = None
            self.error = None
            self.thread = threading.Thread(target=self._run_target)
            self.thread.daemon = True
            self.thread.start()
            # printColoured(f"🧵 Worker thread [{self.agent_name}] started.", self.log_color)
        else:
            printColoured(f"🧵 Worker [{self.agent_name}] already running.", self.log_color)

    def join(self, timeout: Optional[float] = None) -> None:
        if self.thread and self.thread.is_alive():
            # printColoured(f"🧵 Worker waiting for [{self.agent_name}]...", self.log_color)
            self.thread.join(timeout)
            if self.thread.is_alive():
                 printColoured(f"🧵⚠️ Worker [{self.agent_name}] timed out after {timeout}s.", "yellow")
            # else:
                 # printColoured(f"🧵 Worker thread [{self.agent_name}] finished.", self.log_color)
        # elif self.thread:
             # printColoured(f"🧵 Worker [{self.agent_name}] was already finished.", self.log_color)
        # else:
             # printColoured(f"🧵 Worker [{self.agent_name}] was never started.", "yellow")

    def get_result(self) -> Any:
        if self.error:
            printColoured(f"🧵 Result for [{self.agent_name}]: Error occurred -> {self.error}", "red")
            return f"Error in worker: {self.error}"
        if self.thread and not self.thread.is_alive():
             return self.result
        elif not self.thread:
             printColoured(f"🧵 Result for [{self.agent_name}]: Thread never started.", "yellow")
             return None
        else: # Thread still running
             printColoured(f"🧵 Result for [{self.agent_name}]: Thread still running.", "yellow")
             return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 AgentGen Legacy Framework Test Suite (Concise & Coloured) 🚀")
    print("="*60 + "\n")

    # Initialize AgentGen
    ag = AgentGen() # Defaults to openai

    # ---- 1. Create Agent Configurations ----
    printColoured("\n--- Creating Agent Configurations ---", "magenta")
    ag.create_agent(
        name="Summarizer",
        system_prompt="You are an expert summarizer. Condense text into 1-2 key bullet points. Be extremely concise.",
        max_tokens=50,
        description="Summarizes text very concisely.",
        log_color="cyan"
    )
    ag.create_agent(
        name="CreativeWriter",
        system_prompt="You are a creative writer. Write a short, imaginative piece (max 3 sentences).",
        temperature=0.8,
        max_tokens=75,
        description="Writes very short creative text.",
        log_color="yellow"
    )
    ag.create_agent(
        name="CodeGenerator",
        system_prompt="You are a coding assistant. Generate a short Python code snippet.",
        max_tokens=100,
        description="Generates short Python code snippets.",
        log_color="blue"
    )
    ag.create_agent(
        name="ToolTester",
        system_prompt="Use the provided tools to answer the question concisely.",
        max_tokens=50,
        description="Tests tool usage.",
        log_color="green",
        tool_names=["get_current_datetime"]  # Predefined tools for this agent
    )
    
    # Create a new agent with context and system_context
    weather_context = """
    Weather in San Francisco, California:
    - Generally mild temperatures year-round
    - Famous for fog, especially in summer
    - Rainy season from November to April
    - Microclimates vary dramatically across different neighborhoods
    - Average temperature range: 50-65°F (10-18°C)
    """
    
    context_tools = ["get_weather", "get_current_datetime"]
    available_tools = ag.get_available_tools()
    available_tool_names = [tool['name'] for tool in available_tools]
    
    # Filter to only available tools
    valid_context_tools = [name for name in context_tools if name in available_tool_names]
    
    ag.create_agent(
        name="WeatherExpert",
        system_prompt="You are a weather expert who gives concise weather information. Use the provided context and tools to give accurate responses.",
        max_tokens=100,
        description="Provides weather information and forecasts.",
        log_color="blue",
        context=weather_context,
        system_context="Remember to be brief but informative about weather conditions.",
        tool_names=valid_context_tools
    )

    # ---- 2. Direct TextGen Method Calls (Inherited) ----
    printColoured("\n\n--- Test 1: Direct Inherited Method Calls ---", "magenta")

    # Example 1a: Simple chat_completion using CodeGenerator config defaults
    printColoured("\nExample 1a: Direct chat_completion (CodeGenerator defaults)", "white")
    code_prompt = "Write a simple Python function that adds two numbers."
    # Fetch config first
    code_gen_config = ag.get_agent_config("CodeGenerator")
    # Call inherited method directly, passing params explicitly (NO agent_name)
    direct_chat_result = ag.chat_completion(
        user_prompt=code_prompt, 
        system_prompt=code_gen_config.system_prompt if code_gen_config else None,
        temperature=code_gen_config.temperature if code_gen_config else None,
        max_tokens=code_gen_config.max_tokens if code_gen_config else None
    )
    printColoured(f">>> Result (CodeGenerator):\n{direct_chat_result}", ag.agent_configs.get("CodeGenerator", AgentConfig(name='temp')).log_color)

    # Example 1b: Structured output call (without using a named agent config)
    printColoured("\nExample 1b: Direct structured_output (no specific config)", "white")
    struct_prompt = "Extract name and title: 'Jane Doe is the Lead Engineer.'"
    direct_struct_result = ag.structured_output(
        user_prompt=struct_prompt,
        system_prompt="Extract JSON: {name: string, title: string}. ONLY JSON.",
        max_tokens=50
    )
    printColoured(f">>> Result (Structured):\n{json.dumps(direct_struct_result, indent=2)}", "white")

    # Example 1c: Chat completion using a tool
    printColoured("\nExample 1c: Direct chat_completion (ToolTester with get_current_datetime tool)", "white")
    tool_prompt = "What is the current date and time, write a poem about it?"
    # Remove debug message
    available_tools = ag.get_available_tools()
    available_tool_names = [tool['name'] for tool in available_tools]
    
    if 'get_current_datetime' in available_tool_names:
        # Fetch config first
        tool_tester_config = ag.get_agent_config("ToolTester")
        # Call inherited method directly, passing params explicitly (NO agent_name)
        direct_tool_result = ag.chat_completion(
            user_prompt=tool_prompt,
            system_prompt=tool_tester_config.system_prompt if tool_tester_config else None,
            tool_names=['get_current_datetime'],
            max_tokens=tool_tester_config.max_tokens if tool_tester_config else None
        )
        printColoured(f">>> Result (ToolTester):\n{direct_tool_result}", ag.agent_configs.get("ToolTester", AgentConfig(name='temp')).log_color)
    else:
        printColoured("Tool 'get_current_datetime' not found, skipping example 1c.", "yellow")

    # ---- 3. Simple Loop Test ----
    printColoured("\n\n--- Test 2: Simple Loop (Summarizer) ---", "magenta")
    prompt2 = "The quick brown fox jumps over the lazy dog near the bank of the river. This sentence is famous for containing all the letters of the English alphabet. It's often used for testing typewriters and keyboards."
    printColoured(f"Running loop_simple (Summarizer) for: '{prompt2[:50]}...'", "white")
    result2 = ag.loop_simple(user_prompt=prompt2, agent_name="Summarizer", max_depth=1)
    printColoured(f">>> Result (Summarizer):\n{result2}", ag.agent_configs["Summarizer"].log_color)

    # ---- 4. React Loop Test ----
    printColoured("\n\n--- Test 3: React Loop (CreativeWriter) ---", "magenta")
    prompt3 = "Write a short story (1-2 sentences) about a robot discovering music."
    printColoured(f"Running loop_react (CreativeWriter) for: '{prompt3}'", "white")
    result3 = ag.loop_react(user_prompt=prompt3, agent_name="CreativeWriter", max_depth=2)
    printColoured(f">>> Result (CreativeWriter - React):\n{result3}", ag.agent_configs["CreativeWriter"].log_color)
    
    # ---- 4b. Test with predefined tools ----
    if "WeatherExpert" in ag.agent_configs:
        printColoured("\n\n--- Test 3b: Agent with Context and Predefined Tools ---", "magenta")
        weather_prompt = "What's the weather like in San Francisco, use your context knowledge?"
        printColoured(f"Running loop_simple with WeatherExpert for: '{weather_prompt}'", "white")
        result_weather = ag.loop_simple(user_prompt=weather_prompt, agent_name="WeatherExpert", max_depth=3)
        printColoured(f">>> Result (WeatherExpert):\n{result_weather}", ag.agent_configs["WeatherExpert"].log_color)

    # ---- 5. Triage Agent Test ----
    printColoured("\n\n--- Test 4: Triage Agent ---", "magenta")
    handoff_options = [ag.agent_configs[name].model_dump(include={'name', 'description'})
                       for name in ag.agent_configs.keys()]
    prompt4 = "Generate a python function to calculate factorial."
    printColoured(f"Running triage_agent for: '{prompt4}'", "white")
    chosen_agents = ag.triage_agent(user_prompt=prompt4, handoff_agents=handoff_options)
    printColoured(f"Triage Result: {chosen_agents}", "magenta")

    if isinstance(chosen_agents, list) and chosen_agents:
        chosen_agent_name = chosen_agents[0]
        printColoured(f"--- Acting on Triage: Running '{chosen_agent_name}' --- ", "white")
        agent_color = ag.agent_configs.get(chosen_agent_name).log_color
        triage_action_result = ag.loop_simple(user_prompt=prompt4, agent_name=chosen_agent_name, max_depth=1)
        printColoured(f">>> Result from '{chosen_agent_name}':\n{triage_action_result}", agent_color)
    else:
        printColoured("No specific agent chosen by triage or error occurred.", "yellow")

    # ---- 6. Worker (Threading) Test ----
    printColoured("\n\n--- Test 5: Worker Threading (Concurrent Loops) ---", "magenta")
    printColoured("Starting Summarizer and CreativeWriter loops concurrently...", "white")

    worker1_prompt = "Artificial intelligence (AI) is intelligence demonstrated by machines... [long text snipped] ...achieving its goals."
    worker1 = Worker(
        target_func=ag.loop_simple,
        kwargs={'user_prompt': worker1_prompt, 'agent_name': 'Summarizer', 'max_depth': 1, 'agent_gen_instance': ag}
    )

    worker2_prompt = "A lonely lighthouse on a stormy night (1 sentence)."
    worker2 = Worker(
        target_func=ag.loop_react,
        kwargs={'user_prompt': worker2_prompt, 'agent_name': 'CreativeWriter', 'max_depth': 1, 'agent_gen_instance': ag}
    )

    worker1.start()
    worker2.start()

    printColoured("Waiting for workers to finish (max 60s)...", "white")
    worker1.join(timeout=60)
    worker2.join(timeout=60)

    printColoured("--- Worker Results ---", "magenta")
    result_w1 = worker1.get_result()
    result_w2 = worker2.get_result()

    printColoured(f">>> Worker 1 (Summarizer) Result:\n{result_w1}", ag.agent_configs["Summarizer"].log_color)
    printColoured(f">>> Worker 2 (CreativeWriter) Result:\n{result_w2}", ag.agent_configs["CreativeWriter"].log_color)

    print("\n" + "="*60)
    print("🏁 Test Suite Complete 🏁")
    print("="*60 + "\n")

