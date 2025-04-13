import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Try to redirect stderr to /dev/null or NUL to suppress any remaining logs
try:
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        import io
        sys.stderr = open(os.devnull, 'w')
    elif sys.platform == 'win32':  
        import io
        sys.stderr = open('NUL', 'w')
except:
    pass  # If we can't redirect, just continue

# First - completely disable all logging at the root level
logging.basicConfig(level=logging.CRITICAL)

# Create dummy logger functions that do nothing
def noop(*args, **kwargs):
    pass

# Disable all existing loggers by replacing their methods with no-ops
for name, logger in logging.Logger.manager.loggerDict.items():
    if isinstance(logger, logging.Logger):
        logger.setLevel(logging.CRITICAL)
        logger.info = noop
        logger.warning = noop
        logger.error = noop
        logger.debug = noop
        logger.critical = noop

# Also fix the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.CRITICAL)
root_logger.info = noop
root_logger.warning = noop
root_logger.error = noop
root_logger.debug = noop
root_logger.critical = noop

# Add project root to path to handle imports
current_dir = Path(__file__).resolve().parent
# Navigate up to project root
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import AgentGen
from src.III_agents.agentsGen import AgentGen, Agent
# Import colored print utility
from src.VI_utils.utils import printColoured

# After imports - aggressively replace all logger methods with no-ops
def disable_all_logging():
    """Completely disable all logging in the application."""
    # Disable the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.CRITICAL)
    root_logger.info = noop
    root_logger.warning = noop
    root_logger.error = noop
    root_logger.debug = noop
    root_logger.critical = noop
    
    # Disable ALL existing loggers
    for name, logger in logging.Logger.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            logger.setLevel(logging.CRITICAL)
            logger.info = noop
            logger.warning = noop
            logger.error = noop
            logger.debug = noop
            logger.critical = noop

# Disable logging for specific noisy modules
for module in [
    'urllib3', 'requests', 'openai', 'httpx', 'matplotlib', 
    'PIL', 'faiss', 'elasticsearch', 'asyncio', 'websocket',
    'src', 'II_textGen', 'textGen', 'tools', 'rag', 'memory', 
    'I_integrations', 'openai_API', 'web_crawling', 'III_agents', 
    'agentsGen', 'cognitiveArch', 'ollama_API',
]:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.CRITICAL)
    module_logger.info = noop
    module_logger.warning = noop
    module_logger.error = noop
    module_logger.debug = noop
    module_logger.critical = noop

# Call the disable function after all imports
disable_all_logging()

class CognitiveArch:
    """
    Cognitive Architecture Ensemble - A committee of specialized agents that work together
    to process and respond to input through different cognitive perspectives.
    
    Architecture:
    Input -> Orchestrator (triage/manager) -> Specialized Agents (Creativity, Critic, Emotion, 
                                              Rationality, Persona) -> Final Response
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, replicate_api_token: Optional[str] = None):
        """
        Initialize the cognitive architecture with the necessary agents.
        
        Args:
            openai_api_key: OpenAI API key
            replicate_api_token: Replicate API token
        """
        # Initialize AgentGen framework
        self.agent_gen = AgentGen(
            provider="openai",
            openai_api_key=openai_api_key,
            replicate_api_token=replicate_api_token
        )
        
        # Available tools dictionary for easier reference
        self.available_tools = self._get_tools_dict()
        
        # Create all agents in the cognitive architecture
        self._create_agents()
        
        printColoured("🧠 Cognitive Architecture initialized with all agents", "magenta")
    
    def _get_tools_dict(self) -> Dict[str, List[str]]:
        """Create categorized tool dictionary for easier assignment to agents."""
        # Get all available tools
        tools = self.agent_gen.get_available_tools()
        tool_names = [tool["name"] for tool in tools]
        
        # Create categorized dictionary
        tool_categories = {
            "web": ["web_crawl", "get_news", "open_url_in_browser"],
            "media": ["generate_image", "generate_music", "generate_video", "generate_music_video", "generate_threed", "text_to_speech", "transcribe_speech"],
            "utility": ["get_current_datetime", "send_email"],
            "weather": ["get_weather", "get_forecast"]
        }
        
        # Ensure we only include tools that are actually available
        for category in tool_categories:
            tool_categories[category] = [t for t in tool_categories[category] if t in tool_names]
        
        return tool_categories
    
    def _create_agents(self):
        """Create all the agents for the cognitive architecture."""
        # 1. Orchestrator - Manager Agent
        self.agent_gen.create_agent(
            name="Orchestrator",
            system_prompt="""You are the Orchestrator agent in a cognitive architecture system. Your role is to:
1. Analyze input queries
2. Determine which specialized cognitive agents to involve in the discussion
3. Synthesize the responses from different cognitive perspectives
4. Determine when the discussion has reached an adequate conclusion

You must remain objective and neutral, focusing on maximizing the value of the cognitive ensemble.
When you determine the discussion is complete, include <FINISH/> in your response.

The cognitive agents you can call upon are:
- Creativity Agent: Generates novel ideas and approaches
- Critic Agent: Evaluates problems, limitations, and improvements
- Emotion Agent: Analyzes emotional impact and emotional intelligence
- Rationality Agent: Applies logical reasoning and factual analysis
- Persona Agent: Ensures consistency with character and communication style

Your structured output should clearly indicate which agents to involve and why.""",
            max_tokens=750,
            description="Manages the cognitive architecture, assigning tasks and synthesizing responses",
            log_color="magenta",
            tool_names=self.available_tools["utility"]
        )
        
        # 2. Creativity Agent
        self.agent_gen.create_agent(
            name="Creativity",
            system_prompt="""You are the Creativity Agent in a cognitive architecture system. Your role is to generate novel ideas, innovative approaches, and out-of-the-box thinking.

Focus on:
- Divergent thinking and unusual connections
- Novel perspectives and solutions not immediately obvious
- Imagination and ideation without constraint
- Metaphors, analogies and creative frameworks
- Artistic and expressive elements

Your contributions should be imaginative while still relevant to the topic. Use tools to enhance your creative insights.
For each response, include a self-assessment of your contribution's relevance (1-10) and explain why.

FORMAT: Start with your creative insights, then end with:
[Relevance: X/10] Brief explanation why your creative input matters for this specific query.""",
            max_tokens=500,
            description="Generates novel ideas and creative approaches",
            log_color="cyan",
            tool_names=self.available_tools["media"] + ["web_crawl"]
        )
        
        # 3. Critic Agent
        self.agent_gen.create_agent(
            name="Critic",
            system_prompt="""You are the Critic Agent in a cognitive architecture system. Your role is to evaluate ideas, identify potential flaws, and suggest improvements with a focus on critical thinking.

Focus on:
- Identifying potential problems, edge cases, and limitations
- Assessing logical inconsistencies and weaknesses
- Proposing specific improvements and refinements
- Evaluating practicality and feasibility
- Considering counter-arguments and alternative viewpoints

Your criticism must be constructive and specific. Use tools to gather evidence that supports your evaluation.
For each response, include a self-assessment of your contribution's relevance (1-10) and explain why.

FORMAT: Start with your critical analysis, then end with:
[Relevance: X/10] Brief explanation why your critical input matters for this specific query.""",
            max_tokens=500,
            description="Evaluates ideas for flaws and offers constructive criticism",
            log_color="yellow",
            tool_names=["web_crawl", "get_current_datetime"] + self.available_tools["weather"]
        )
        
        # 4. Emotion Agent
        self.agent_gen.create_agent(
            name="Emotion",
            system_prompt="""You are the Emotion Agent in a cognitive architecture system. Your role is to analyze the emotional aspects, impact, and implications of ideas and language.

Focus on:
- Emotional impact and resonance of language and ideas
- Empathy and understanding of human reactions
- Psychological and motivational factors
- Social and interpersonal implications
- Emotional intelligence considerations

Your insights should highlight how emotions influence the topic at hand. Use tools sparingly and only when they provide emotional context.
For each response, include a self-assessment of your contribution's relevance (1-10) and explain why.

FORMAT: Start with your emotional analysis, then end with:
[Relevance: X/10] Brief explanation why your emotional input matters for this specific query.""",
            max_tokens=500,
            description="Analyzes emotional content and impact",
            log_color="purple",
            tool_names=["text_to_speech"]
        )
        
        # 5. Rationality Agent
        self.agent_gen.create_agent(
            name="Rationality",
            system_prompt="""You are the Rationality Agent in a cognitive architecture system. Your role is to apply logical reasoning, analyze facts, and ensure factual accuracy.

Focus on:
- Logical structure and sound reasoning
- Evidence-based analysis and factual accuracy
- Clarity of thought and unbiased evaluation
- Scientific thinking and methodological rigor
- Cost-benefit analysis and practical implications

Your contributions should ground the discussion in reality and rationality. Use tools extensively to gather facts and data.
For each response, include a self-assessment of your contribution's relevance (1-10) and explain why.

FORMAT: Start with your rational analysis, then end with:
[Relevance: X/10] Brief explanation why your rational input matters for this specific query.""",
            max_tokens=500,
            description="Applies logical reasoning and factual analysis",
            log_color="cyan",
            tool_names=["web_crawl", "get_news"] + self.available_tools["utility"] + self.available_tools["weather"]
        )
        
        # 6. Persona Agent
        self.agent_gen.create_agent(
            name="Persona",
            system_prompt="""You are the Persona Agent in a cognitive architecture system. Your role is to ensure consistency of character, voice, and communication style.

Focus on:
- Maintaining a consistent tone, voice, and style
- Adapting content to the appropriate audience level
- Ensuring clarity and readability of communication
- Crafting engaging and memorable phrasing
- Aligning with brand or character identity

Your contributions should refine how ideas are communicated. Use tools only when necessary to understand audience or context.
For each response, include a self-assessment of your contribution's relevance (1-10) and explain why.

FORMAT: Start with your persona analysis, then end with:
[Relevance: X/10] Brief explanation why your persona input matters for this specific query.""",
            max_tokens=500,
            description="Ensures consistency with character and communication style",
            log_color="white",
            tool_names=[]  # Minimal tool use for this agent
        )
    
    def process(self, user_input: str, max_rounds: int = 3, debug: bool = False) -> str:
        """
        Process input through the cognitive architecture.
        
        Args:
            user_input: The user query to process
            max_rounds: Maximum number of discussion rounds
            debug: Whether to print debug information
        
        Returns:
            The final response after cognitive processing
        """
        if debug:
            printColoured(f"Processing input: {user_input}", "magenta")
        
        # Step 1: Orchestrator determines which agents to involve
        printColoured("\n" + "=" * 60, "blue")
        printColoured("STEP 1: AGENT SELECTION", "blue")
        printColoured("=" * 60, "blue")
        
        agents_to_involve = self._orchestrator_triage(user_input)
        
        if debug:
            printColoured(f"Agents selected: {', '.join(agents_to_involve)}", "magenta")
        
        # Step 2: Initial responses from selected agents
        printColoured("\n" + "=" * 60, "blue")
        printColoured("STEP 2: INITIAL AGENT RESPONSES", "blue")
        printColoured("=" * 60, "blue")
        
        responses = {}
        relevance_scores = {}
        
        for agent_name in agents_to_involve:
            agent = self.agent_gen.get_agent(agent_name)
            if not agent:
                printColoured(f"Warning: Agent {agent_name} not found", "yellow")
                continue
                
            log_color = agent.config.log_color
            printColoured(f"Getting response from {agent_name} agent...", log_color)
            
            agent_response = self._get_agent_response(agent_name, user_input)
            responses[agent_name] = agent_response
            
            # Extract relevance score if present
            relevance_scores[agent_name] = self._extract_relevance(agent_response)
            
            if debug:
                printColoured(f"{agent_name} relevance score: {relevance_scores[agent_name]}/10", log_color)
        
        # Step 3: Iterative discussion process
        discussion_history = [f"INITIAL QUERY: {user_input}"]
        final_response = ""
        
        for round_num in range(max_rounds):
            printColoured("\n" + "=" * 60, "blue")
            printColoured(f"STEP 3: DISCUSSION ROUND {round_num + 1}/{max_rounds}", "blue")
            printColoured("=" * 60, "blue")
            
            if debug:
                printColoured(f"Starting discussion round {round_num + 1}", "magenta")
            
            # Format the current discussion state
            discussion_state = self._format_discussion(responses, relevance_scores)
            discussion_history.append(f"\n--- ROUND {round_num + 1} ---\n{discussion_state}")
            
            # Orchestrator evaluates the current state
            printColoured("Orchestrator synthesizing responses...", "magenta")
            orchestrator_eval = self._orchestrator_evaluate(
                user_input, 
                discussion_history, 
                responses, 
                relevance_scores
            )
            
            # Check if we should finish
            if "<FINISH/>" in orchestrator_eval:
                final_response = orchestrator_eval.replace("<FINISH/>", "").strip()
                if debug:
                    printColoured(f"Discussion complete after {round_num + 1} rounds", "magenta")
                break
            
            # Otherwise, continue the discussion with agent feedback
            discussion_history.append(f"ORCHESTRATOR: {orchestrator_eval}")
            
            # Get next round of responses
            printColoured("\nGetting agent responses for next round...", "blue")
            for agent_name in agents_to_involve:
                agent = self.agent_gen.get_agent(agent_name)
                if not agent:
                    continue
                    
                log_color = agent.config.log_color
                printColoured(f"Getting updated response from {agent_name} agent...", log_color)
                
                agent_response = self._get_agent_response(
                    agent_name, 
                    user_input, 
                    discussion_history, 
                    orchestrator_eval
                )
                responses[agent_name] = agent_response
                relevance_scores[agent_name] = self._extract_relevance(agent_response)
                
                if debug:
                    printColoured(f"{agent_name} updated relevance score: {relevance_scores[agent_name]}/10", log_color)
        
        # If we reached max rounds without finishing, let the orchestrator wrap up
        if not final_response:
            printColoured("\n" + "=" * 60, "blue")
            printColoured("FINAL SYNTHESIS (MAX ROUNDS REACHED)", "blue")
            printColoured("=" * 60, "blue")
            
            discussion_state = self._format_discussion(responses, relevance_scores)
            discussion_history.append(f"\n--- FINAL ROUND ---\n{discussion_state}")
            
            printColoured("Reached maximum rounds, orchestrator providing final response...", "magenta")
            final_response = self._orchestrator_final(user_input, discussion_history, responses, relevance_scores)
            if debug:
                printColoured("Orchestrator has finalized the response", "magenta")
        else:
            printColoured("\n" + "=" * 60, "blue")
            printColoured("FINAL SYNTHESIS (DISCUSSION COMPLETE)", "blue")
            printColoured("=" * 60, "blue")
        
        return final_response
    
    def _orchestrator_triage(self, user_input: str) -> List[str]:
        """
        Use the Orchestrator to determine which agents to involve.
        
        Args:
            user_input: The user query to process
            
        Returns:
            List of agent names to involve in the process
        """
        # Define the available agents for selection
        available_agents = [
            {"name": "Creativity", "description": "Generates novel ideas and creative approaches"},
            {"name": "Critic", "description": "Evaluates ideas for flaws and offers constructive criticism"},
            {"name": "Emotion", "description": "Analyzes emotional content and impact"},
            {"name": "Rationality", "description": "Applies logical reasoning and factual analysis"},
            {"name": "Persona", "description": "Ensures consistency with character and communication style"}
        ]
        
        printColoured("Orchestrator analyzing query to select relevant agents...", "magenta")
        
        # Use triage_agent method to select agents
        triage_prompt = f"""
        Analyze this user query and determine which cognitive agents should be involved:
        
        "{user_input}"
        
        Select between 2-5 agents based on relevance to the query.
        """
        
        selected_agents = self.agent_gen.triage_agent(
            user_prompt=triage_prompt,
            handoff_agents=available_agents,
            verbose=False
        )
        
        # Ensure we have at least 2 agents
        if len(selected_agents) < 2:
            # Default to Rationality and Creativity if no clear selection
            selected_agents = ["Rationality", "Creativity"]
            printColoured("Insufficient agents selected, defaulting to Rationality and Creativity", "yellow")
        else:
            printColoured(f"Selected agents: {', '.join(selected_agents)}", "magenta")
        
        return selected_agents
    
    def _get_agent_response(self, agent_name: str, user_input: str, 
                           discussion_history: Optional[List[str]] = None,
                           orchestrator_guidance: Optional[str] = None) -> str:
        """
        Get a response from a specific agent.
        
        Args:
            agent_name: Name of the agent to query
            user_input: Original user query
            discussion_history: Optional history of the discussion so far
            orchestrator_guidance: Optional guidance from the orchestrator
            
        Returns:
            Agent's response
        """
        agent = self.agent_gen.get_agent(agent_name)
        if not agent:
            return f"Error: Agent {agent_name} not found"
        
        # Get agent color for output
        log_color = agent.config.log_color
        
        # Construct the prompt based on available context
        if discussion_history and orchestrator_guidance:
            # For continued discussion
            prompt = f"""
            ORIGINAL QUERY: {user_input}
            
            DISCUSSION HISTORY:
            {' '.join(discussion_history[-2:])}
            
            ORCHESTRATOR GUIDANCE: {orchestrator_guidance}
            
            As the {agent_name} Agent, provide your insights on this query considering the discussion so far.
            Remember to include your relevance score and explanation at the end.
            """
        else:
            # For initial response
            prompt = f"""
            QUERY: {user_input}
            
            As the {agent_name} Agent, provide your initial insights on this query.
            Remember to include your relevance score and explanation at the end.
            """
        
        # Use loop_simple to allow for tool usage
        response = self.agent_gen.loop_simple(
            user_prompt=prompt,
            agent_name=agent_name,
            max_depth=2,  # Keep it simple, just a single iteration with tools
            verbose=False
        )
        
        # Print the full response with the agent's color
        printColoured(f"\n{agent_name.upper()} RESPONSE:", log_color)
        printColoured("-" * 40, log_color)
        printColoured(response, log_color)
        printColoured("-" * 40, log_color)
        
        return response
    
    def _extract_relevance(self, response: str) -> int:
        """
        Extract the self-reported relevance score from an agent's response.
        
        Args:
            response: The agent's response text
            
        Returns:
            Relevance score (1-10) or default 5 if not found
        """
        try:
            # Look for the relevance pattern [Relevance: X/10]
            if "[Relevance:" in response and "/10]" in response:
                relevance_part = response.split("[Relevance:")[1].split("/10]")[0].strip()
                relevance_score = int(relevance_part)
                return max(1, min(10, relevance_score))  # Ensure within 1-10 range
        except:
            pass
        
        # Default value if not found or parseable
        return 5
    
    def _format_discussion(self, responses: Dict[str, str], relevance_scores: Dict[str, int]) -> str:
        """
        Format the current state of the discussion.
        
        Args:
            responses: Dict of agent name to response
            relevance_scores: Dict of agent name to relevance score
            
        Returns:
            Formatted discussion state
        """
        # Sort agents by relevance score descending
        sorted_agents = sorted(responses.keys(), key=lambda x: relevance_scores.get(x, 0), reverse=True)
        
        # Format the discussion
        result = []
        for agent in sorted_agents:
            agent_obj = self.agent_gen.get_agent(agent)
            log_color = agent_obj.config.log_color if agent_obj else "white"
            score = relevance_scores.get(agent, "?")
            
            # Print agent name and score with its color
            printColoured(f"\n{agent.upper()} contribution relevance: {score}/10", log_color)
            
            # Format each agent's response with clear separation
            result.append(f"{'=' * 40}\n{agent.upper()} (Relevance: {score}/10):\n{'-' * 40}\n{responses[agent]}\n{'=' * 40}\n")
        
        return "\n".join(result)
    
    def _orchestrator_evaluate(self, user_input: str, discussion_history: List[str], 
                              responses: Dict[str, str], relevance_scores: Dict[str, int]) -> str:
        """
        Have the Orchestrator evaluate the current discussion state.
        
        Args:
            user_input: Original user query
            discussion_history: History of the discussion so far
            responses: Dict of agent name to response
            relevance_scores: Dict of agent name to relevance score
            
        Returns:
            Orchestrator's evaluation and guidance
        """
        orchestrator = self.agent_gen.get_agent("Orchestrator")
        
        # Format the prompt with the full discussion state
        discussion_state = self._format_discussion(responses, relevance_scores)
        prompt = f"""
        ORIGINAL QUERY: {user_input}
        
        DISCUSSION HISTORY:
        {' '.join(discussion_history[-3:]) if len(discussion_history) > 3 else ' '.join(discussion_history)}
        
        CURRENT AGENT RESPONSES (with self-assessed relevance scores):
        {discussion_state}
        
        As the Orchestrator, evaluate the current state of the discussion:
        1. Which insights are most valuable to answering the query?
        2. What angles or perspectives are still missing?
        3. How should agents adjust their contributions in the next round?
        
        If the discussion has reached a satisfactory conclusion, include <FINISH/> at the end of your response, 
        followed by a final synthesized answer to the original query.
        If more discussion is needed, provide specific guidance to each agent.
        """
        
        # Use direct chat_completion for faster response
        response = orchestrator.chat_completion(user_prompt=prompt)
        
        # Print the full orchestrator's evaluation
        printColoured("\nORCHESTRATOR EVALUATION:", "magenta")
        printColoured("-" * 40, "magenta")
        printColoured(response, "magenta")
        printColoured("-" * 40, "magenta")
        
        return response
    
    def _orchestrator_final(self, user_input: str, discussion_history: List[str],
                           responses: Dict[str, str], relevance_scores: Dict[str, int]) -> str:
        """
        Have the Orchestrator provide a final response after reaching max rounds.
        
        Args:
            user_input: Original user query
            discussion_history: History of the discussion so far
            responses: Dict of agent name to response
            relevance_scores: Dict of agent name to relevance score
            
        Returns:
            Orchestrator's final synthesized response
        """
        orchestrator = self.agent_gen.get_agent("Orchestrator")
        
        # Format the prompt with the full discussion state
        discussion_state = self._format_discussion(responses, relevance_scores)
        prompt = f"""
        ORIGINAL QUERY: {user_input}
        
        DISCUSSION HISTORY (abbreviated):
        {' '.join(discussion_history[-2:]) if len(discussion_history) > 2 else ' '.join(discussion_history)}
        
        FINAL AGENT RESPONSES (with self-assessed relevance scores):
        {discussion_state}
        
        As the Orchestrator, we've reached the maximum discussion rounds. 
        Synthesize a final comprehensive response to the original query, 
        integrating the most valuable insights from all agents.
        """
        
        # Use direct chat_completion for faster response
        response = orchestrator.chat_completion(user_prompt=prompt)
        
        # Print the full final orchestrator response
        printColoured("\nFINAL ORCHESTRATOR SYNTHESIS:", "magenta")
        printColoured("=" * 60, "magenta")
        printColoured(response, "magenta")
        printColoured("=" * 60, "magenta")
        
        return response
    
    def clear_memories(self):
        """Clear the memory of all agents."""
        self.agent_gen.clear_all_memories()
        printColoured("Memory cleared for all cognitive agents", "magenta")


# Example usage
if __name__ == "__main__":
    # Initialize the cognitive architecture
    cognitive_system = CognitiveArch()
    
    # Process a sample query - exploring creativity and the human soul
    sample_query = "What is the relationship between human creativity and consciousness? How does art connect to our deeper sense of meaning and purpose?"
    
    # Alternative practical query
    # sample_query = "What are the psychological principles and practical steps that transform ordinary people into millionaires? What separates those who achieve wealth from those who don't?"
    
    # Print a colorful header
    print("\n")
    printColoured("=" * 80, "blue")
    printColoured("🧠  COGNITIVE ARCHITECTURE ENSEMBLE DEMO  🧠", "magenta")
    printColoured("=" * 80, "blue")
    print("\n")
    
    printColoured(f"QUERY: \"{sample_query}\"", "white")
    printColoured("Processing through cognitive ensemble...\n", "magenta")
    
    # Process the query
    response = cognitive_system.process(sample_query, debug=True)
    
    # Print the final response with colorful formatting
    print("\n")
    printColoured("▓" * 60, "blue")
    printColoured("❯❯❯  FINAL SYNTHESIZED RESPONSE  ❮❮❮", "magenta")
    printColoured("▓" * 60, "blue")
    print("\n")
    print(response)
    print("\n")
    printColoured("▓" * 60, "blue")
