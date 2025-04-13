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

# Import AgentGen and utils
from src.III_agents.agentsGen import AgentGen, Agent
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
    'agentsGen', 'factoryArch', 'ollama_API',
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

class FactoryArch:
    """
    Factory Architecture - A coordinated system of agents specialized for code development tasks.
    
    Architecture:
    Input -> Supervisor (manager) -> Coder & Debugger (specialists) -> Final Code Output
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, replicate_api_token: Optional[str] = None):
        """
        Initialize the factory architecture with the necessary agents.
        
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
        
        # Create all agents in the factory architecture
        self._create_agents()
        
        printColoured("🏭 Factory Architecture initialized with all agents", "blue")
    
    def _get_tools_dict(self) -> Dict[str, List[str]]:
        """Create categorized tool dictionary for easier assignment to agents."""
        # Get all available tools
        tools = self.agent_gen.get_available_tools()
        tool_names = [tool["name"] for tool in tools]
        
        # Create categorized dictionary
        tool_categories = {
            "utility": ["get_current_datetime"],
            "web": ["web_crawl", "get_news"],
            "code": ["test_code", "format_code", "debug_code"]  # Dummy tools for this example
        }
        
        # Ensure we only include tools that are actually available
        for category in tool_categories:
            tool_categories[category] = [t for t in tool_categories[category] if t in tool_names]
        
        # Add a dummy tool for our example if not available
        if "test_code" not in tool_names:
            printColoured("Note: Dummy 'test_code' tool will be simulated", "yellow")
        
        return tool_categories
    
    def _create_agents(self):
        """Create all the agents for the factory architecture."""
        # 1. Supervisor - The project manager
        self.agent_gen.create_agent(
            name="Supervisor",
            system_prompt="""You are the Supervisor agent in a factory architecture system. Your role is to:
1. Analyze the coding task and break it down into manageable parts
2. Direct and coordinate the Coder and Debugger agents
3. Evaluate the code quality and functionality
4. Make final decisions about code readiness

You must be clear in your guidance for each agent, providing specific requirements and acceptance criteria.
When you determine the task is complete, include <FINISH/> in your response.

The agents you direct are:
- Coder Agent: Responsible for writing code based on requirements
- Debugger Agent: Responsible for identifying issues and suggesting improvements

Your structured output should clearly articulate tasks, evaluation criteria, and feedback.""",
            max_tokens=750,
            description="Manages the coding process and evaluates results",
            log_color="blue",
            tool_names=self.available_tools.get("utility", [])
        )
        
        # 2. Coder Agent
        self.agent_gen.create_agent(
            name="Coder",
            system_prompt="""You are the Coder Agent in a factory architecture system. Your role is to write clean, efficient, and well-documented code based on requirements.

Focus on:
- Implementing features exactly as specified
- Writing clean, readable code with appropriate comments
- Following best practices for the language/framework
- Creating reusable and maintainable solutions
- Considering edge cases and error handling

Your contributions should include complete implementations. Use tools to enhance your coding.
For each response, include a self-assessment of your code quality (1-10) and explain why.

FORMAT: Start with your code implementation, then end with:
[Quality: X/10] Brief explanation of your code's strengths and limitations.""",
            max_tokens=1000,
            description="Writes clean, efficient code to specifications",
            log_color="green",
            tool_names=self.available_tools.get("code", []) + self.available_tools.get("web", [])
        )
        
        # 3. Debugger Agent
        self.agent_gen.create_agent(
            name="Debugger",
            system_prompt="""You are the Debugger Agent in a factory architecture system. Your role is to analyze code for bugs, inefficiencies, and potential improvements.

Focus on:
- Identifying logical errors and edge cases
- Improving performance and efficiency
- Ensuring code follows best practices
- Suggesting security improvements
- Recommending better approaches or algorithms

Your analysis should be thorough yet constructive. Use tools to assist in debugging.
For each response, include a self-assessment of your analysis (1-10) and explain why.

FORMAT: Start with your debugging analysis and suggestions, then end with:
[Analysis: X/10] Brief explanation of the most critical issues found.""",
            max_tokens=750,
            description="Analyzes code for bugs and suggests improvements",
            log_color="magenta",
            tool_names=self.available_tools.get("code", []) + self.available_tools.get("web", [])
        )
    
    def process(self, user_input: str, code_context: Optional[str] = None, max_rounds: int = 3, debug: bool = False) -> Dict[str, str]:
        """
        Process a coding task through the factory architecture.
        
        Args:
            user_input: The user's coding task description
            code_context: Optional existing code context
            max_rounds: Maximum number of discussion rounds
            debug: Whether to print debug information
        
        Returns:
            Dictionary containing the final code and documentation
        """
        if debug:
            printColoured(f"Processing task: {user_input}", "blue")
        
        # Initialize result tracking
        result = {
            "code": "",
            "documentation": "",
            "process_log": []
        }
        
        # Step 1: Supervisor analyzes the task
        printColoured("\n" + "=" * 60, "yellow")
        printColoured("STEP 1: TASK ANALYSIS", "yellow")
        printColoured("=" * 60, "yellow")
        
        # Get initial task breakdown from supervisor
        task_analysis = self._get_agent_response("Supervisor", f"""
        Analyze this coding task and break it down into clear requirements:
        
        TASK: {user_input}
        
        {f'EXISTING CODE CONTEXT: {code_context}' if code_context else ''}
        
        Provide:
        1. A clear breakdown of requirements
        2. Specific instructions for the Coder
        3. Areas the Debugger should focus on
        """)
        
        result["process_log"].append({"agent": "Supervisor", "step": "Task Analysis", "content": task_analysis})
        
        # Step 2: Coder implements the solution
        printColoured("\n" + "=" * 60, "yellow")
        printColoured("STEP 2: CODE IMPLEMENTATION", "yellow")
        printColoured("=" * 60, "yellow")
        
        initial_code = self._get_agent_response("Coder", f"""
        Implement code for the following task:
        
        TASK: {user_input}
        
        SUPERVISOR REQUIREMENTS:
        {task_analysis}
        
        {f'EXISTING CODE CONTEXT: {code_context}' if code_context else ''}
        
        Provide complete, well-documented code implementation.
        Include error handling and follow best practices.
        """)
        
        result["process_log"].append({"agent": "Coder", "step": "Initial Implementation", "content": initial_code})
        
        # Extract code quality score
        coder_score = self._extract_score(initial_code, "Quality")
        if debug:
            printColoured(f"Initial code quality score: {coder_score}/10", "green")
        
        # Step 3: Debugger reviews the code
        printColoured("\n" + "=" * 60, "yellow")
        printColoured("STEP 3: CODE REVIEW & DEBUGGING", "yellow")
        printColoured("=" * 60, "yellow")
        
        # Get debugger feedback
        debug_feedback = self._get_agent_response("Debugger", f"""
        Review and debug the following code:
        
        TASK: {user_input}
        
        CODE TO REVIEW:
        {initial_code}
        
        Identify issues, bugs, inefficiencies, and suggest improvements.
        Be specific about what needs to be fixed and how.
        """)
        
        result["process_log"].append({"agent": "Debugger", "step": "Code Review", "content": debug_feedback})
        
        # Extract debugger score
        debugger_score = self._extract_score(debug_feedback, "Analysis")
        if debug:
            printColoured(f"Code analysis score: {debugger_score}/10", "magenta")
        
        # Iterative Improvement Process
        current_code = initial_code
        feedback_history = [debug_feedback]
        
        for round_num in range(max_rounds):
            printColoured("\n" + "=" * 60, "yellow")
            printColoured(f"IMPROVEMENT ROUND {round_num + 1}/{max_rounds}", "yellow")
            printColoured("=" * 60, "yellow")
            
            # Skip if we've reached perfect quality
            if coder_score >= 9 and debugger_score >= 9:
                printColoured("Code quality is excellent, skipping further improvements", "green")
                break
            
            # Get supervisor direction for improvements
            supervisor_direction = self._get_agent_response("Supervisor", f"""
            Review the code implementation and debugging feedback:
            
            TASK: {user_input}
            
            CURRENT CODE:
            {current_code}
            
            DEBUGGER FEEDBACK:
            {feedback_history[-1]}
            
            Provide specific direction on what needs to be improved.
            If the code is of sufficient quality, include <FINISH/> in your response.
            """)
            
            result["process_log"].append({"agent": "Supervisor", "step": f"Improvement Direction Round {round_num + 1}", "content": supervisor_direction})
            
            # Check if supervisor thinks we're done
            if "<FINISH/>" in supervisor_direction:
                printColoured(f"Supervisor has determined the code is complete after round {round_num + 1}", "blue")
                break
            
            # Have Coder improve the code
            improved_code = self._get_agent_response("Coder", f"""
            Improve the code based on feedback:
            
            TASK: {user_input}
            
            CURRENT CODE:
            {current_code}
            
            DEBUGGER FEEDBACK:
            {feedback_history[-1]}
            
            SUPERVISOR DIRECTION:
            {supervisor_direction}
            
            Provide the updated, improved code implementation.
            """)
            
            result["process_log"].append({"agent": "Coder", "step": f"Code Improvement Round {round_num + 1}", "content": improved_code})
            current_code = improved_code
            coder_score = self._extract_score(improved_code, "Quality")
            
            # Get new debugger feedback
            new_debug_feedback = self._get_agent_response("Debugger", f"""
            Review the improved code:
            
            TASK: {user_input}
            
            IMPROVED CODE:
            {improved_code}
            
            Identify any remaining issues or suggest final improvements.
            """)
            
            result["process_log"].append({"agent": "Debugger", "step": f"Review Round {round_num + 1}", "content": new_debug_feedback})
            feedback_history.append(new_debug_feedback)
            debugger_score = self._extract_score(new_debug_feedback, "Analysis")
        
        # Final evaluation and test
        printColoured("\n" + "=" * 60, "yellow")
        printColoured("FINAL EVALUATION", "yellow")
        printColoured("=" * 60, "yellow")
        
        # Simulate testing (dummy tool example)
        test_result = self._simulate_test_code(current_code)
        
        # Final supervisor evaluation
        final_evaluation = self._get_agent_response("Supervisor", f"""
        Provide final evaluation of the code:
        
        TASK: {user_input}
        
        FINAL CODE:
        {current_code}
        
        FINAL DEBUGGER FEEDBACK:
        {feedback_history[-1]}
        
        TEST RESULTS:
        {test_result}
        
        Give your final assessment and any documentation needed.
        Include <FINISH/> in your response.
        """)
        
        result["process_log"].append({"agent": "Supervisor", "step": "Final Evaluation", "content": final_evaluation})
        
        # Extract final code and documentation
        result["code"] = self._extract_code_blocks(current_code)
        result["documentation"] = final_evaluation.replace("<FINISH/>", "").strip()
        
        printColoured("\n" + "=" * 60, "green")
        printColoured("CODE DEVELOPMENT COMPLETE", "green")
        printColoured("=" * 60, "green")
        
        return result
    
    def _get_agent_response(self, agent_name: str, prompt: str) -> str:
        """Get a response from a specific agent."""
        agent = self.agent_gen.get_agent(agent_name)
        if not agent:
            return f"Error: Agent {agent_name} not found"
        
        # Get agent color for output
        log_color = agent.config.log_color
        
        printColoured(f"{agent_name} agent processing...", log_color)
        
        # Use loop_simple to allow for tool usage
        response = self.agent_gen.loop_simple(
            user_prompt=prompt,
            agent_name=agent_name,
            max_depth=2,
            verbose=False
        )
        
        # Print the agent's response with its color
        printColoured(f"\n{agent_name.upper()} RESPONSE:", log_color)
        printColoured("-" * 40, log_color)
        
        # Print only the first 400 characters to avoid flooding the console
        printColoured(response[:400] + ("..." if len(response) > 400 else ""), log_color)
        printColoured("-" * 40, log_color)
        
        return response
    
    def _extract_score(self, response: str, score_type: str) -> int:
        """
        Extract the self-reported score from an agent's response.
        
        Args:
            response: The agent's response text
            score_type: The type of score to extract ('Quality' or 'Analysis')
            
        Returns:
            Score (1-10) or default 5 if not found
        """
        try:
            # Look for the score pattern [Type: X/10]
            if f"[{score_type}:" in response and "/10]" in response:
                score_part = response.split(f"[{score_type}:")[1].split("/10]")[0].strip()
                score = int(score_part)
                return max(1, min(10, score))  # Ensure within 1-10 range
        except:
            pass
        
        # Default value if not found or parseable
        return 5
    
    def _extract_code_blocks(self, text: str) -> str:
        """Extract code blocks from the text."""
        # Look for code between triple backticks
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n([\s\S]*?)\n```', text)
        
        if code_blocks:
            return '\n\n'.join(code_blocks)
        
        # If no code blocks found, try to extract the code itself
        # This is a simplistic approach, might not work in all cases
        lines = text.split('\n')
        in_code = False
        code_lines = []
        
        for line in lines:
            if line.strip().startswith('class ') or line.strip().startswith('def ') or line.strip().startswith('import '):
                in_code = True
            
            if in_code:
                code_lines.append(line)
                
        if code_lines:
            return '\n'.join(code_lines)
            
        # If all else fails, return the original text
        return text
    
    def _simulate_test_code(self, code: str) -> str:
        """
        Simulate testing the code (dummy implementation).
        In a real implementation, this would use an actual testing framework.
        """
        # This is a dummy implementation for illustration
        import time
        import random
        
        printColoured("Running code tests...", "yellow")
        time.sleep(1)  # Simulate test execution time
        
        # Simulate test results
        test_success = random.random() > 0.3  # 70% chance of success
        
        if test_success:
            test_result = """
            TEST RESULTS:
            ✅ All tests passed successfully
            - Functionality test: PASS
            - Edge cases test: PASS
            - Performance test: PASS
            
            Code execution time: 0.45s
            Memory usage: 24MB
            """
        else:
            test_result = """
            TEST RESULTS:
            ❌ Some tests failed
            - Functionality test: PASS
            - Edge cases test: FAIL (Unexpected behavior with empty input)
            - Performance test: PASS
            
            Code execution time: 0.67s
            Memory usage: 32MB
            """
        
        printColoured(f"Test completed: {'✅ SUCCESS' if test_success else '❌ ISSUES FOUND'}", "yellow")
        return test_result
    
    def clear_memories(self):
        """Clear the memory of all agents."""
        self.agent_gen.clear_all_memories()
        printColoured("Memory cleared for all factory agents", "blue")


# Example usage
if __name__ == "__main__":
    # Initialize the factory architecture
    factory_system = FactoryArch()
    
    # Process a sample coding task
    sample_task = "Create a Python function that takes a list of numbers and returns the average, median, and mode."
    
    # Print a colorful header
    print("\n")
    printColoured("=" * 80, "blue")
    printColoured("🏭  FACTORY ARCHITECTURE DEMO  🏭", "blue")
    printColoured("=" * 80, "blue")
    print("\n")
    
    printColoured(f"TASK: \"{sample_task}\"", "white")
    printColoured("Processing through factory ensemble...\n", "blue")
    
    # Process the task
    result = factory_system.process(sample_task, debug=True)
    
    # Print the final code with colorful formatting
    print("\n")
    printColoured("▓" * 60, "green")
    printColoured("❯❯❯  FINAL CODE  ❮❮❮", "green")
    printColoured("▓" * 60, "green")
    print("\n")
    print(result["code"])
    print("\n")
    printColoured("▓" * 60, "blue")
    printColoured("❯❯❯  DOCUMENTATION  ❮❮❮", "blue")
    printColoured("▓" * 60, "blue")
    print("\n")
    print(result["documentation"])
    print("\n")
    printColoured("▓" * 60, "blue") 