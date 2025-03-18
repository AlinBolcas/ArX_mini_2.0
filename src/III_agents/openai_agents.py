import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any, Union, Type, Callable, TypeVar, Generic, Awaitable, Set
from enum import Enum
from pydantic import BaseModel, Field
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openai_agents")

# Direct imports from OpenAI Agents SDK with correct class names
from agents import (
    Agent, Runner, Tool, ModelSettings, 
    InputGuardrail, OutputGuardrail, GuardrailFunctionOutput,
    models, result as result_module, 
    OpenAIChatCompletionsModel, OpenAIResponsesModel
)

# Import from submodules
from agents.run_context import RunContextWrapper as Context
from agents.tracing import Span, Trace, add_trace_processor, set_tracing_disabled

# Define aliases for renamed or missing classes
Result = result_module.RunResult
# Missing classes we need to mimic
class ModelConfig:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

class ModelInterface:
    """Base class for model interfaces"""
    pass

class HandoffFilter:
    def __init__(self, filter_function):
        self.filter_function = filter_function

class HandoffPrompt:
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template

class AgentOrchestrator:
    def __init__(self, name: str, instructions: str, agents: List[Agent], **kwargs):
        self.name = name
        self.instructions = instructions
        self.agents = agents
        self.kwargs = kwargs

# Alias for ProcessorConfig which doesn't seem to exist
class ProcessorConfig:
    @staticmethod
    def configure_default_processors():
        # This would actually set up the default OpenAI Dashboard tracing
        pass

# Type variables for generic typing
T = TypeVar('T', bound=BaseModel)
G = TypeVar('G')

class OpenAIAgentsAPI:
    """
    A comprehensive wrapper for the OpenAI Agents SDK that simplifies agent creation,
    configuration, execution, and orchestration.
    
    This class provides a unified interface for working with the OpenAI Agents SDK,
    making it easier to:
    - Create and configure individual agents
    - Define tools for agents to use
    - Implement guardrails for input and output validation
    - Orchestrate multiple agents with handoffs
    - Run agents with proper context management
    - Stream agent results
    - Trace agent execution
    
    The class follows a builder pattern for fluent API design, allowing for
    chained method calls to configure and run agents.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gpt-4o",
        enable_tracing: bool = True
    ):
        """
        Initialize the OpenAI Agents API wrapper.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var
            default_model: Default model to use for agents
            enable_tracing: Whether to enable tracing by default
        """
        # Set API key if provided, otherwise use environment variable
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAI API key provided or found in environment. "
                "Please provide an API key or set OPENAI_API_KEY in your environment."
            )
        
        # Ensure the API key is set in the environment for the SDK
        os.environ["OPENAI_API_KEY"] = self.api_key
        
        # Default settings
        self.default_model = default_model
        self.enable_tracing = enable_tracing
        
        # Store created agents for reuse
        self._agents: Dict[str, Agent] = {}
        
        # Initialize a default runner for convenience
        self.runner = Runner()
        
        # Configure tracing if enabled
        if enable_tracing:
            self._configure_tracing()
        
        logger.info(f"OpenAI Agents API initialized with default model: {default_model}")
    
    def _configure_tracing(self) -> None:
        """Configure tracing for the Agents SDK."""
        # Configure tracing with default settings
        # This uses the default OpenAI Dashboard tracing
        ProcessorConfig.configure_default_processors()
        logger.info("Tracing configured for OpenAI Dashboard")
    
    def create_agent(
        self,
        name: str,
        instructions: str,
        model: Optional[str] = None,
        tools: Optional[List[Tool]] = None,
        handoffs: Optional[List[Union[Agent, str]]] = None,
        handoff_description: Optional[str] = None,
        input_guardrails: Optional[List[InputGuardrail]] = None,
        output_guardrails: Optional[List[OutputGuardrail]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        model_settings: Optional[ModelSettings] = None,
        model_config: Optional[ModelConfig] = None,
        model_interface: Optional[ModelInterface] = None,
        handoff_filters: Optional[List[HandoffFilter]] = None,
        handoff_prompt: Optional[HandoffPrompt] = None,
        **kwargs
    ) -> Agent:
        """
        Create and configure an OpenAI Agent.
        
        Args:
            name: Name of the agent
            instructions: Instructions for the agent
            model: Model to use (defaults to instance default_model)
            tools: List of tools the agent can use
            handoffs: List of agents this agent can hand off to
            handoff_description: Description for when other agents consider handing off to this agent
            input_guardrails: List of input guardrails
            output_guardrails: List of output guardrails
            output_type: Pydantic model for structured output
            model_settings: Model settings for temperature, etc.
            model_config: Model configuration
            model_interface: Custom model interface
            handoff_filters: Filters for controlling when handoffs occur
            handoff_prompt: Custom prompt for handoff decisions
            **kwargs: Additional keyword arguments to pass to Agent constructor
            
        Returns:
            Configured Agent instance
        """
        # Resolve string references to agents in handoffs
        resolved_handoffs = None
        if handoffs:
            resolved_handoffs = []
            for handoff in handoffs:
                if isinstance(handoff, str):
                    if handoff in self._agents:
                        resolved_handoffs.append(self._agents[handoff])
                    else:
                        raise ValueError(f"Agent with name '{handoff}' not found for handoff")
                else:
                    resolved_handoffs.append(handoff)
        
        # Configure model if specified
        if not model_config and model:
            model_config = ModelConfig(model=model or self.default_model)
        
        # Create the agent
        agent = Agent(
            name=name,
            instructions=instructions,
            tools=tools,
            handoffs=resolved_handoffs,
            handoff_description=handoff_description,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            output_type=output_type,
            model_settings=model_settings,
            model_config=model_config or ModelConfig(model=self.default_model),
            model_interface=model_interface,
            handoff_filters=handoff_filters,
            handoff_prompt=handoff_prompt,
            **kwargs
        )
        
        # Store the agent for later reference
        self._agents[name] = agent
        
        logger.info(f"Created agent: {name}")
        return agent
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """Get a previously created agent by name."""
        return self._agents.get(name)
    
    def create_tool(
        self, 
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        is_cached: bool = False,
        is_remote: bool = False
    ) -> Tool:
        """
        Create a tool from a function for agents to use.
        
        Args:
            function: The function to wrap as a tool
            name: Optional custom name for the tool (defaults to function name)
            description: Optional custom description (defaults to function docstring)
            is_cached: Whether to cache results of this tool
            is_remote: Whether this tool makes remote calls
            
        Returns:
            Tool instance that can be passed to an agent
        """
        tool = Tool(
            function=function,
            name=name or function.__name__,
            description=description or function.__doc__ or f"Tool based on {function.__name__}",
            is_cached=is_cached,
            is_remote=is_remote
        )
        
        logger.info(f"Created tool: {tool.name}")
        return tool
    
    def create_input_guardrail(
        self, 
        guardrail_function: Callable[[Any, Agent, str], Awaitable[GuardrailFunctionOutput]]
    ) -> InputGuardrail:
        """
        Create an input guardrail for validating user input to agents.
        
        Args:
            guardrail_function: Async function that validates input and returns GuardrailFunctionOutput
            
        Returns:
            InputGuardrail instance
        """
        return InputGuardrail(guardrail_function=guardrail_function)
    
    def create_output_guardrail(
        self, 
        guardrail_function: Callable[[Any, Agent, str], Awaitable[GuardrailFunctionOutput]]
    ) -> OutputGuardrail:
        """
        Create an output guardrail for validating agent output.
        
        Args:
            guardrail_function: Async function that validates output and returns GuardrailFunctionOutput
            
        Returns:
            OutputGuardrail instance
        """
        return OutputGuardrail(guardrail_function=guardrail_function)
    
    def create_guardrail_output(
        self, 
        output_info: Any,
        tripwire_triggered: bool = False
    ) -> GuardrailFunctionOutput:
        """
        Create a guardrail function output for use in custom guardrails.
        
        Args:
            output_info: Information about the guardrail check result
            tripwire_triggered: Whether the guardrail was triggered
            
        Returns:
            GuardrailFunctionOutput instance
        """
        return GuardrailFunctionOutput(
            output_info=output_info,
            tripwire_triggered=tripwire_triggered
        )
    
    async def run_agent(
        self,
        agent: Union[Agent, str],
        input_data: str,
        context: Optional[Context] = None,
        stream: bool = False,
        callback: Optional[Callable[[Any], None]] = None
    ) -> Result:
        """
        Run an agent with the given input.
        
        Args:
            agent: Agent instance or name of previously created agent
            input_data: Input data to send to the agent
            context: Optional context for the agent run
            stream: Whether to stream results (if callback provided)
            callback: Optional callback function for streaming events
            
        Returns:
            Result object containing agent output
        """
        # Resolve agent if name provided
        if isinstance(agent, str):
            if agent in self._agents:
                agent = self._agents[agent]
            else:
                raise ValueError(f"Agent with name '{agent}' not found")
        
        # Run the agent
        if stream and callback:
            # Run with streaming
            result = await Runner.run(
                agent, 
                input_data, 
                context=context,
                stream=True
            )
            
            # Process streaming events
            async for event in result.stream():
                callback(event)
            
            return result
        else:
            # Run without streaming
            result = await Runner.run(
                agent, 
                input_data, 
                context=context
            )
            return result
    
    def run_agent_sync(
        self,
        agent: Union[Agent, str],
        input_data: str,
        context: Optional[Context] = None
    ) -> Result:
        """
        Run an agent synchronously with the given input.
        
        Args:
            agent: Agent instance or name of previously created agent
            input_data: Input data to send to the agent
            context: Optional context for the agent run
            
        Returns:
            Result object containing agent output
        """
        # Create and run event loop if we're not in one already
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an event loop, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the agent
        try:
            return loop.run_until_complete(
                self.run_agent(agent, input_data, context)
            )
        finally:
            # Don't close the loop if it was already running
            if not loop.is_running():
                loop.close()
    
    class GuardrailBuilder(Generic[T]):
        """
        Builder class for creating guardrails with a fluent API.
        """
        def __init__(self, parent: 'OpenAIAgentsAPI', output_type: Type[T]):
            self.parent = parent
            self.output_type = output_type
            self.guardrail_agent: Optional[Agent] = None
        
        def with_agent(self, agent: Agent) -> 'OpenAIAgentsAPI.GuardrailBuilder[T]':
            """Set a specific agent to use for the guardrail check."""
            self.guardrail_agent = agent
            return self
        
        def with_new_agent(
            self, 
            name: str, 
            instructions: str,
            **kwargs
        ) -> 'OpenAIAgentsAPI.GuardrailBuilder[T]':
            """Create a new agent for the guardrail check."""
            self.guardrail_agent = self.parent.create_agent(
                name=name,
                instructions=instructions,
                output_type=self.output_type,
                **kwargs
            )
            return self
            
        def build_input_guardrail(self) -> InputGuardrail:
            """Build an input guardrail with the configured settings."""
            if not self.guardrail_agent:
                raise ValueError("Guardrail agent not specified. Call with_agent() or with_new_agent() first.")
            
            # Create guardrail function
            async def guardrail_function(ctx, agent, input_data):
                agent_to_use = self.guardrail_agent
                result = await Runner.run(agent_to_use, input_data, context=ctx.context)
                final_output = result.final_output_as(self.output_type)
                # Default implementation checks if 'is_allowed' field exists
                # Otherwise, creates a basic condition whether the guardrail should trigger
                if hasattr(final_output, 'is_allowed'):
                    tripwire_triggered = not final_output.is_allowed
                elif hasattr(final_output, 'should_block'):
                    tripwire_triggered = final_output.should_block
                else:
                    # No standard field found, don't trigger by default
                    tripwire_triggered = False
                    
                return GuardrailFunctionOutput(
                    output_info=final_output,
                    tripwire_triggered=tripwire_triggered,
                )
            
            return InputGuardrail(guardrail_function=guardrail_function)
        
        def build_output_guardrail(self) -> OutputGuardrail:
            """Build an output guardrail with the configured settings."""
            if not self.guardrail_agent:
                raise ValueError("Guardrail agent not specified. Call with_agent() or with_new_agent() first.")
            
            # Create guardrail function
            async def guardrail_function(ctx, agent, output_data):
                agent_to_use = self.guardrail_agent
                # Prepare a prompt that includes the output to check
                guardrail_prompt = f"Please evaluate the following output: {output_data}"
                result = await Runner.run(agent_to_use, guardrail_prompt, context=ctx.context)
                final_output = result.final_output_as(self.output_type)
                # Similar logic as input guardrail
                if hasattr(final_output, 'is_allowed'):
                    tripwire_triggered = not final_output.is_allowed
                elif hasattr(final_output, 'should_block'):
                    tripwire_triggered = final_output.should_block
                else:
                    tripwire_triggered = False
                    
                return GuardrailFunctionOutput(
                    output_info=final_output,
                    tripwire_triggered=tripwire_triggered,
                )
            
            return OutputGuardrail(guardrail_function=guardrail_function)
    
    def create_guardrail_builder(self, output_type: Type[T]) -> GuardrailBuilder[T]:
        """
        Create a guardrail builder for fluent guardrail creation.
        
        Args:
            output_type: Pydantic model for guardrail output type
            
        Returns:
            GuardrailBuilder instance
        """
        return self.GuardrailBuilder(self, output_type)
    
    def create_context(self, **kwargs) -> Context:
        """
        Create a context object for agent runs.
        
        Args:
            **kwargs: Key-value pairs to add to the context
            
        Returns:
            Context object
        """
        return Context(**kwargs)
    
    def create_orchestrator(
        self,
        name: str,
        instructions: str,
        agents: List[Agent],
        **kwargs
    ) -> AgentOrchestrator:
        """
        Create an orchestrator for managing multiple agents.
        
        Args:
            name: Name of the orchestrator
            instructions: Instructions for the orchestrator
            agents: List of agents to orchestrate
            **kwargs: Additional arguments for the orchestrator
            
        Returns:
            AgentOrchestrator instance
        """
        orchestrator = AgentOrchestrator(
            name=name,
            instructions=instructions,
            agents=agents,
            **kwargs
        )
        
        logger.info(f"Created orchestrator: {name} with {len(agents)} agents")
        return orchestrator
    
    def create_chat_model(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> OpenAIChatCompletionsModel:
        """
        Create an OpenAI Chat Completions model interface.
        
        Args:
            model: Model name (defaults to instance default)
            temperature: Temperature setting (0-2)
            max_tokens: Maximum tokens to generate
            
        Returns:
            OpenAIChatCompletionsModel instance
        """
        return OpenAIChatCompletionsModel(
            model=model or self.default_model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def create_responses_model(
        self,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_output_tokens: Optional[int] = None,
    ) -> OpenAIResponsesModel:
        """
        Create an OpenAI Responses model interface.
        
        Args:
            model: Model name (defaults to instance default)
            temperature: Temperature setting (0-2)
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            OpenAIResponsesModel instance
        """
        return OpenAIResponsesModel(
            model=model or self.default_model,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
    
    def create_handoff_filter(
        self,
        filter_function: Callable[[Any, Agent, str, Set[Agent]], Awaitable[bool]]
    ) -> HandoffFilter:
        """
        Create a handoff filter for controlling when handoffs occur.
        
        Args:
            filter_function: Async function that decides whether a handoff should occur
            
        Returns:
            HandoffFilter instance
        """
        return HandoffFilter(filter_function=filter_function)
    
    def create_handoff_prompt(
        self,
        prompt_template: str
    ) -> HandoffPrompt:
        """
        Create a custom handoff prompt for handoff decisions.
        
        Args:
            prompt_template: Template string for the handoff prompt
            
        Returns:
            HandoffPrompt instance
        """
        return HandoffPrompt(prompt_template=prompt_template)
    
    async def get_result_as_model(self, result: Result, model_type: Type[T]) -> T:
        """
        Get a result as a specific Pydantic model type.
        
        Args:
            result: Result from agent run
            model_type: Pydantic model class to convert to
            
        Returns:
            Instance of the specified model type
        """
        return result.final_output_as(model_type)
    
    def get_result_as_model_sync(self, result: Result, model_type: Type[T]) -> T:
        """
        Synchronously get a result as a specific Pydantic model type.
        
        Args:
            result: Result from agent run
            model_type: Pydantic model class to convert to
            
        Returns:
            Instance of the specified model type
        """
        return result.final_output_as(model_type)


# Example usage
if __name__ == "__main__":
    import asyncio
    from pydantic import BaseModel
    
    # Define output models
    class HomeworkOutput(BaseModel):
        is_homework: bool
        reasoning: str
    
    class MathAnswer(BaseModel):
        equation: str
        steps: List[str]
        answer: float
        explanation: str
    
    async def main():
        # Initialize the API
        api = OpenAIAgentsAPI(default_model="gpt-4o")
        
        # Define a tool
        def calculate_square_root(number: float) -> float:
            """Calculate the square root of a number."""
            import math
            return math.sqrt(number)
        
        square_root_tool = api.create_tool(calculate_square_root)
        
        # Create a guardrail
        guardrail_builder = api.create_guardrail_builder(HomeworkOutput)
        
        guardrail_agent = api.create_agent(
            name="Homework Check",
            instructions="You evaluate if the user is asking about homework. Return is_homework=True only if the question is appropriate to answer.",
            output_type=HomeworkOutput
        )
        
        input_guardrail = guardrail_builder.with_agent(guardrail_agent).build_input_guardrail()
        
        # Create specialist agents
        math_agent = api.create_agent(
            name="Math Tutor",
            instructions="You help with math problems. Show your work step by step.",
            tools=[square_root_tool],
            handoff_description="Specialist for math questions",
            output_type=MathAnswer
        )
        
        history_agent = api.create_agent(
            name="History Tutor",
            instructions="You provide assistance with historical questions. Explain context clearly.",
            handoff_description="Specialist for history questions"
        )
        
        # Create a triage agent with handoffs
        triage_agent = api.create_agent(
            name="Triage Agent",
            instructions="You direct questions to the appropriate specialist agent.",
            handoffs=[math_agent, history_agent],
            input_guardrails=[input_guardrail]
        )
        
        # Run the agent
        result = await api.run_agent(
            triage_agent,
            "What is the square root of 144?",
        )
        
        # Print the result
        print("\nFinal output:", result.final_output)
        
        # Get structured output if available
        try:
            math_answer = api.get_result_as_model_sync(result, MathAnswer)
            print(f"\nStructured answer: {math_answer.answer}")
            print(f"Explanation: {math_answer.explanation}")
        except Exception as e:
            print(f"Could not parse structured output: {e}")
    
    # Run the example
    asyncio.run(main()) 