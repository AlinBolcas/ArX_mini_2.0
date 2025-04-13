#!/usr/bin/env python
"""
AgentsGen.py

A comprehensive wrapper for the OpenAI Agents SDK (openai-agents) designed to 
facilitate rapid prototyping and development of hierarchical agentic systems.

This module provides a simplified interface over the core SDK primitives:
- Agent creation (basic, triage, structured output)
- Tool definition and integration (@function_tool)
- Handoffs between agents
- Input Guardrails for validation
- Workflow execution using the Runner

Advanced features:
- Multi-agent orchestration patterns (parallel, sequential, evaluate-improve)
- Streaming support for real-time agent outputs
- Tracing and visualization configuration
- Context management with custom objects
- Forced tool usage and model settings control
- Dynamic instructions and lifecycle hooks

Visit the OpenAI traces dashboard to visualize your agent runs:
https://platform.openai.com/traces

Based on documentation and examples from:
- https://github.com/openai/openai-agents-python
- https://openai.github.io/openai-agents-python/quickstart/
"""

import os
import asyncio
from typing import List, Dict, Any, Optional, Type, Callable, Union
from pydantic import BaseModel, Field

# Core SDK components
from agents import Agent, Runner, function_tool, InputGuardrail, GuardrailFunctionOutput, RunResult

# --- Example Pydantic Models for Structured Output & Guardrails ---

class SimpleResponse(BaseModel):
    """A basic Pydantic model for structured output."""
    answer: str = Field(..., description="The direct answer to the query.")
    confidence: int = Field(..., description="Confidence score (1-10) in the answer.")

class HomeworkCheckOutput(BaseModel):
    """Output schema for the homework checking guardrail."""
    is_homework_request: bool = Field(..., description="True if the user query appears to be a homework question.")
    reasoning: str = Field(..., description="Explanation for why it is or isn't considered homework.")
    topic: Optional[str] = Field(None, description="Identified topic of the query (e.g., Math, History).")

# --- Example Tool Definition ---

@function_tool
def get_current_year() -> int:
    """Returns the current calendar year."""
    from datetime import datetime
    print("🛠️ TOOL CALLED: get_current_year()")
    return datetime.now().year

# --- Wrapper Class ---

class AgentsGen:
    """
    Wrapper for the OpenAI Agents SDK to simplify agentic workflow creation.
    """
    def __init__(self, default_model: str = "gpt-4o-mini"):
        """
        Initialize the AgentsGen wrapper.

        Args:
            default_model (str): The default OpenAI model to use for agents if not specified.
        """
        self.default_model = default_model
        # Ensure API key is set for tracing and functionality
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ WARNING: OPENAI_API_KEY environment variable not set. SDK tracing and potentially other features might be disabled.")

    def create_agent(
        self,
        name: str,
        instructions: str,
        model: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        **kwargs: Any
    ) -> Agent:
        """
        Creates a basic Agent instance.

        Args:
            name (str): A descriptive name for the agent.
            instructions (str): The core instructions for the agent's behavior.
            model (Optional[str]): The specific OpenAI model to use (defaults to self.default_model).
            tools (Optional[List[Callable]]): A list of functions decorated with @function_tool.
            output_type (Optional[Type[BaseModel]]): Pydantic model for structured output. If provided, the agent will try to format its output accordingly.
            **kwargs: Additional keyword arguments passed directly to the Agent constructor.

        Returns:
            Agent: An initialized Agent object.
        """
        print(f"🧬 Creating Agent: {name}")
        return Agent(
            name=name,
            instructions=instructions,
            model=model or self.default_model,
            tools=tools or [],
            output_type=output_type,
            **kwargs
        )

    def create_triage_agent(
        self,
        name: str,
        instructions: str,
        handoff_agents: List[Agent],
        model: Optional[str] = None,
        input_guardrails: Optional[List[InputGuardrail]] = None,
        **kwargs: Any
    ) -> Agent:
        """
        Creates a Triage Agent designed to hand off tasks to other agents.

        Args:
            name (str): Name for the triage agent.
            instructions (str): Instructions for how to decide which agent to hand off to. Should reference the purpose of handoff_agents.
            handoff_agents (List[Agent]): A list of Agent objects that this agent can delegate tasks to. Ensure these agents have `handoff_description` set.
            model (Optional[str]): Specific model for the triage agent.
            input_guardrails (Optional[List[InputGuardrail]]): Guardrails to run before triage logic.
            **kwargs: Additional keyword arguments for the Agent constructor.

        Returns:
            Agent: An initialized Triage Agent object.
        """
        # More defensive printing that works even with dummy classes
        try:
            agent_names = [getattr(a, 'name', f"Agent_{i}") for i, a in enumerate(handoff_agents)]
            print(f"🧬 Creating Triage Agent: {name} (Handoffs: {agent_names})")
        except Exception as e:
            print(f"🧬 Creating Triage Agent: {name} (Handoffs: {len(handoff_agents)} agents)")
        
        # Validate that handoff agents have descriptions
        for agent in handoff_agents:
            if not getattr(agent, 'handoff_description', None):
                print(f"⚠️ WARNING: Handoff agent '{agent.name}' lacks a 'handoff_description'. Triage may be less effective.")
        
        return Agent(
            name=name,
            instructions=instructions,
            model=model or self.default_model,
            handoffs=handoff_agents,
            input_guardrails=input_guardrails or [],
            **kwargs
        )
        
    def define_input_guardrail(
        self,
        guardrail_agent: Agent,
        guardrail_function: Callable[[Any, Agent, Any], GuardrailFunctionOutput]
    ) -> InputGuardrail:
        """
        Defines an Input Guardrail using a dedicated agent and processing function.

        Args:
            guardrail_agent (Agent): An agent specifically designed to perform the guardrail check (often uses structured output).
            guardrail_function (Callable): An async function that takes (ctx, agent, input_data), runs the guardrail_agent, 
                                           parses its output, and returns a GuardrailFunctionOutput.

        Returns:
            InputGuardrail: The configured input guardrail object.
        """
        return InputGuardrail(guardrail_function=guardrail_function)

    async def run_workflow(
        self,
        entry_agent: Agent,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> RunResult:
        """
        DEPRECATED: Use Runner.run directly instead.
        This method remains for backward compatibility.
        """
        print("⚠️ run_workflow is deprecated, consider using Runner.run directly")
        return await Runner.run(entry_agent, user_input, context=context)

    def create_agent_with_dynamic_instructions(
        self,
        name: str,
        instructions_func: Callable[[Any, Agent], str],
        **kwargs
    ) -> Agent:
        """
        Creates an agent with dynamic instructions that can adapt based on context.
        
        Args:
            name: Name for the agent
            instructions_func: Function that returns instructions string based on context
            **kwargs: Additional Agent parameters
        """
        print(f"🧬 Creating Agent with dynamic instructions: {name}")
        return Agent(name=name, instructions=instructions_func, **kwargs)

    def create_agent_with_hooks(
        self,
        name: str,
        instructions: str,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_llm_start: Optional[Callable] = None,
        on_llm_end: Optional[Callable] = None,
        on_handoff: Optional[Callable] = None,
        **kwargs
    ) -> Agent:
        """Creates an agent with lifecycle hooks for monitoring execution flow."""
        hooks = {}
        if on_tool_start: hooks["on_tool_start"] = on_tool_start
        if on_tool_end: hooks["on_tool_end"] = on_tool_end
        if on_llm_start: hooks["on_llm_start"] = on_llm_start
        if on_llm_end: hooks["on_llm_end"] = on_llm_end
        if on_handoff: hooks["on_handoff"] = on_handoff
        
        return Agent(name=name, instructions=instructions, hooks=hooks, **kwargs)

    async def run_workflow_with_context_class(
        self,
        entry_agent: Agent,
        user_input: str,
        context_object: Any
    ) -> RunResult:
        """
        Runs a workflow with a typed context object that will be passed to all agents and tools.
        
        This allows for proper context typing and dependency injection throughout the agent workflow.
        The context_object can be any Python object - dataclass, Pydantic model, etc.
        """
        print(f"🚀 Running Workflow with typed context")
        try:
            result = await Runner.run(
                entry_agent, 
                user_input, 
                context=context_object
            )
            return result
        except Exception as e:
            print(f"❌ ERROR running workflow: {e}")
            import traceback
            traceback.print_exc()
            return RunResult(final_output=f"Workflow execution failed: {str(e)}")

    async def run_agents_in_parallel(
        self,
        agents: List[Agent],
        user_input: str,
        context: Optional[Any] = None
    ) -> List[RunResult]:
        """
        Run multiple agents in parallel using asyncio.gather.
        
        This is useful when you have multiple tasks that don't depend on each other
        and want to speed up execution.
        """
        tasks = [Runner.run(agent, user_input, context=context) for agent in agents]
        return await asyncio.gather(*tasks)

    async def run_sequential_chain(
        self,
        agents: List[Agent],
        initial_input: str,
        context: Optional[Any] = None,
        transform_output: Optional[Callable[[str], str]] = None
    ) -> RunResult:
        """
        Run a chain of agents sequentially, where each agent's output becomes input for the next.
        
        Args:
            agents: List of agents to chain
            initial_input: Starting input for the first agent
            transform_output: Optional function to transform output between agents
        """
        current_input = initial_input
        
        for i, agent in enumerate(agents):
            print(f"🔄 Running agent {i+1}/{len(agents)}: {agent.name}")
            result = await Runner.run(agent, current_input, context=context)
            
            # Get output for next agent
            current_input = result.final_output
            
            # Transform if needed
            if transform_output:
                current_input = transform_output(current_input)
        
        # Return the final result
        return result

    async def run_evaluate_improve_loop(
        self,
        worker_agent: Agent,
        evaluator_agent: Agent,
        user_input: str,
        max_iterations: int = 3,
        context: Optional[Any] = None
    ) -> RunResult:
        """
        Run an agent, evaluate its output, and improve until criteria are met.
        
        This implements the "run-evaluate-improve" pattern from the multi-agent documentation.
        
        Args:
            worker_agent: Agent that performs the task
            evaluator_agent: Agent that evaluates the output against criteria
            user_input: Initial user request
            max_iterations: Maximum number of improvement cycles
        """
        iteration = 0
        current_result = None
        
        while iteration < max_iterations:
            # Run the worker agent
            if iteration == 0:
                # First run with original input
                current_result = await Runner.run(worker_agent, user_input, context=context)
            else:
                # Subsequent runs with feedback
                improved_prompt = f"""
Original request: {user_input}

Previous attempt: 
{current_result.final_output}

Feedback from evaluator:
{evaluation_result.final_output}

Please improve your response based on this feedback.
"""
                current_result = await Runner.run(worker_agent, improved_prompt, context=context)
            
            # Run the evaluator
            evaluation_prompt = f"""
Original request: {user_input}

Response to evaluate:
{current_result.final_output}

Evaluate if this response fully addresses the request. If it's perfect, say "PERFECT".
If it needs improvement, explain what needs to be fixed.
"""
            evaluation_result = await Runner.run(evaluator_agent, evaluation_prompt, context=context)
            
            # Check if we're done
            if "PERFECT" in evaluation_result.final_output:
                print(f"✅ Evaluation passed after {iteration+1} iterations")
                break
            
            print(f"🔄 Iteration {iteration+1}/{max_iterations} completed. Continuing to improve.")
            iteration += 1
        
        return current_result

    def configure_tracing(
        self,
        export_enabled: bool = True,
        console_enabled: bool = False,
        trace_server: Optional[str] = None
    ) -> None:
        """
        Configure tracing for agent runs.
        
        Tracing allows you to visualize and debug your agent workflows in the
        OpenAI dashboard or a custom trace server.
        
        Args:
            export_enabled: Whether to export traces to OpenAI
            console_enabled: Whether to print trace info to console
            trace_server: Optional URL of a custom trace server
        """
        from agents.tracing import setup_tracing
        
        # Get trace server URL from env if not provided
        if not trace_server and os.getenv("OPENAI_TRACE_SERVER"):
            trace_server = os.getenv("OPENAI_TRACE_SERVER")
        
        # Configure tracing with the requested settings
        setup_tracing(
            export_enabled=export_enabled,
            console_enabled=console_enabled,
            trace_server=trace_server
        )
        
        print(f"🔍 Tracing configured: export={'✅' if export_enabled else '❌'}, console={'✅' if console_enabled else '❌'}")
        if trace_server:
            print(f"🔍 Using custom trace server: {trace_server}")
        else:
            print(f"🔍 Traces will be available at: https://platform.openai.com/traces")

    def get_trace_visualization_link(self, trace_id: str) -> str:
        """
        Get a link to visualize a specific trace in the OpenAI dashboard.
        
        Args:
            trace_id: The ID of the trace to visualize
            
        Returns:
            URL to view the trace
        """
        return f"https://platform.openai.com/traces/{trace_id}"

    def create_agent_with_forced_tool_use(
        self,
        name: str,
        instructions: str,
        tools: List[Callable],
        tool_choice: Union[str, bool] = "required",
        model: Optional[str] = None,
        **kwargs
    ) -> Agent:
        """
        Create an agent that is forced to use tools according to the specified tool_choice.
        
        Args:
            name: Agent name
            instructions: Agent instructions
            tools: List of tool functions
            tool_choice: How tools should be used:
                - "auto": Model decides whether to use tools
                - "required": Model must use a tool (but can choose which)
                - "none": Model must not use a tool
                - "tool_name": String matching a specific tool name to force that tool
            model: Optional model override
        """
        from agents import ModelSettings
        
        # Create model settings with tool_choice
        model_settings = ModelSettings(
            tool_choice=tool_choice,
        )
        
        return Agent(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model or self.default_model,
            model_settings=model_settings,
            **kwargs
        )

    async def run_workflow_with_streaming(
        self,
        entry_agent: Agent,
        user_input: str,
        stream_handler: Callable,
        context: Optional[Any] = None
    ) -> RunResult:
        """
        Run a workflow with streaming outputs processed by the provided handler.
        
        Args:
            entry_agent: Agent to start the workflow
            user_input: User's input query
            stream_handler: Callback function to process streaming events
            context: Optional context object
        """
        try:
            result = await Runner.run(
                entry_agent,
                user_input,
                context=context,
                stream=True,
                stream_handler=stream_handler
            )
            return result
        except Exception as e:
            print(f"❌ ERROR running workflow: {e}")
            import traceback
            traceback.print_exc()
            return RunResult(final_output=f"Workflow execution failed: {str(e)}")

# --- Example Usage ---

if __name__ == "__main__":
    # Initialize the AgentsGen wrapper
    agents_gen = AgentsGen(default_model="gpt-4o-mini")
    
    # Main function to run the cognitive swarm demo based on ArX architecture
    async def run_cognitive_swarm_demo():
        print("\n" + "="*80)
        print("🧠 COGNITIVE SWARM ARCHITECTURE DEMO".center(80))
        print("="*80)
        
        # Create the shared memory tool
        @function_tool
        def save_to_memory(key: str, value: str) -> str:
            """Save information to shared memory for other agents to access."""
            memory[key] = value
            print(f"💾 MEMORY: Saved '{key}' to shared memory")
            return f"Memory updated: {key}"
        
        @function_tool
        def get_from_memory(key: str) -> Dict[str, Any]:
            """Retrieve information from shared memory."""
            value = memory.get(key, "No data found")
            print(f"💾 MEMORY: Retrieved '{key}' from shared memory")
            return {"key": key, "value": value}
        
        @function_tool
        def get_all_memory() -> Dict[str, Any]:
            """Get all stored information from shared memory."""
            print(f"💾 MEMORY: Retrieved all data from shared memory")
            return memory
        
        # Initialize shared memory
        memory = {}
        
        # Create the cognitive agents
        print("🧬 Creating cognitive agents...")
        
        # Core cognitive agents
        persona = agents_gen.create_agent(
            name="Persona Agent",
            instructions="""You are the Persona agent in the cognitive swarm. 
            Analyze queries from the user's perspective, considering their identity, background, and goals. 
            Focus on understanding the user's context and intent.
            When asked to analyze something, provide a brief but insightful perspective on the human aspects.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Considers the user's perspective and human aspects of queries"
        )
        
        creativity = agents_gen.create_agent(
            name="Creativity Agent",
            instructions="""You are the Creativity agent in the cognitive swarm.
            Generate innovative approaches and ideas that are outside conventional thinking.
            When asked to analyze something, provide creative, novel perspectives that others might miss.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Generates innovative approaches and unconventional ideas"
        )
        
        rationality = agents_gen.create_agent(
            name="Rationality Agent",
            instructions="""You are the Rationality agent in the cognitive swarm.
            Apply logical reasoning, critical thinking, and analytical perspectives.
            When asked to analyze something, focus on facts, data, logical implications, and rational frameworks.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Applies logical reasoning and analytical perspectives"
        )
        
        emotion = agents_gen.create_agent(
            name="Emotion Agent",
            instructions="""You are the Emotion agent in the cognitive swarm.
            Focus on the emotional dimensions, feelings, and psychological aspects.
            When asked to analyze something, highlight emotional impacts, affective responses, and emotional implications.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Considers the emotional and psychological dimensions"
        )
        
        critic = agents_gen.create_agent(
            name="Critic Agent",
            instructions="""You are the Critic agent in the cognitive swarm.
            Evaluate proposals and ideas critically to identify flaws, weaknesses, and potential problems.
            When asked to analyze something, provide constructive criticism and point out potential issues.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Evaluates ideas critically to identify flaws and weaknesses"
        )
        
        future_vision = agents_gen.create_agent(
            name="Future Vision Agent",
            instructions="""You are the Future Vision agent in the cognitive swarm.
            Consider long-term implications, future scenarios, and emerging trends.
            When asked to analyze something, project how it might evolve over time and consider future impacts.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Projects long-term implications and future scenarios"
        )
        
        curiosity = agents_gen.create_agent(
            name="Curiosity Agent",
            instructions="""You are the Curiosity agent in the cognitive swarm.
            Identify important questions that should be asked and areas that need exploration.
            When asked to analyze something, highlight gaps in knowledge and generate important questions.
            """,
            tools=[save_to_memory, get_from_memory],
            handoff_description="Identifies important questions and areas needing exploration"
        )
        
        # Create the orchestrator agent that coordinates the cognitive swarm
        orchestrator = agents_gen.create_triage_agent(
            name="Orchestrator Agent",
            instructions="""You are the central Orchestrator agent that coordinates the cognitive swarm.
            
            Your role is to:
            1. Understand the user's query
            2. Determine which cognitive agents would provide the most valuable perspectives
            3. Route the query to those agents
            4. Synthesize their outputs into a comprehensive response
            
            You have these specialist agents available:
            - Persona: Considers user perspective and human aspects
            - Creativity: Generates innovative approaches
            - Rationality: Applies logical reasoning
            - Emotion: Analyzes emotional dimensions
            - Critic: Evaluates ideas critically
            - Future Vision: Considers long-term implications
            - Curiosity: Asks important questions
            
            For complex queries, use multiple agents to get diverse perspectives.
            For simpler queries, use only the most relevant agents.
            """,
            handoff_agents=[
                persona,
                creativity,
                rationality,
                emotion,
                critic,
                future_vision,
                curiosity
            ]
        )
        
        # Create the synthesizer agent that integrates all perspectives
        synthesizer = agents_gen.create_agent(
            name="Synthesizer Agent",
            instructions="""You are the Synthesizer agent that integrates diverse perspectives from the cognitive swarm.
            
            Your task is to:
            1. Retrieve all perspectives from memory
            2. Identify points of agreement and disagreement
            3. Weigh different viewpoints based on relevance and insight
            4. Create a comprehensive, balanced response that incorporates multiple perspectives
            5. Ensure the final output is coherent, insightful, and actionable
            
            Present your synthesis in a structured format with:
            - Summary (high-level overview)
            - Key Insights (important points from different perspectives)
            - Recommendations (concrete suggestions or answers)
            - Considerations (important caveats or limitations)
            """,
            tools=[get_all_memory]
        )
        
        # Define complex test query
        test_query = """
        In the age of AI, what can a CGI character artist in VFX do to stay relevant? When AI is generating concept art and 3d models, what can I do to support myself and earn money if everything I knew so far is rendered redundant?
        How can I earn a living moving forward? What kind of apps or products tied to AI and CGI could I build that would provide enough value to people that they'd pay money for it during a recession? And also, what kind of ideas of products, projects or apps would not be low hanging fruits to have enough time to build them in a rapidly competitive and advancing tech landscape?
        """
        
        print("\n" + "-"*80)
        print(f"📋 TEST QUERY: {test_query}")
        print("-"*80)
        
        # Run the orchestrator agent first to determine which specialists to use
        print("\n👨‍🚀 Running Orchestrator Agent...")
        try:
            orchestrator_result = await Runner.run(orchestrator, test_query)
            print(f"\nOrchestrator output: {orchestrator_result.final_output}")
        except Exception as e:
            print(f"\n❌ Error with Orchestrator: {str(e)}")
        
        # For demo purposes, let's directly run a sequential chain with all cognitive agents
        print("\n🧠 Running full cognitive swarm sequentially...")
        
        # First, save the query to memory for agents to access
        memory["user_query"] = test_query
        
        # Run each cognitive agent sequentially
        cognitive_agents = [
            ("persona_perspective", persona),
            ("creative_analysis", creativity),
            ("rational_analysis", rationality),
            ("emotional_analysis", emotion),
            ("critical_analysis", critic),
            ("future_projection", future_vision),
            ("curiosity_questions", curiosity)
        ]
        
        for memory_key, agent in cognitive_agents:
            print(f"\n🔄 Running {agent.name}...")
            prompt = f"Analyze this query from your specialist perspective: {test_query}"
            
            try:
                agent_result = await Runner.run(agent, prompt)
                
                # Print the agent's output in a formatted way
                print("\n" + "-"*80)
                print(f"✅ {agent.name.upper()} OUTPUT:".center(80))
                print("-"*80)
                print(f"\n{agent_result.final_output}\n")
                
                # Save directly to memory instead of calling the function tool
                memory[memory_key] = agent_result.final_output
                print(f"Saved to memory as '{memory_key}'")
                
            except Exception as e:
                print(f"\n❌ Error with {agent.name}: {str(e)}")
                continue
        
        # Run the synthesizer to integrate all perspectives
        print("\n🧩 Running Synthesizer to integrate all perspectives...")
        try:
            synthesis_result = await Runner.run(synthesizer, "Synthesize all perspectives on the user query")
            
            print("\n" + "="*80)
            print("🌟 FINAL SYNTHESIS 🌟".center(80))
            print("="*80)
            print(f"\n{synthesis_result.final_output}")
        except Exception as e:
            print(f"\n❌ Error with Synthesizer: {str(e)}")
        
        # Print all memory contents at the end
        print("\n" + "="*80)
        print("📋 FINAL MEMORY CONTENTS:".center(80))
        print("="*80)
        
        for key, value in memory.items():
            print(f"\n- {key}:")
            print(f"{value}")
        
        print("\n" + "="*80)
        print("✅ COGNITIVE SWARM DEMO COMPLETE".center(80))
        print("="*80)
    
    # Run the cognitive swarm demo
    asyncio.run(run_cognitive_swarm_demo())

# Example stream handler for use with run_workflow_with_streaming
async def example_stream_handler(event_type, event_data):
    """Example handler function for streaming events."""
    if event_type == "thinking":
        print(f"🤔 Thinking: {event_data}")
    elif event_type == "tool_calls":
        print(f"🛠️ Tool calls: {event_data}")
    elif event_type == "content":
        print(f"📄 Content: {event_data}")
    elif event_type == "tool_results":
        print(f"🧰 Tool results: {event_data}")