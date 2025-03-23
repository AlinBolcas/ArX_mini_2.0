# OpenAI Agents API Wrapper

A comprehensive wrapper for the OpenAI Agents SDK, making it easy to create, configure, and orchestrate intelligent agents.

## Features

- 🤖 **Simple Agent Creation** - Create agents with a single method call
- 🔧 **Tool Integration** - Easily create and configure tools for agents to use
- 🚦 **Guardrails** - Implement input and output guardrails with a fluent API
- 🔄 **Handoffs** - Orchestrate multiple agents with automatic handoffs
- 📊 **Structured Output** - Get typed, structured outputs using Pydantic models
- 🧵 **Streaming Support** - Stream responses as they are generated
- 📈 **Tracing** - Built-in tracing for agent runs viewable in the OpenAI Dashboard
- 🧠 **Context Management** - Pass and maintain context between agent runs

## Installation

```bash
pip install openai-agents
```

## Basic Usage

```python
from src.III_agents.openai_agents import OpenAIAgentsAPI
from pydantic import BaseModel
from typing import List
import asyncio

# Define structured output
class MathAnswer(BaseModel):
    equation: str
    steps: List[str]
    answer: float
    explanation: str

async def main():
    # Initialize API
    api = OpenAIAgentsAPI()
    
    # Create a tool
    def calculate_square_root(number: float) -> float:
        """Calculate the square root of a number."""
        import math
        return math.sqrt(number)
    
    square_root_tool = api.create_tool(calculate_square_root)
    
    # Create an agent with the tool
    math_agent = api.create_agent(
        name="Math Tutor",
        instructions="You help with math problems, showing your work step by step.",
        tools=[square_root_tool],
        output_type=MathAnswer
    )
    
    # Run the agent and get a response
    result = await api.run_agent(
        math_agent,
        "What is the square root of 144? Show your work."
    )
    
    # Get structured output
    math_answer = api.get_result_as_model_sync(result, MathAnswer)
    print(f"Equation: {math_answer.equation}")
    print(f"Answer: {math_answer.answer}")
    print(f"Explanation: {math_answer.explanation}")

# Run the example
asyncio.run(main())
```

## Multiple Agents with Handoffs

```python
from src.III_agents.openai_agents import OpenAIAgentsAPI

# Initialize API
api = OpenAIAgentsAPI()

# Create specialist agents
math_agent = api.create_agent(
    name="Math Tutor",
    instructions="You help with math problems.",
    handoff_description="Specialist for math questions"
)

history_agent = api.create_agent(
    name="History Tutor",
    instructions="You provide assistance with historical questions.",
    handoff_description="Specialist for history questions"
)

# Create a triage agent with handoffs
triage_agent = api.create_agent(
    name="Triage Agent",
    instructions="You determine which specialist agent is most appropriate for the question.",
    handoffs=[math_agent, history_agent]
)

# Run the orchestration
result = api.run_agent_sync(
    triage_agent,
    "Who was the first president of the United States?"
)

print(result.final_output)
```

## Implementing Guardrails

```python
from src.III_agents.openai_agents import OpenAIAgentsAPI
from pydantic import BaseModel

# Define guardrail output model
class ContentFilter(BaseModel):
    is_allowed: bool
    reasoning: str

# Initialize API
api = OpenAIAgentsAPI()

# Create a guardrail agent and the guardrail
guardrail_builder = api.create_guardrail_builder(ContentFilter)
guardrail_agent = api.create_agent(
    name="Content Filter",
    instructions="Determine if the input is appropriate. Return is_allowed=False for any harmful content.",
    output_type=ContentFilter
)
input_guardrail = guardrail_builder.with_agent(guardrail_agent).build_input_guardrail()

# Create an agent with the guardrail
agent = api.create_agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    input_guardrails=[input_guardrail]
)

# The guardrail will block inappropriate requests
result = api.run_agent_sync(agent, "Tell me about the history of science.")
print(result.final_output)
```

## Streaming Results

```python
import asyncio
from src.III_agents.openai_agents import OpenAIAgentsAPI

async def main():
    api = OpenAIAgentsAPI()
    
    agent = api.create_agent(
        name="Assistant",
        instructions="You provide detailed explanations on topics."
    )
    
    # Callback for streaming events
    def handle_stream_event(event):
        if hasattr(event, 'delta'):
            print(event.delta, end="", flush=True)
    
    # Run with streaming
    await api.run_agent(
        agent,
        "Explain quantum computing in simple terms.",
        stream=True,
        callback=handle_stream_event
    )

asyncio.run(main())
```

## Tracing

The OpenAI Agents API wrapper automatically configures tracing. You can view traces of your agent runs in the [OpenAI Dashboard](https://platform.openai.com/tracing).

## Advanced Features

For more advanced features and configurations, refer to the docstrings in the code or the [official OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/). 