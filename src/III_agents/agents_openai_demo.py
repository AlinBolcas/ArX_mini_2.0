#!/usr/bin/env python
"""
ArX Cognitive Swarm Architecture

A general-purpose cognitive architecture that demonstrates:
- Terminal: Processes user input and delivers output
- Orchestrator: Central coordination of cognitive functions
- Specialized Cognitive Agents: Persona, Critic, Rationality, Creativity, Emotion, Future Vision, Curiosity
- Shared Memory: Allows agents to build on each others' work
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Import core components from the OpenAI Agents SDK
from agents import Agent, Runner, function_tool

# ---- Output Schema ----
class CognitiveOutput(BaseModel):
    summary: str = Field(..., description="Summary of the cognitive process")
    insights: List[str] = Field(..., description="Key insights generated")
    recommendation: str = Field(..., description="Final recommendation or answer")
    reasoning: str = Field(..., description="Reasoning behind the recommendation")
    confidence: int = Field(..., description="Confidence level (1-10)")

# ---- Shared Memory ----
class SharedMemory:
    def __init__(self):
        self.data = {}
    
    def update(self, key: str, value: Any):
        self.data[key] = value
        return f"Memory updated: {key}"
    
    def get(self, key: str) -> Any:
        return self.data.get(key, None)
    
    def get_all(self) -> Dict[str, Any]:
        return self.data

# Initialize shared memory
memory = SharedMemory()

# ---- Tool Functions ----
@function_tool
def save_to_memory(key: str, value: str) -> str:
    """Save information to shared memory for other agents to access."""
    result = memory.update(key, value)
    print(f"💾 MEMORY: Saved '{key}' to shared memory")
    return result

@function_tool
def get_from_memory(key: str) -> Dict[str, Any]:
    """Retrieve information from shared memory."""
    value = memory.get(key)
    print(f"💾 MEMORY: Retrieved '{key}' from shared memory")
    return {"key": key, "value": value}

@function_tool
def get_all_memory() -> Dict[str, Any]:
    """Get all stored information from shared memory."""
    result = memory.get_all()
    print(f"💾 MEMORY: Retrieved all data from shared memory")
    return result

# ---- Create Cognitive Swarm Agents ----
async def create_cognitive_swarm():
    """Create all agents in the cognitive swarm architecture."""
    print("\n🔄 Creating the Cognitive Swarm Architecture...")
    
    # Terminal Agent - Processes user input and delivers output
    terminal_agent = Agent(
        name="Terminal",
        instructions="""You are the Terminal agent.
        Process the user input to make it ready for cognitive analysis.
        Extract key questions, intentions, and context.
        Save the processed input to memory as 'user_query'.
        Keep your response brief and to the point.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Persona Agent - Considers the user perspective
    persona_agent = Agent(
        name="Persona",
        instructions="""You are the Persona agent.
        Analyze the user_query from the perspective of user identity and needs.
        Consider who the user is, their background, and their likely goals.
        Save your analysis to memory as 'persona_perspective'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Creativity Agent - Generates innovative ideas
    creativity_agent = Agent(
        name="Creativity",
        instructions="""You are the Creativity agent.
        Generate innovative approaches to the user_query.
        Think outside the box and avoid conventional thinking.
        Save your creative ideas to memory as 'creative_approaches'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Emotion Agent - Considers emotional aspects
    emotion_agent = Agent(
        name="Emotion",
        instructions="""You are the Emotion agent.
        Analyze the emotional dimensions of the user_query.
        Consider feelings, aspirations, and emotional needs involved.
        Save your emotional analysis to memory as 'emotional_aspects'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Rationality Agent - Applies logical reasoning
    rationality_agent = Agent(
        name="Rationality",
        instructions="""You are the Rationality agent.
        Apply logical reasoning to the user_query.
        Consider facts, data, and logical implications.
        Save your rational analysis to memory as 'rational_analysis'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Critic Agent - Evaluates ideas critically
    critic_agent = Agent(
        name="Critic",
        instructions="""You are the Critic agent.
        Evaluate proposed approaches critically.
        Identify potential flaws, weaknesses, and counterarguments.
        Save your critical analysis to memory as 'critical_evaluation'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Future Vision Agent - Considers long-term implications
    future_vision_agent = Agent(
        name="Future Vision",
        instructions="""You are the Future Vision agent.
        Consider long-term implications and future scenarios.
        Project potential outcomes and future developments.
        Save your future analysis to memory as 'future_projection'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Curiosity Agent - Asks important questions
    curiosity_agent = Agent(
        name="Curiosity",
        instructions="""You are the Curiosity agent.
        Identify important questions that should be asked.
        Consider gaps in knowledge and areas that need exploration.
        Save your questions to memory as 'curiosity_questions'.
        Keep your response concise.
        """,
        tools=[save_to_memory, get_from_memory],
        model="gpt-4o-mini"
    )
    
    # Output Agent - Synthesizes final response
    output_agent = Agent(
        name="Output",
        instructions="""You are the Output agent.
        Synthesize all memory data into a cohesive response.
        Format your output according to the CognitiveOutput schema.
        Balance all perspectives from the cognitive agents.
        """,
        tools=[get_all_memory],
        output_type=CognitiveOutput,
        model="gpt-4o"
    )
    
    # Orchestrator agent manages the workflow
    orchestrator_agent = Agent(
        name="Orchestrator",
        instructions="""You are the Orchestrator agent that coordinates the cognitive swarm.
        
        You should strategically activate the appropriate cognitive agents based on the query.
        You have these agents available:
        - Terminal: Processes initial user input
        - Persona: Considers user perspective
        - Creativity: Generates innovative approaches
        - Emotion: Analyzes emotional dimensions
        - Rationality: Applies logical reasoning
        - Critic: Evaluates ideas critically
        - Future Vision: Considers long-term implications
        - Curiosity: Asks important questions
        - Output: Synthesizes final response
        
        Not all agents need to be activated for every query. Choose wisely based on the nature of the input.
        Keep your coordination messages extremely brief.
        Don't do any of the tasks yourself - just coordinate the agent activations.
        """,
        handoffs=[
            terminal_agent,
            persona_agent, 
            creativity_agent,
            emotion_agent,
            rationality_agent,
            critic_agent,
            future_vision_agent,
            curiosity_agent,
            output_agent
        ],
        model="gpt-4o-mini"
    )
    
    print("✓ Created all agents")
    
    return {
        "orchestrator": orchestrator_agent,
        "terminal": terminal_agent,
        "persona": persona_agent,
        "creativity": creativity_agent,
        "emotion": emotion_agent,
        "rationality": rationality_agent,
        "critic": critic_agent,
        "future_vision": future_vision_agent,
        "curiosity": curiosity_agent,
        "output": output_agent
    }

def display_cognitive_output(output: CognitiveOutput):
    """Display the final cognitive output with formatting."""
    print("\n" + "="*80)
    print(f"✨ COGNITIVE SWARM OUTPUT ✨".center(80))
    print("="*80)
    
    print(f"\n📝 SUMMARY")
    print(f"{output.summary}")
    
    print(f"\n💡 INSIGHTS")
    for insight in output.insights:
        print(f"  • {insight}")
    
    print(f"\n🎯 RECOMMENDATION")
    print(f"{output.recommendation}")
    
    print(f"\n🧠 REASONING")
    print(f"{output.reasoning}")
    
    print(f"\n⚖️ CONFIDENCE: {output.confidence}/10")
    
    print("\n" + "="*80)

def print_agent_output(agent_name, output):
    """Print agent output with clear formatting."""
    print("\n" + "="*80)
    print(f"🤖 {agent_name} OUTPUT".center(80))
    print("="*80)
    print(f"\n{output}\n")

async def run_cognitive_swarm(user_input: str):
    """Run the full cognitive swarm."""
    try:
        # Create cognitive swarm agents
        swarm = await create_cognitive_swarm()
        
        print("\n" + "="*80)
        print(f"🔄 COGNITIVE SWARM: STARTING PROCESS".center(80))
        print("="*80)
        
        print("\n" + "-"*80)
        print(f"📋 USER INPUT:")
        print("-"*80)
        print(f"{user_input.strip()}\n")
        
        start_time = time.time()
        
        # Terminal processes the input
        print("\n" + "-"*80)
        print("🔍 STEP 1: TERMINAL PROCESSING")
        print("-"*80)
        terminal_result = await Runner.run(swarm["terminal"], user_input)
        print_agent_output("TERMINAL", terminal_result.final_output)
        
        # Orchestrator decides which agents to activate
        # Since true handoffs might be complex, we'll simulate it for demonstration
        
        # Activate Persona Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: PERSONA")
        print("-"*80)
        persona_result = await Runner.run(swarm["persona"], "Get the user_query from memory and analyze it from a user perspective. Save your analysis as 'persona_perspective'.")
        print_agent_output("PERSONA", persona_result.final_output)
        
        # Activate Creativity Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: CREATIVITY")
        print("-"*80)
        creativity_result = await Runner.run(swarm["creativity"], "Get the user_query from memory and generate creative approaches. Save your ideas as 'creative_approaches'.")
        print_agent_output("CREATIVITY", creativity_result.final_output)
        
        # Activate Emotion Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: EMOTION")
        print("-"*80)
        emotion_result = await Runner.run(swarm["emotion"], "Get the user_query from memory and analyze its emotional dimensions. Save your analysis as 'emotional_aspects'.")
        print_agent_output("EMOTION", emotion_result.final_output)
        
        # Activate Rationality Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: RATIONALITY")
        print("-"*80)
        rationality_result = await Runner.run(swarm["rationality"], "Get the user_query from memory and apply logical reasoning. Save your analysis as 'rational_analysis'.")
        print_agent_output("RATIONALITY", rationality_result.final_output)
        
        # Activate Critic Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: CRITIC")
        print("-"*80)
        critic_result = await Runner.run(swarm["critic"], "Get the user_query from memory and critically evaluate it. Also review any available memory data. Save your evaluation as 'critical_evaluation'.")
        print_agent_output("CRITIC", critic_result.final_output)
        
        # Activate Future Vision Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: FUTURE VISION")
        print("-"*80)
        future_result = await Runner.run(swarm["future_vision"], "Get the user_query from memory and consider long-term implications. Save your analysis as 'future_projection'.")
        print_agent_output("FUTURE VISION", future_result.final_output)
        
        # Activate Curiosity Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: CURIOSITY")
        print("-"*80)
        curiosity_result = await Runner.run(swarm["curiosity"], "Get the user_query from memory and identify important questions that should be asked. Save your questions as 'curiosity_questions'.")
        print_agent_output("CURIOSITY", curiosity_result.final_output)
        
        # Final Output synthesis
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: OUTPUT")
        print("-"*80)
        output_result = await Runner.run(swarm["output"], "Synthesize all data in memory into a complete CognitiveOutput.")
        
        # If the result contains a final output, display it
        if hasattr(output_result, 'final_output_as'):
            try:
                final_output = output_result.final_output_as(CognitiveOutput)
                display_cognitive_output(final_output)
            except Exception as e:
                print(f"\n❌ ERROR: Could not parse final output as CognitiveOutput: {str(e)}")
                print(f"\nRaw output: {output_result.final_output}")
        else:
            print("\n❌ ERROR: No final output was produced.")
            if hasattr(output_result, 'final_output'):
                print(f"\nRaw output: {output_result.final_output}")
        
        end_time = time.time()
        print(f"\n⏱️ Process completed in {end_time - start_time:.2f} seconds")
        
        # Also print the memory to debug
        print("\n" + "="*80)
        print("📋 FINAL SHARED MEMORY CONTENTS:".center(80))
        print("="*80)
        all_memory = memory.get_all()
        for key, value in all_memory.items():
            print(f"\n- {key}:")
            print(f"{value[:200]}..." if len(str(value)) > 200 else f"{value}")
        
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {str(e)}")
        traceback.print_exc()

async def main():
    """Run the cognitive swarm with a sample input."""
    user_input = """
    What are the potential implications of AI systems becoming increasingly autonomous in decision-making processes?
    """
    await run_cognitive_swarm(user_input)

if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        asyncio.run(main())
    else:
        print("Please set the OPENAI_API_KEY environment variable.") 