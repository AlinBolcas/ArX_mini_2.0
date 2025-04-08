#!/usr/bin/env python
"""
ArX Cognitive Swarm Architecture - Ollama Version

A demonstration of using a cognitive swarm with Ollama local models
"""

import os
import asyncio
import time
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Add parent directory to path to access ollama_API
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.I_integrations.ollama_API import OllamaWrapper

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

def print_agent_output(agent_name, output):
    """Print agent output with clear formatting."""
    print("\n" + "="*80)
    print(f"🤖 {agent_name} OUTPUT".center(80))
    print("="*80)
    print(f"\n{output}\n")

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

async def run_ollama_cognitive_swarm(user_input: str, model: str = "llama3:latest"):
    """Run a simplified cognitive swarm using Ollama."""
    try:
        print("\n🔄 Creating the Cognitive Swarm Architecture with Ollama...")
        
        # Create an OllamaWrapper instance
        ollama = OllamaWrapper(model=model)
        
        print("\n" + "="*80)
        print(f"🔄 OLLAMA COGNITIVE SWARM: STARTING PROCESS".center(80))
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
        
        terminal_result = ollama.chat_completion(
            user_prompt=user_input,
            system_prompt="""You are the Terminal agent.
            Process the user input to make it ready for cognitive analysis.
            Extract key questions, intentions, and context.
            Keep your response brief and to the point."""
        )
        print_agent_output("TERMINAL", terminal_result)
        memory.update('user_query', terminal_result)
        
        # Activate Persona Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: PERSONA")
        print("-"*80)
        
        user_query = memory.get('user_query')
        persona_result = ollama.chat_completion(
            user_prompt=f"Analyze this query from a user perspective: {user_query}",
            system_prompt="""You are the Persona agent.
            Analyze the user query from the perspective of user identity and needs.
            Consider who the user is, their background, and their likely goals.
            Keep your response concise."""
        )
        print_agent_output("PERSONA", persona_result)
        memory.update('persona_perspective', persona_result)
        
        # Activate Creativity Agent
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: CREATIVITY")
        print("-"*80)
        
        creativity_result = ollama.chat_completion(
            user_prompt=f"Generate creative approaches for this query: {user_query}",
            system_prompt="""You are the Creativity agent.
            Generate innovative approaches to the user query.
            Think outside the box and avoid conventional thinking.
            Keep your response concise."""
        )
        print_agent_output("CREATIVITY", creativity_result)
        memory.update('creative_approaches', creativity_result)
        
        # Final Output synthesis
        print("\n" + "-"*80)
        print("🔍 ACTIVATING: OUTPUT")
        print("-"*80)
        
        # Prepare memory data for the output agent
        memory_data = memory.get_all()
        memory_context = "\n".join([f"{key}: {value}" for key, value in memory_data.items()])
        
        output_object = ollama.structured_output(
            user_prompt=f"Synthesize this information into a coherent response:\n{memory_context}",
            system_prompt="""You are the Output agent.
            Synthesize all memory data into a cohesive response.
            Your output should include a summary, insights, recommendation, reasoning, and confidence level (1-10).
            Balance all perspectives from the cognitive agents.""",
            output_class=CognitiveOutput
        )
        
        # Display the final output
        if isinstance(output_object, CognitiveOutput):
            display_cognitive_output(output_object)
        else:
            print(f"\n❌ ERROR: Unexpected output format: {type(output_object)}")
            print(f"\nRaw output: {output_object}")
        
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
    """Run the Ollama-powered cognitive swarm with a sample input."""
    # First, verify Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("Error: Ollama service is not running. Please start it with 'ollama serve'")
            return
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Please make sure Ollama is installed and running with 'ollama serve'")
        return
    
    # Show available models
    print("Checking available Ollama models...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = [model["name"] for model in response.json()["models"]]
        print(f"Available models: {', '.join(models)}")
        
        # Ask user to select a model
        print("\nAvailable models:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")
        
        model_choice = input("\nSelect model number (or press Enter for default llama3): ")
        if model_choice.strip() and model_choice.isdigit() and 1 <= int(model_choice) <= len(models):
            selected_model = models[int(model_choice)-1]
        else:
            selected_model = "llama3:latest"
        
        print(f"Using model: {selected_model}")
    except Exception as e:
        print(f"Could not get model list: {e}")
        selected_model = "llama3:latest"
    
    user_input = input("\nEnter your query (or press Enter for default): ")
    if not user_input.strip():
        user_input = "What are the potential benefits of using local AI models instead of cloud-based ones?"
    
    await run_ollama_cognitive_swarm(user_input, selected_model)

if __name__ == "__main__":
    asyncio.run(main())