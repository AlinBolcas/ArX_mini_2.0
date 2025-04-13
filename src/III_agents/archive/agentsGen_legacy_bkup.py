import os
import sys
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
# Navigate up to the project root (2 levels up from src/III_agents/)
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import our API wrappers
from src.II_textGen.textGen import TextGen

# ---- Output Schema ----
class CognitiveOutput(BaseModel):
    summary: str = Field(..., description="Summary of the cognitive process")
    insights: List[str] = Field(..., description="Key insights generated")
    recommendation: str = Field(..., description="Final recommendation or answer")
    reasoning: str = Field(..., description="Reasoning behind the recommendation")
    confidence: int = Field(..., description="Confidence level (1-10)")


class AgentGen(TextGen):
    """
    AgentGen: A lightweight yet powerful agent framework built on TextGen.
    """

    def __init__(self, api_keys_path: str = None, short_term_limit: int = 8000,
                 chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(api_keys_path, short_term_limit, chunk_size, chunk_overlap)

    ### TOOL SELECTION ###
    def select_best_tools(self, user_prompt: str, top_k: int = 3) -> list:
        """
        Dynamically selects the most relevant tools based on the user prompt.
        """
        available_tools = self.get_available_tools()
        print("Available tools:\n" + ", ".join(available_tools) if available_tools else "No tools available.")
        
        if not available_tools:
            print("⚠️ No tools available for selection.")
            return []

        tool_selection_prompt = (
            f"Given the following tools:\n{', '.join(available_tools)}\n"
            f"Which {top_k} tools would be most useful for this task: {user_prompt}?"
            "Return a structured JSON list of tool names."
        )

        selected_tools = self.structured_output(
            user_prompt=tool_selection_prompt,
            system_prompt="Analyze the given tools and select the most relevant ones for the task.",
        )

        # Ensure the response is a list and print selected tools
        selected_tools = selected_tools if isinstance(selected_tools, list) else []
        
        print(f"🛠️ Selected Tools for '{user_prompt}': {selected_tools}")

        return selected_tools
                            
    def agent_loop(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                  max_tokens: int = None, contex: str = None, system_contex: str = None, 
                  max_depth: int = 5, verbose: bool = True) -> str:
        """
        Iteratively executes tools until an optimal response is achieved.
        """
        response = ""
        for i in range(max_depth):
            tool_names = self.select_best_tools(user_prompt)
            response = self.chat_completion(
                user_prompt=f"🔄 REFINE ITERATIVELY USING TOOLS FOR:\n{user_prompt}\n\n"
                            f"💡 CURRENT OUTPUT:\n{response}\n"
                            "Indicate 'FINAL RESPONSE:' when the answer is complete.",
                system_prompt=system_prompt or "Use available tools when required to refine the response. "
                                               "Stop when the solution is optimal and explicitly state 'FINAL RESPONSE:'.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex, 
                tool_names=tool_names
            )

            if verbose:
                print(f"🔄 Base Loop Iteration {i+1} Response:\n{response}\n")
                
            if "FINAL RESPONSE:" in response:
                break
        return response

    def REACT_loop(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                   max_tokens: int = None, contex: str = None, system_contex: str = None, 
                   max_depth: int = 5, verbose: bool = True) -> str:
        """
        Uses cascading Observation → Reflection → Action steps iteratively.
        """
        response = ""
        for i in range(max_depth):
            tool_names = self.select_best_tools(user_prompt)

            # Step 1: OBSERVATION
            observation = self.chat_completion(
                user_prompt=f"🔍 OBSERVE the situation given:\n{user_prompt}\n"
                            f"💡 CURRENT OUTPUT:\n{response}\n"
                            "Describe key details, insights, and any missing elements.",
                system_prompt="Extract relevant observations and key insights from the given context.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            # Step 2: REFLECTION
            reflection = self.chat_completion(
                user_prompt=f"🤔 REFLECT upon:\n{user_prompt}\n\n"
                            f"🔍 OBSERVATION:\n{observation}\n"
                            "Identify patterns, inconsistencies, and potential improvements.",
                system_prompt="Analyze the observations, identify missing aspects, and suggest improvements.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            # Step 3: ACTION
            response = self.chat_completion(
                user_prompt=f"🚀 ACT based on:\n{user_prompt}\n\n"
                            f"🔍 OBSERVATION:\n{observation}\n"
                            f"🤔 REFLECTION:\n{reflection}\n"
                            "Formulate an optimized response, taking all insights into account. "
                            "Indicate 'FINAL RESPONSE:' when the answer is fully optimized.",
                system_prompt="Synthesize observations and reflections into a final, actionable response.",
                temperature=temperature, max_tokens=max_tokens, contex=contex, system_contex=system_contex,
                tool_names=tool_names
            )

            if verbose:
                print(f"🔄 ReAct Loop Iteration {i+1} Response:\n{response}\n")

            if "FINAL RESPONSE:" in response:
                break
        return response

    def triage_agent(self, user_prompt: str, system_prompt: str = None, temperature: float = None,
                      max_tokens: int = None, contex: str = None, system_contex: str = None, 
                      max_depth: int = 5, verbose: bool = True) -> str:
        """
        Handoff to another agent.
        """
        pass
        #     Creates a Triage Agent designed to hand off tasks to other agents.

        #     Args:
        #         name (str): Name for the triage agent.
        #         instructions (str): Instructions for how to decide which agent to hand off to. Should reference the purpose of handoff_agents.
        #         handoff_agents (List[Agent]): A list of Agent objects that this agent can delegate tasks to. Ensure these agents have `handoff_description` set.
        #         model (Optional[str]): Specific model for the triage agent.
        #         input_guardrails (Optional[List[InputGuardrail]]): Guardrails to run before triage logic.
        #         **kwargs: Additional keyword arguments for the Agent constructor.

        #     Returns:
        #         Agent: An initialized Triage Agent object.
        #     """
        #     # More defensive printing that works even with dummy classes
        #     try:
        #         agent_names = [getattr(a, 'name', f"Agent_{i}") for i, a in enumerate(handoff_agents)]
        #         print(f"🧬 Creating Triage Agent: {name} (Handoffs: {agent_names})")
        #     except Exception as e:
        #         print(f"🧬 Creating Triage Agent: {name} (Handoffs: {len(handoff_agents)} agents)")
            
        #     # Validate that handoff agents have descriptions
        #     for agent in handoff_agents:
        #         if not getattr(agent, 'handoff_description', None):
        #             print(f"⚠️ WARNING: Handoff agent '{agent.name}' lacks a 'handoff_description'. Triage may be less effective.")
            
        #     return Agent(
        #         name=name,
        #         instructions=instructions,
        #         model=model or self.default_model,
        #         handoffs=handoff_agents,
        #         input_guardrails=input_guardrails or [],
        #         **kwargs
        #     )

class Worker:
    def __init__(self, agent: AgentGen):
        self.agent = agent

    def work():
        # should run in a new thread/concurent future
        pass

if __name__ == "__main__":
    print("=== AgentGen Advanced Test Suite ===")
    
    # Initialize AgentGen with default settings.
    ag = AgentGen()
