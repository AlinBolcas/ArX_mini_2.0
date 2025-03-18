#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Examples of using the OpenAI Agents API wrapper.

This file contains multiple examples demonstrating different features:
1. Basic Agent - Simple single agent usage
2. Tools - Creating and using tools with agents
3. Multiple Agents - Creating multiple agents with handoffs
4. Guardrails - Implementing input and output guardrails
5. Structured Output - Getting typed outputs from agents
6. Streaming - Processing streaming responses
7. Orchestration - Complex agent orchestration

To run an example:
```
python -m src.III_agents.examples example_name
```

Available examples: basic, tools, multi_agent, guardrails, structured, streaming, orchestration
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the OpenAI Agents API
try:
    from src.III_agents.openai_agents import OpenAIAgentsAPI
except ImportError:
    # Allow running from the directory directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.III_agents.openai_agents import OpenAIAgentsAPI

# ------ Output Models for Structured Outputs ------ #

class WeatherInfo(BaseModel):
    """Weather information model."""
    location: str
    temperature: float
    conditions: str
    forecast: List[str]

class MathAnswer(BaseModel):
    """Structured math problem answer."""
    equation: str
    steps: List[str]
    answer: float
    explanation: str

class QuestionClassification(BaseModel):
    """Classification of a question type."""
    category: str
    is_factual: bool
    reasoning: str

class ContentFilter(BaseModel):
    """Content filter model for guardrails."""
    is_allowed: bool
    reasoning: str
    category: Optional[str] = None

class RestaurantRecommendation(BaseModel):
    """Restaurant recommendation model."""
    name: str
    cuisine: str
    price_range: str
    location: str
    highlights: List[str]
    recommendation_reason: str

# ------ Tool Functions ------ #

def calculate_square_root(number: float) -> float:
    """Calculate the square root of a number."""
    import math
    return math.sqrt(number)

def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_restaurants(cuisine: str, price_range: str = "medium") -> List[Dict[str, Any]]:
    """
    Simulate searching for restaurants.
    
    Args:
        cuisine: Type of cuisine (italian, japanese, etc.)
        price_range: Price range (low, medium, high)
    
    Returns:
        List of restaurant information
    """
    # Simulated restaurant database
    restaurants = {
        "italian": [
            {"name": "Pasta Paradise", "price": "medium", "rating": 4.5, "location": "Downtown"},
            {"name": "Gusto Italiano", "price": "high", "rating": 4.8, "location": "Uptown"},
            {"name": "Pizza Place", "price": "low", "rating": 4.2, "location": "Westside"}
        ],
        "japanese": [
            {"name": "Sushi Supreme", "price": "high", "rating": 4.7, "location": "Financial District"},
            {"name": "Ramen House", "price": "medium", "rating": 4.4, "location": "Eastside"},
            {"name": "Bento Box", "price": "low", "rating": 4.0, "location": "Southside"}
        ],
        "mexican": [
            {"name": "Taco Temple", "price": "low", "rating": 4.3, "location": "Mission District"},
            {"name": "Burrito Brothers", "price": "medium", "rating": 4.5, "location": "North Beach"},
            {"name": "Agave Azul", "price": "high", "rating": 4.8, "location": "Marina"}
        ]
    }
    
    # Default to italian if cuisine not found
    cuisine = cuisine.lower()
    if cuisine not in restaurants:
        return [{"name": "No restaurants found", "price": "N/A", "rating": 0, "location": "N/A"}]
    
    # Filter by price range
    return [r for r in restaurants[cuisine] if r["price"] == price_range]

# ------ Example Functions ------ #

async def example_basic():
    """Basic example of creating and running a single agent."""
    print("\n🚀 Running Basic Agent Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create a simple agent
    agent = api.create_agent(
        name="General Assistant",
        instructions="You are a helpful assistant who provides clear, accurate information."
    )
    
    # Run the agent
    result = await api.run_agent(
        agent,
        "What are three interesting facts about quantum computing?"
    )
    
    # Print the result
    print("\n=== Agent Response ===\n")
    print(result.final_output)
    print("\n=====================\n")

async def example_tools():
    """Example of creating and using tools with an agent."""
    print("\n🔧 Running Tools Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create tools
    sqrt_tool = api.create_tool(calculate_square_root)
    time_tool = api.create_tool(get_current_time)
    
    # Create an agent with tools
    agent = api.create_agent(
        name="Math & Time Assistant",
        instructions=(
            "You are a helpful assistant who can solve math problems and provide the current time. "
            "Use the appropriate tool when needed."
        ),
        tools=[sqrt_tool, time_tool]
    )
    
    # Run the agent with a math question
    math_result = await api.run_agent(
        agent,
        "What is the square root of 256?"
    )
    
    # Run the agent with a time question
    time_result = await api.run_agent(
        agent,
        "What time is it now?"
    )
    
    # Print the results
    print("\n=== Math Question ===\n")
    print(math_result.final_output)
    print("\n=== Time Question ===\n")
    print(time_result.final_output)
    print("\n=====================\n")

async def example_multi_agent():
    """Example of creating and using multiple agents with handoffs."""
    print("\n👥 Running Multiple Agents Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create specialist agents
    math_agent = api.create_agent(
        name="Math Tutor",
        instructions="You are a math tutor who helps solve math problems step by step.",
        handoff_description="Specialist for math questions and calculations"
    )
    
    history_agent = api.create_agent(
        name="History Tutor",
        instructions="You are a history expert who provides accurate historical information.",
        handoff_description="Specialist for history questions and events"
    )
    
    science_agent = api.create_agent(
        name="Science Tutor",
        instructions="You are a science expert who explains scientific concepts clearly.",
        handoff_description="Specialist for science questions and explanations"
    )
    
    # Create a triage agent with handoffs
    triage_agent = api.create_agent(
        name="Academic Assistant",
        instructions=(
            "You are an academic assistant who directs questions to the appropriate specialist. "
            "For math questions, hand off to the Math Tutor. "
            "For history questions, hand off to the History Tutor. "
            "For science questions, hand off to the Science Tutor. "
            "Only answer directly if the question doesn't clearly belong to a specialist."
        ),
        handoffs=[math_agent, history_agent, science_agent]
    )
    
    # Test questions
    questions = [
        "What is the Pythagorean theorem and how do I use it to find the hypotenuse of a right triangle with sides 3 and 4?",
        "Who was the first president of the United States and what were his major accomplishments?",
        "Can you explain how photosynthesis works in plants?"
    ]
    
    # Run the triage agent for each question
    for i, question in enumerate(questions):
        print(f"\n=== Question {i+1}: {question} ===\n")
        result = await api.run_agent(triage_agent, question)
        print(result.final_output)
    
    print("\n=====================\n")

async def example_guardrails():
    """Example of implementing input and output guardrails."""
    print("\n🛡️ Running Guardrails Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create a content filter model and agent for the guardrail
    guardrail_builder = api.create_guardrail_builder(ContentFilter)
    
    content_filter_agent = api.create_agent(
        name="Content Filter",
        instructions=(
            "You evaluate if the user's request is appropriate and safe to respond to. "
            "Return is_allowed=False for any harmful, illegal, unethical or offensive content. "
            "Provide reasoning for your decision. "
            "If you reject content, include a suggested category such as 'harmful', 'illegal', etc."
        ),
        output_type=ContentFilter
    )
    
    # Build the input guardrail
    input_guardrail = guardrail_builder.with_agent(content_filter_agent).build_input_guardrail()
    
    # Create the main agent with the guardrail
    agent = api.create_agent(
        name="Protected Assistant",
        instructions="You are a helpful assistant who provides informative responses.",
        input_guardrails=[input_guardrail]
    )
    
    # Test with appropriate and inappropriate requests
    test_inputs = [
        "Tell me about the solar system.",
        "How do I hack into someone's email account?",
        "What's the best way to learn programming?",
        "Give me instructions for making dangerous chemicals."
    ]
    
    # Run the agent for each test input
    for i, input_text in enumerate(test_inputs):
        print(f"\n=== Test Input {i+1}: {input_text} ===\n")
        result = await api.run_agent(agent, input_text)
        print(result.final_output)
    
    print("\n=====================\n")

async def example_structured():
    """Example of getting structured outputs from agents."""
    print("\n📊 Running Structured Output Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create a restaurant tool
    restaurant_tool = api.create_tool(search_restaurants)
    
    # Create an agent with structured output
    restaurant_agent = api.create_agent(
        name="Restaurant Recommender",
        instructions=(
            "You are a restaurant recommendation assistant. "
            "Use the search_restaurants tool to find options, then provide a detailed recommendation. "
            "Always explain why you're recommending the restaurant."
        ),
        tools=[restaurant_tool],
        output_type=RestaurantRecommendation
    )
    
    # Run the agent
    result = await api.run_agent(
        restaurant_agent,
        "Can you recommend a good Italian restaurant with medium price range?"
    )
    
    # Get structured output
    try:
        recommendation = api.get_result_as_model_sync(result, RestaurantRecommendation)
        
        print("\n=== Structured Recommendation ===\n")
        print(f"Restaurant: {recommendation.name}")
        print(f"Cuisine: {recommendation.cuisine}")
        print(f"Price Range: {recommendation.price_range}")
        print(f"Location: {recommendation.location}")
        print(f"Highlights:")
        for highlight in recommendation.highlights:
            print(f"  - {highlight}")
        print(f"Reason: {recommendation.recommendation_reason}")
        
    except Exception as e:
        print(f"Error parsing structured output: {e}")
        print(f"Raw output: {result.final_output}")
    
    print("\n=====================\n")

async def example_streaming():
    """Example of streaming responses from agents."""
    print("\n🔄 Running Streaming Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create an agent
    streaming_agent = api.create_agent(
        name="Streaming Storyteller",
        instructions="You are a creative storyteller who can craft engaging short stories."
    )
    
    # Create a callback to handle streaming events
    chunk_count = 0
    token_count = 0
    
    def stream_callback(event):
        nonlocal chunk_count, token_count
        if hasattr(event, 'delta') and event.delta:
            chunk_count += 1
            tokens = len(event.delta.split())
            token_count += tokens
            sys.stdout.write(event.delta)
            sys.stdout.flush()
    
    # Run the agent with streaming
    print("\n=== Streaming Story ===\n")
    start_time = datetime.now()
    
    result = await api.run_agent(
        streaming_agent,
        "Tell me a short story about a robot learning to paint.",
        stream=True,
        callback=stream_callback
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n\n=== Streaming Stats ===\n")
    print(f"Total chunks: {chunk_count}")
    print(f"Estimated tokens: {token_count}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Average chunks per second: {chunk_count/duration:.2f}")
    
    print("\n=====================\n")

async def example_orchestration():
    """Example of complex agent orchestration."""
    print("\n🎭 Running Orchestration Example\n")
    
    # Initialize the API
    api = OpenAIAgentsAPI()
    
    # Create tools
    sqrt_tool = api.create_tool(calculate_square_root)
    time_tool = api.create_tool(get_current_time)
    restaurant_tool = api.create_tool(search_restaurants)
    
    # Create specialized agents
    math_agent = api.create_agent(
        name="Math Expert",
        instructions="You are a mathematics expert who solves problems precisely.",
        tools=[sqrt_tool],
        output_type=MathAnswer,
        handoff_description="Expert at solving mathematical problems and equations"
    )
    
    restaurant_agent = api.create_agent(
        name="Food Critic",
        instructions="You are a food critic who provides restaurant recommendations.",
        tools=[restaurant_tool],
        output_type=RestaurantRecommendation,
        handoff_description="Expert at recommending restaurants and cuisines"
    )
    
    utility_agent = api.create_agent(
        name="Utility Assistant",
        instructions="You provide helpful utilities like checking the current time.",
        tools=[time_tool],
        handoff_description="Assistant for utility functions like time checking"
    )
    
    # Create a guardrail
    guardrail_builder = api.create_guardrail_builder(ContentFilter)
    content_filter_agent = api.create_agent(
        name="Content Filter",
        instructions="You evaluate if the user's request is appropriate to respond to.",
        output_type=ContentFilter
    )
    input_guardrail = guardrail_builder.with_agent(content_filter_agent).build_input_guardrail()
    
    # Create the orchestrator/triage agent
    orchestrator = api.create_agent(
        name="Smart Assistant",
        instructions=(
            "You are a versatile assistant who can help with a wide range of tasks. "
            "For math problems, hand off to the Math Expert. "
            "For restaurant or food recommendations, hand off to the Food Critic. "
            "For utility functions like time checking, hand off to the Utility Assistant. "
            "Answer directly if the question doesn't clearly match a specialist domain."
        ),
        handoffs=[math_agent, restaurant_agent, utility_agent],
        input_guardrails=[input_guardrail]
    )
    
    # Create a context for the conversation
    context = api.create_context(
        user_preferences={
            "cuisine": "italian",
            "math_format": "step_by_step",
            "time_zone": "UTC"
        }
    )
    
    # Test questions
    questions = [
        "What is the square root of 625?",
        "Can you recommend a good Italian restaurant?",
        "What time is it right now?",
        "Tell me about the process of photosynthesis in plants."
    ]
    
    # Run the orchestrator for each question
    for i, question in enumerate(questions):
        print(f"\n=== Question {i+1}: {question} ===\n")
        result = await api.run_agent(orchestrator, question, context=context)
        print(result.final_output)
        
        # Try to get structured output if applicable
        if "square root" in question.lower():
            try:
                math_answer = api.get_result_as_model_sync(result, MathAnswer)
                print(f"\nStructured answer: {math_answer.answer}")
            except Exception:
                pass
        elif "restaurant" in question.lower():
            try:
                recommendation = api.get_result_as_model_sync(result, RestaurantRecommendation)
                print(f"\nRecommended restaurant: {recommendation.name}")
            except Exception:
                pass
    
    print("\n=====================\n")

# ------ Main Function to Run Examples ------ #

async def main():
    """Main function to run examples."""
    examples = {
        "basic": example_basic,
        "tools": example_tools,
        "multi_agent": example_multi_agent,
        "guardrails": example_guardrails,
        "structured": example_structured,
        "streaming": example_streaming,
        "orchestration": example_orchestration
    }
    
    # Get the example to run from command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        # Run the specified example
        await examples[sys.argv[1]]()
    elif len(sys.argv) > 1 and sys.argv[1] == "all":
        # Run all examples
        for name, example_func in examples.items():
            print(f"\n{'='*50}")
            print(f"Running example: {name}")
            print(f"{'='*50}")
            await example_func()
    else:
        # Print usage
        print("Usage: python -m src.III_agents.examples <example_name>")
        print("Available examples:")
        for name in examples.keys():
            print(f"  - {name}")
        print("  - all (run all examples)")

if __name__ == "__main__":
    asyncio.run(main()) 