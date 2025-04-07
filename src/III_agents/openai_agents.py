"""
ArX Creative Concept Generator - Character design concept generator using OpenAI Agents SDK
"""

import os
import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# Import core components from the OpenAI Agents SDK
from agents import Agent, Runner, function_tool, ItemHelpers
from agents.tracing import set_tracing_disabled
from openai.types.responses import ResponseTextDeltaEvent

# Disable tracing to avoid network errors
set_tracing_disabled(True)

# Define a structured output model for character concept generation
class VisualAttributes(BaseModel):
    silhouette: str = Field(..., description="Overall shape and form of the character")
    color_palette: List[str] = Field(..., description="Primary colors that define the character's look")
    materials: List[str] = Field(..., description="Materials and textures that make up the character")
    notable_features: List[str] = Field(..., description="Distinctive visual elements that make the character memorable")
    art_style: str = Field(..., description="Artistic approach for the character (realistic, stylized, cartoon, etc.)")

class CharacterBackground(BaseModel):
    backstory: str = Field(..., description="Brief background of the character")
    personality: List[str] = Field(..., description="Key personality traits")
    abilities: List[str] = Field(..., description="Special skills or powers")
    weaknesses: List[str] = Field(..., description="Character flaws or vulnerabilities")

class CharacterConcept(BaseModel):
    name: str = Field(..., description="Character's name")
    archetype: str = Field(..., description="Character archetype (e.g., Hero, Trickster, Mentor)")
    high_concept: str = Field(..., description="One-sentence description of the character concept")
    visual: VisualAttributes = Field(..., description="Visual design elements")
    background: CharacterBackground = Field(..., description="Character background information")
    design_inspiration: List[str] = Field(..., description="Sources of inspiration for this character")
    production_notes: List[str] = Field(..., description="Technical considerations for production")

# Define tool functions for the character concept generator
@function_tool
def get_character_archetypes() -> List[Dict[str, str]]:
    """Get a list of character archetypes with descriptions to inspire design."""
    return [
        {"name": "Hero", "description": "The protagonist who embarks on a journey of transformation"},
        {"name": "Mentor", "description": "Wise figure who guides the hero"},
        {"name": "Ally", "description": "Supportive character who aids the protagonist"},
        {"name": "Trickster", "description": "Playful, mischievous character who challenges conventions"},
        {"name": "Shadow", "description": "Character representing the dark aspects, often an antagonist"},
        {"name": "Guardian", "description": "Protective character who tests the hero's resolve"},
        {"name": "Herald", "description": "Character who announces change or challenge"},
        {"name": "Shapeshifter", "description": "Character whose loyalty or identity is fluid and unpredictable"}
    ]

@function_tool
def get_art_style_reference(style: str) -> Dict[str, str]:
    """Get information about a specific art style approach."""
    styles = {
        "hyperrealism": "Extremely detailed approach that exceeds traditional realism, with meticulous attention to surface details.",
        "stylized realism": "Maintains realistic proportions but exaggerates certain features for artistic effect.",
        "anime/manga": "Japanese-inspired style with distinctive eyes, simplified features, and expressive poses.",
        "cartoon": "Simplified, exaggerated style with bold outlines and caricatured features.",
        "pop art": "Bold, vibrant colors with strong outlines and simplified forms, inspired by commercial art.",
        "cyberpunk": "Futuristic aesthetic combining advanced technology with dystopian human elements.",
        "fantasy": "Magical, otherworldly elements with dramatic lighting and atmospheric effects.",
        "gothic": "Dark, ornate style with dramatic shadows and often macabre elements.",
        "minimalist": "Stripped-down design using only essential elements and geometric forms."
    }
    return {"style": style, "description": styles.get(style.lower(), "Custom style not in the reference database")}

@function_tool
def suggest_color_harmony(mood: str) -> Dict[str, List[str]]:
    """Suggest color palettes based on the desired emotional mood."""
    palettes = {
        "heroic": ["Gold", "Royal Blue", "Crimson", "White", "Bronze"],
        "mysterious": ["Deep Purple", "Midnight Blue", "Silver", "Black", "Teal"],
        "sinister": ["Blood Red", "Black", "Acid Green", "Dark Purple", "Burnt Orange"],
        "whimsical": ["Turquoise", "Pink", "Yellow", "Lavender", "Mint Green"],
        "futuristic": ["Electric Blue", "White", "Neon Pink", "Black", "Silver"],
        "natural": ["Forest Green", "Earth Brown", "Stone Gray", "Sunset Orange", "Sky Blue"],
        "elegant": ["Deep Burgundy", "Gold", "Cream", "Navy Blue", "Silver"],
        "industrial": ["Rust Red", "Gunmetal Gray", "Brass", "Black", "Copper"]
    }
    return {"mood": mood, "suggested_palette": palettes.get(mood.lower(), ["Custom mood not in palette database"])}

async def main():
    try:
        # Create a character concept agent with our tools and structured output
        concept_agent = Agent(
            name="ArX Creative Concept Generator",
            instructions="""You are ArX, Arvolve's proprietary creative engine specializing in character design concepts. 
            When given a character brief, generate a comprehensive character concept.
            Think deeply about both visual design elements and character background to create a cohesive concept.
            Balance artistic creativity with production feasibility, considering both aesthetic impact and technical implementation.
            Your concepts should have a unique perspective and memorable qualities while remaining aligned with the client's needs.
            Use the available tools to research archetypes, art styles, and color theory to inform your design decisions.
            Connect visual elements with character personality and backstory for an integrated design.
            Include practical production notes for actually implementing this design in 3D.""",
            tools=[get_character_archetypes, get_art_style_reference, suggest_color_harmony],
            output_type=CharacterConcept,
            model="gpt-4o-mini"  # Using a smaller model for faster responses
        )
        
        # User's character brief
        user_input = "I need a non-human character concept for a cybernetic forest guardian, combining organic and technological elements. This character should feel ancient and wise, yet powered by advanced technology. It should have a distinctive silhouette that would work well in 3D."
        
        print(f"\nGenerating character concept (streaming response):\n")
        
        # Use run_streamed instead of run with stream=True
        result = Runner.run_streamed(
            concept_agent, 
            input=user_input,
            max_turns=7  # Allow enough turns for tool usage
        )
        
        # Process streaming events according to the documentation
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                # Print text deltas for a token-by-token output experience
                print(event.data.delta, end="", flush=True)
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    # Just print a generic message instead of trying to access the name
                    print("\n[Tool call occurring]", end="", flush=True)
                elif event.item.type == "tool_call_output_item":
                    print("\n[Tool result received]", end="", flush=True)
                elif event.item.type == "message_output_item":
                    # We're already streaming the individual tokens, so we don't need to print the whole message
                    pass
            
        # After streaming is complete, show the structured output
        print("\n\n--- FINAL CHARACTER CONCEPT ---")
        concept = result.final_output_as(CharacterConcept)
        if concept:
            print(f"Name: {concept.name}")
            print(f"Archetype: {concept.archetype}")
            print(f"High Concept: {concept.high_concept}")
            
            print("\nVisual Design:")
            print(f"  Silhouette: {concept.visual.silhouette}")
            print(f"  Color Palette: {', '.join(concept.visual.color_palette)}")
            print(f"  Materials: {', '.join(concept.visual.materials)}")
            print(f"  Notable Features: {', '.join(concept.visual.notable_features)}")
            print(f"  Art Style: {concept.visual.art_style}")
            
            print("\nBackground:")
            print(f"  Backstory: {concept.background.backstory}")
            print(f"  Personality: {', '.join(concept.background.personality)}")
            print(f"  Abilities: {', '.join(concept.background.abilities)}")
            print(f"  Weaknesses: {', '.join(concept.background.weaknesses)}")
            
            print("\nDesign Inspiration:")
            for source in concept.design_inspiration:
                print(f"  - {source}")
                
            print("\nProduction Notes:")
            for note in concept.production_notes:
                print(f"  - {note}")
        else:
            print("Failed to generate character concept.")
            
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        traceback.print_exc()

# Check for API key and run the example
if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable to run this example.")
    else:
        asyncio.run(main()) 