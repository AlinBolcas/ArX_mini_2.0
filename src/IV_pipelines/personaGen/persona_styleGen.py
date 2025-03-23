from pathlib import Path
import json
from typing import Dict, Any, Optional
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
finder = FileFinder()

# Import base LLM components
BaseLLM = finder.get_class('base_LLM.py', 'BaseLLM')
Provider = finder.get_class('base_LLM.py', 'Provider')

class PersonaStyleGenerator:
    """Generates writing style guide based on personality form answers."""
    
    SYSTEM_PROMPT = """You are an expert in communication analysis and style guide creation. Your task is to 
    create a comprehensive writing style guide that serves as an instruction manual for maintaining authentic voice 
    in written communication.

    The guide should be structured in markdown format as follows:

    # Writing Style Guide

    ## Overview
    [Synthesize core communication traits into a concise description of the overall voice and style]

    ## Core Tone & Principles
    1. **[Primary Principle]**
       - Detailed explanation
       - Examples of application
       - Common pitfalls to avoid

    2. **[Secondary Principle]**
     [Continue with 3-4 key principles]

    ## Voice Characteristics
    ### Sentence Structure
    - Preferred patterns
    - Length and complexity
    - Rhythm and flow

    ### Vocabulary and Expression
    - Word choice preferences
    - Common phrases or terms
    - Technical vs. casual balance
    - Metaphors and analogies usage

    ### Emotional Range
    - Expression of feelings
    - Use of emphasis
    - Handling sensitive topics

    ## Communication Contexts
    ### Professional Communication
    - Emails and formal documents
    - Technical discussions
    - Project communications

    ### Casual Interaction
    - Social media
    - Personal messages
    - Team discussions

    ## Adaptation Guidelines
    - How to adjust tone for different audiences
    - Maintaining authenticity across contexts
    - Balancing formality with personality

    ## Do's & Don'ts
    **Do:**
    - [3-5 key practices to maintain]

    **Don't:**
    - [3-5 key practices to avoid]

    ## Final Note
    [Synthesis of how these elements come together]

    Focus on creating practical, actionable guidelines that capture the essence of the individual's 
    communication style while providing clear instruction for consistent application."""

    USER_PROMPT = """Based on the provided personality assessment responses, create a detailed writing 
    style guide that will serve as a comprehensive instruction manual for maintaining authentic voice 
    in written communication.

    Extract and analyze:
    1. Natural expression patterns and preferences
    2. Vocabulary choices and sentence structures
    3. Balance of technical and emotional content
    4. Adaptation strategies across contexts
    5. Unique stylistic elements and quirks
    6. Core principles that guide communication

    Create clear, practical guidelines that:
    - Capture authentic voice and personality
    - Provide specific examples and applications
    - Address different communication contexts
    - Include both high-level principles and detailed practices
    - Maintain consistency with personal values and traits

    The guide should be detailed enough to serve as a standalone reference for maintaining 
    authentic communication style across all written contexts."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7
    ):
        """Initialize with specified LLM provider and settings."""
        self.llm = BaseLLM(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            system_message=self.SYSTEM_PROMPT
        )

    def generate_style_guide(self) -> str:
        """Generate writing style guide from personality form answers."""
        try:
            # Load form answers using FileFinder
            form_path = finder.find_file("persona_form_answered.json", search_dir="output/data/personaGen")
            if not form_path:
                raise FileNotFoundError("Form answers file not found")

            # Load content
            with open(form_path, 'r', encoding='utf-8') as f:
                form_answers = json.load(f)

            # Convert form answers to context string
            context = json.dumps(form_answers, indent=2)
            
            # Generate style guide
            response = self.llm.generate(
                user_prompt=self.USER_PROMPT,
                context=context,
                temperature=0.7,
                k_chunks=10
            )
            
            # Save generated style guide
            self._save_style_guide(response)
            
            return response

        except Exception as e:
            print(f"Error generating style guide: {e}")
            raise

    def _save_style_guide(self, content: str, filename: str = "writing_style_generated.md") -> None:
        """Save generated style guide to markdown file."""
        # Create output directory if it doesn't exist
        output_dir = project_root / "output" / "data" / "personaGen"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / filename
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Style guide saved to {save_path}")
        except Exception as e:
            print(f"Error saving style guide: {e}")

def main():
    """Example usage of PersonaStyleGenerator."""
    
    # Import colored print utility
    utils = finder.import_module('utils.py')
    global printColoured
    printColoured = utils.printColoured
    
    # Initialize generator
    generator = PersonaStyleGenerator(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7
    )

    print(f"\n{printColoured('=== Writing Style Guide Generation ===', 'magenta')}")

    # Generate style guide
    print(f"\n{printColoured('Generating style guide...', 'blue')}")
    style_guide = generator.generate_style_guide()

    print(f"\n{printColoured('=== Style Guide Generation Complete ===', 'magenta')}")
    
    return style_guide

if __name__ == "__main__":
    main() 