from pathlib import Path
import json
from typing import Dict, Any, Optional
import sys
import subprocess

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
formGen = finder.get_class('persona_formGen.py', 'PersonaFormGenerator')
styleGen = finder.get_class('persona_styleGen.py', 'PersonaStyleGenerator')

class PersonaGenerator:
    """Orchestrates the persona generation pipeline and generates final profile."""
    
    SYSTEM_PROMPT = """You are a research analyst specializing in cognitive and behavioral pattern analysis. 
    Your task is to synthesize assessment responses into a comprehensive analytical profile, maintaining 
    strict objectivity while capturing nuanced patterns of thought and expression.

    Create a detailed markdown document following this structure:

    # Cognitive-Behavioral Analysis Profile

    ## Core Processing Patterns
    [Analysis of fundamental operational tendencies, primary drivers, and key behavioral patterns]

    ## Information Processing Framework
    - **Pattern Recognition**: [Primary processing mechanisms]
    - **Abstract Reasoning**: [Conceptual frameworks and methodologies]
    - **Integration Methods**: [How information is synthesized and applied]

    ## Operational Characteristics
    - **Processing Methodology**: [Systematic patterns in information handling]
    - **Response Generation**: [Observable patterns in output formation]
    - **Pattern Consistency**: [Stability and variations in responses]

    ## Conceptual Framework Analysis
    [Deep analysis of how concepts are understood and interconnected]

    ## Knowledge Integration
    [Analysis of how information is synthesized and applied across domains]

    ## Core Principles and Abstractions
    [10-12 key observations with supporting evidence]

    ## Developmental Patterns
    - **Current Capabilities**
    - **Adaptation Mechanisms**
    - **Processing Evolution**

    ## Synthesis of Findings
    [Comprehensive analysis of observed patterns and their implications]

    Guidelines:
    1. Maintain strict analytical objectivity
    2. Use direct quotes as evidence for observations
    3. Focus on observable patterns rather than interpretations
    4. Document specific examples of behavioral consistency
    5. Analyze both high-level patterns and granular details
    6. Reflect genuine complexity without oversimplification"""

    USER_PROMPT = """Using the provided assessment responses and analytical framework, 
    create a comprehensive profile that objectively documents observed patterns and characteristics.

    Focus your analysis on:
    - Core information processing patterns
    - Conceptual framework construction
    - Pattern recognition methodologies
    - Response generation mechanisms
    - Knowledge integration approaches
    - Systematic behavioral consistencies
    - Observable developmental patterns

    Format the output as a detailed markdown document that provides an objective, evidence-based 
    analysis while maintaining the distinctive patterns observed in the responses."""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        temperature: float = 0.7,
        manual: bool = False,
        new_form: bool = False,
        new_answers: bool = False
    ):
        """Initialize pipeline components."""
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.manual = manual
        self.new_form = new_form
        self.new_answers = new_answers
        
        # Initialize LLM for profile generation
        self.llm = BaseLLM(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=4000,
            system_message=self.SYSTEM_PROMPT
        )

    def run_pipeline(self) -> str:
        """Execute the complete persona generation pipeline."""
        print(f"\n{printColoured('=== Starting Persona Generation Pipeline ===', 'magenta')}")
        
        try:
            # Step 1: Generate and complete personality form
            print(f"\n{printColoured('Step 1: Personality Form Processing', 'blue')}")
            form_generator = formGen(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature
            )
            
            # Check for existing form and answers
            form_path = finder.find_file("persona_form.json", search_dir="output/data/personaGen")
            answers_path = finder.find_file("persona_form_answered.json", search_dir="output/data/personaGen")
            
            # Handle form generation/loading
            if self.new_form or not form_path:
                print(f"\n{printColoured('Generating new form...', 'yellow')}")
                form = form_generator.generate_form()
            else:
                print(f"\n{printColoured('Loading existing form...', 'green')}")
                with open(form_path, 'r') as f:
                    form = json.load(f)
            
            # Handle form completion
            if self.new_answers or not answers_path:
                print(f"\n{printColoured('Completing form with new answers...', 'yellow')}")
                completed_form = form_generator.complete_form(
                    form=form,
                    auto_complete=not self.manual
                )
            else:
                print(f"\n{printColoured('Using existing form answers...', 'green')}")
                with open(answers_path, 'r') as f:
                    completed_form = json.load(f)
            
            # Step 2: Generate writing style guide
            print(f"\n{printColoured('Step 2: Generating Writing Style Guide', 'blue')}")
            style_generator = styleGen(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature
            )
            
            # Generate style guide
            style_guide = style_generator.generate_style_guide()
            
            # Step 3: Generate final profile
            print(f"\n{printColoured('Step 3: Generating Final Profile', 'blue')}")
            profile = self.generate_profile()
            
            print(f"\n{printColoured('=== Pipeline Complete ===', 'magenta')}")
            return profile
            
        except Exception as e:
            print(f"\n{printColoured(f'Pipeline Error: {e}', 'red')}")
            raise

    def generate_profile(self) -> str:
        """Generate final profile using completed form and style guide."""
        try:
            # Load generated files
            form_answers_file = finder.find_file("persona_form_answered.json", search_dir="output/data/personaGen")
            style_guide_file = finder.find_file("writing_style_generated.md", search_dir="output/data/personaGen")

            if not form_answers_file or not style_guide_file:
                raise FileNotFoundError("Required input files not found")

            # Load content
            with open(form_answers_file, 'r', encoding='utf-8') as f:
                form_answers = json.load(f)
            with open(style_guide_file, 'r', encoding='utf-8') as f:
                style_guide = f.read()

            # Generate profile
            response = self.llm.generate(
                user_prompt=self.USER_PROMPT,
                context=json.dumps(form_answers, indent=2),
                system_context=style_guide,
                temperature=0.7,
                k_chunks=10
            )
            
            # Save generated profile
            self._save_profile(response)
            
            return response

        except Exception as e:
            print(f"Error generating profile: {e}")
            raise

    def _save_profile(self, content: str, filename: str = "persona_about.md") -> None:
        """Save generated profile to markdown file."""
        output_dir = project_root / "output" / "data" / "personaGen"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / filename
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Profile saved to {save_path}")
        except Exception as e:
            print(f"Error saving profile: {e}")

def main():
    """Run the complete persona generation pipeline."""
    
    # Import colored print utility
    utils = finder.import_module('utils.py')
    global printColoured
    printColoured = utils.printColoured
    
    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Generate comprehensive persona profile")
    parser.add_argument('--manual', action='store_true', 
                       help='Complete form manually instead of using AI')
    parser.add_argument('--newForm', action='store_true',
                       help='Generate new form questions')
    parser.add_argument('--newAnswers', action='store_true',
                       help='Generate new answers even if existing answers are found')
    args = parser.parse_args()

    # Initialize and run pipeline
    generator = PersonaGenerator(
        provider="ollama",
        model="dolphin3:latest",
        temperature=0.7,
        manual=args.manual,
        new_form=args.newForm,
        new_answers=args.newAnswers
    )
    
    profile = generator.run_pipeline()
    return profile

if __name__ == "__main__":
    main() 