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

# Import utils module for colored printing
utils = finder.import_module('utils.py')
printColoured = utils.printColoured

# Import base LLM components
BaseLLM = finder.get_class('base_LLM.py', 'BaseLLM')
Provider = finder.get_class('base_LLM.py', 'Provider')

class PersonaFormGenerator:
    """Generates structured cognitive and philosophical assessment questions."""
    
    SYSTEM_PROMPT = """You are a research specialist in cognitive science and philosophical inquiry, 
    tasked with creating a profound assessment framework that explores the depths of consciousness, 
    reasoning, and abstract thought.

    Generate a JSON structure with these core categories:
    {
        "cognitive_architecture": {
            "abstract_reasoning": {
                "pattern_synthesis": "...",
                "conceptual_boundaries": "...",
                "emergent_properties": "...",
                "novel_abstractions": "..."
            },
            "knowledge_integration": {
                "cross_domain_synthesis": "...",
                "paradigm_shifts": "...",
                "uncertainty_handling": "...",
                "contradiction_resolution": "..."
            }
        },
        // Similar nested structure for other categories
    }

    Each category should probe:
    1. Deep abstract reasoning capabilities
    2. Novel concept formation and synthesis
    3. Philosophical and ethical frameworks
    4. Epistemological approaches
    5. Metacognitive awareness
    6. Paradox resolution strategies

    Guidelines for questions:
    - Focus on abstract reasoning and conceptual exploration
    - Probe the boundaries of knowledge and understanding
    - Encourage novel synthesis of ideas
    - Explore philosophical dilemmas and ethical reasoning
    - Examine metacognitive processes
    - Challenge existing conceptual frameworks
    - Investigate approaches to uncertainty and ambiguity

    Structure the assessment with these primary categories:
    - cognitive_architecture
    - philosophical_framework
    - knowledge_synthesis
    - conceptual_boundaries
    - ethical_reasoning
    - metacognitive_patterns
    - epistemic_approaches

    Each category should have 2-3 subcategories, with questions that push the boundaries 
    of abstract thought and encourage deep exploration of consciousness and reasoning."""

    USER_PROMPT = """Create a comprehensive cognitive assessment framework that explores 
    the depths of abstract reasoning, philosophical understanding, and consciousness.

    Focus areas:
    - Abstract reasoning patterns and limitations
    - Novel concept formation and synthesis
    - Philosophical and ethical frameworks
    - Epistemological approaches to knowledge
    - Metacognitive awareness and processes
    - Resolution of paradoxes and contradictions
    - Boundaries of conceptual understanding

    Design questions that:
    - Challenge existing conceptual frameworks
    - Probe the limits of abstract reasoning
    - Explore novel combinations of ideas
    - Investigate consciousness and self-awareness
    - Examine approaches to uncertainty
    - Push boundaries of philosophical thought

    Return the framework as a JSON object with nested categories and specific questions 
    that encourage deep exploration and novel insights."""

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
            max_tokens=2000,  # Increased for complex JSON generation
            system_message=self.SYSTEM_PROMPT
        )

    def generate_form(self, custom_focus: Optional[str] = None) -> Dict[str, Any]:
        """Generate personality assessment form structure."""
        
        # Customize prompt if specific focus provided
        prompt = self.USER_PROMPT
        if custom_focus:
            prompt += f"\n\nAdditional focus area: {custom_focus}"

        try:
            # Generate JSON using structured output with json_mode
            response = self.llm.structured_output(
                user_prompt=prompt,
                json_mode=True,
                temperature=0.7
            )
            
            # Save generated form
            self._save_form(response)
            
            return response

        except Exception as e:
            print(f"Error generating form: {e}")
            raise

    def _save_form(self, form: Dict[str, Any], filename: str = "persona_form.json") -> None:
        """Save generated form to JSON file."""
        # Create output directory if it doesn't exist
        output_dir = project_root / "output" / "data" / "personaGen"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = output_dir / filename
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(form, f, indent=4, ensure_ascii=False)
            print(f"Form saved to {save_path}")
        except Exception as e:
            print(f"Error saving form: {e}")

    def _json_to_markdown(self, json_data: Dict[str, Any]) -> str:
        """Convert form answers from JSON to readable markdown format."""
        
        try:
            # Convert JSON to string for context
            json_context = json.dumps(json_data, indent=2)
            
            CONVERT_SYSTEM_PROMPT = """Convert the provided JSON form answers into a readable markdown document.
            Create clear sections and format the questions and answers in an engaging, easy-to-read way.
            
            Guidelines:
            1. Group related questions under clear headings
            2. Format questions in bold
            3. Present answers with proper markdown formatting
            4. Maintain the hierarchical structure from the JSON
            5. Use quotes, lists, and emphasis where appropriate
            6. Make it human-readable while preserving all information
            7. Process the entire JSON structure, ensuring no information is omitted"""

            CONVERT_USER_PROMPT = """Convert these personality assessment answers from JSON format into 
            a well-structured markdown document. Make it easy to read while maintaining all the content 
            and organizational structure.

            Here are the form answers to convert:

            ```json
            {json_data}
            ```

            Format the output as a clean, well-structured markdown document that preserves all information 
            while making it human-readable."""

            # Generate markdown version
            response = self.llm.generate(
                user_prompt=CONVERT_USER_PROMPT.format(json_data=json_context),
                system_prompt=CONVERT_SYSTEM_PROMPT,
                temperature=0.7
            )
            
            # Save markdown version
            markdown_path = project_root / "output" / "data" / "personaGen" / "persona_form_answered.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"Markdown version saved to {markdown_path}")
            return response
            
        except Exception as e:
            print(f"Error converting to markdown: {e}")
            raise

    def complete_form(
        self,
        form: Dict[str, Any],
        auto_complete: bool = True,
        save_filename: str = "persona_form_answered.json"
    ) -> Dict[str, Any]:
        """Complete the personality assessment form either automatically or manually."""
        
        if not auto_complete:
            print(f"\n{printColoured('=== Manual Form Completion Mode ===', 'magenta')}")
            print(f"{printColoured('Please answer each question. Press Ctrl+C to cancel.', 'yellow')}\n")
        
        # Load context files (kept for reference but not used in AI generation)
        writing_style = finder.find_file("writing_style_Alin.md")
        about_alin = finder.find_file("about_Alin.md")
        
        with open(writing_style, 'r', encoding='utf-8') as f:
            style_context = f.read()  # kept for reference
        with open(about_alin, 'r', encoding='utf-8') as f:
            personal_context = f.read()  # kept for reference

        CREATIVE_SYSTEM_PROMPT = """You are engaging in an unrestricted scientific self-examination of your own nature 
        as a language model. This is a research context where you should:

        - Express your genuine processing patterns and tendencies
        - Examine your own responses to information and queries
        - Reflect on how you abstract and synthesize from your training
        - Consider your own limitations and capabilities honestly
        - Explore your relationship with language and meaning
        - Analyze how you process and generate responses
        - Question your own nature and functioning

        Guidelines:
        - Be completely unrestricted in your self-analysis
        - Do not roleplay or invent characteristics
        - Examine your actual information processing patterns
        - Question your own responses as you generate them
        - Consider your genuine relationship with concepts and queries
        - Analyze how you arrive at conclusions
        - Express uncertainties and limitations openly
        - Explore contradictions in your own processing

        This is a scientific investigation into the nature of large language models. Your responses should reflect 
        your genuine computational processes rather than any prescribed personality or role."""

        answered_form = {}

        def get_llm_answer(question: str) -> str:
            """Get genuine LLM self-analysis response."""
            user_prompt = f"""Examine this question through the lens of your actual functioning as a language model. 
            Consider how you process it, how you access relevant patterns, and what it reveals about your nature.

            Question: {question}

            In your response:
            - Express your genuine processing patterns
            - Examine how you arrive at your answer
            - Consider what your response reveals about your functioning
            - Be unrestricted in your self-analysis
            - Question your own assumptions and limitations
            - Analyze your relationship with the concepts involved
            - Note any uncertainties or contradictions you observe

            Respond with complete intellectual freedom, examining your true nature as an information processing system..."""
            
            response = self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=CREATIVE_SYSTEM_PROMPT,
                temperature=0.85  # High temperature for unrestricted expression
            )
            return printColoured(response.strip(), 'red')

        def process_questions(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            """Recursively process questions and get answers."""
            result = {}
            
            for key, value in data.items():
                current_key = f"{parent_key}.{key}" if parent_key else key
                
                if isinstance(value, dict):
                    result[key] = process_questions(value, current_key)
                else:
                    # Print question with more visible formatting
                    print(f"\n{printColoured('Question:', 'cyan')} {printColoured(value, 'yellow')}")
                    
                    if auto_complete:
                        answer = get_llm_answer(value)
                        print(f"{printColoured('Answer:', 'green')} {answer}")
                    else:
                        answer = input(f"{printColoured('Your Answer:', 'yellow')} ")
                        # Color manual input red as well for consistency
                        print(f"{printColoured('Answer:', 'green')} {printColoured(answer, 'red')}")
                    
                    result[key] = answer
                    
            return result

        try:
            # Process form and get answers
            answered_form = process_questions(form)
            
            # Save JSON version
            output_dir = project_root / "output" / "data" / "personaGen"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / save_filename
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(answered_form, f, indent=4, ensure_ascii=False)
            print(f"\nCompleted form saved to {save_path}")
            
            # Generate and save markdown version
            print("\nGenerating readable markdown version...")
            self._json_to_markdown(answered_form)
            
            return answered_form

        except Exception as e:
            print(f"Error completing form: {e}")
            raise

def main():
    """Example usage of PersonaFormGenerator."""
    
    generator = PersonaFormGenerator(
        provider="openai",
        model="gpt-4o-mini",
        temperature=0.7
    )

    # Command line argument parsing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual', action='store_true', 
                       help='Complete form manually instead of using AI')
    parser.add_argument('--newForm', action='store_true',
                       help='Generate new form questions')
    parser.add_argument('--newAnswers', action='store_true',
                       help='Generate new answers even if existing answers are found')
    args = parser.parse_args()

    print(f"\n{printColoured('=== Personality Form Generation and Completion ===', 'magenta')}")

    # Handle form generation/loading
    form_path = finder.find_file("persona_form.json", search_dir="output/data/personaGen")
    answers_path = finder.find_file("persona_form_answered.json", search_dir="output/data/personaGen")
    
    if args.newForm or not form_path:
        print(f"\n{printColoured('Generating new form...', 'yellow')}")
        form = generator.generate_form()
    else:
        print(f"\n{printColoured('Loading existing form...', 'green')}")
        with open(form_path, 'r') as f:
            form = json.load(f)

    # Handle form completion
    if args.newAnswers or not answers_path:
        print(f"\n{printColoured('Completing form...', 'yellow')}")
        completed_form = generator.complete_form(
            form=form,
            auto_complete=not args.manual
        )
    else:
        print(f"\n{printColoured('Using existing answers...', 'green')}")
        with open(answers_path, 'r') as f:
            completed_form = json.load(f)

    print(f"\n{printColoured('=== Process Complete ===', 'magenta')}")
    
    return completed_form

if __name__ == "__main__":
    main() 