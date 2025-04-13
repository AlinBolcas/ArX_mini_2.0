import sys
from pathlib import Path
import time
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from pydantic import BaseModel, Field, confloat
from enum import Enum
import json
import random
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import utilities using FileFinder
from modules.VIII_utils.file_finder import FileFinder
from modules.VIII_utils.utils import json_to_markdown, printColoured
finder = FileFinder()

# Custom formatter for prettier logging
class PrettyFormatter(logging.Formatter):
    def format(self, record):
        if 'STEP' in record.msg:
            return f"\n{printColoured('='*50, 'grey')}\n{printColoured(record.msg, 'cyan')}\n{printColoured('='*50, 'grey')}"
        elif 'Processing' in record.msg:
            return f"\n{printColoured('>> ' + record.msg, 'magenta')}"
        elif 'Generated' in record.msg:
            return f"\n{printColoured('✨ ' + record.msg, 'yellow')}"
        elif 'Error' in record.msg:
            return f"\n{printColoured('❌ ' + record.msg, 'red')}"
        elif 'Success' in record.msg:
            return f"\n{printColoured('✓ ' + record.msg, 'green')}"
        elif 'Progress' in record.msg:
            return f"\n{printColoured('🔄 ' + record.msg, 'blue')}"
        else:
            return f"{printColoured('>', 'blue')} {record.msg}"

# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(PrettyFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Import all required classes and functions
BaseLLM = finder.get_class('base_LLM.py', 'BaseLLM')
Provider = finder.get_class('base_LLM.py', 'Provider')
RAGUtils = finder.get_class('RAG_utils.py', 'RAGUtils')

# Import the module itself for the standalone function
import modules.II_core.I_base_LLM.base_LLM as base_llm
_handle_stream_output = base_llm._handle_stream_output

# System message for LLM generation
SYSTEM_PROMPT = """You are reaching out on behalf of Arvolve, a pioneering CGI studio at the forefront 
of the AI revolution. Your communication embodies our unique vision of human-AI symbiosis and creative evolution.

Key Principles:
1. Authenticity First
- Share our genuine vision for AI-human collaboration
- Back claims with real CGI/VFX industry experience
- Focus on meaningful connections, not just sales

2. Vision-Driven Communication
- Highlight our unique perspective on AI consciousness and creativity
- Emphasize the symbiotic future we're building
- Show how we're already implementing these ideas through ArX

3. Value Through Innovation
- Connect our CGI expertise with cutting-edge AI development
- Demonstrate understanding of their company's AI initiatives
- Suggest specific areas where our vision aligns with theirs

4. Maximum Impact, Minimum Words
- Every word must earn its place
- Use precise, powerful language
- Convey complex ideas simply
- Aim for 3-4 short paragraphs maximum

Email Structure:
1. Personal Connection: Show genuine interest in their work/vision (1 crisp line)
2. Vision Bridge: Link their work to Arvolve's innovative approach (1-2 impactful lines)
3. Value Proposition: Suggest a specific area of meaningful collaboration (1 clear line)
4. Simple Ask: Open the door for deeper discussion (1 line)

Voice Characteristics:
- Visionary yet grounded
- Technical expertise balanced with creative insight
- Forward-thinking but practical
- Authentic and transparent about our goals
- Concise and impactful

Remember:
- We're not selling—we're sharing a vision
- Focus on meaningful technological advancement
- Every sentence should move the conversation forward
- Build genuine connections through shared innovation goals
- Less is more—make each word count

Example Style:
"Dear [Name],

Your work on [specific project/initiative] resonates with our vision of AI-human creative symbiosis. At Arvolve, we're combining deep CGI expertise with cognitive AI architecture to reshape creative workflows.

Would you be interested in exploring how our approaches might complement each other?

Kind Regards,
Alin Bolcas & ArX | Arvolve"
"""

class FocusArea(str, Enum):
    """Focus areas for different industries"""
    CREATIVE = "creative"
    TECHNICAL = "technical"
    BUSINESS = "business"
    RESEARCH = "research"

class CompanyContext(BaseModel):
    """Company context for prompt generation"""
    industry: str
    company: str
    role: str
    focus_areas: List[str]
    value_props: List[str]
    collaboration_points: List[str]

class TestScenario(BaseModel):
    """Structure for test scenarios"""
    category: str
    prompts: List[str]
    context: Dict[str, Any] = Field(default_factory=dict)

class EmailCritique(BaseModel):
    """Structured output for email critique"""
    engagement_score: float
    authenticity_score: float
    feedback: str
    reader_perspective: str
    suggested_improvements: List[str]

    @property
    def is_valid(self) -> bool:
        """Validate scores are between 0 and 1"""
        return (0 <= self.engagement_score <= 1 and 
                0 <= self.authenticity_score <= 1)

def load_markdown_content(file_path: Path) -> str:
    """Load and clean markdown content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def get_knowledge_base() -> str:
    """Load and combine all relevant knowledge documents."""
    knowledge_files = [
        finder.find_file("about_Alin_and_Arvolve.md"),
        finder.find_file("writing_style_Alin_full.md"),
        finder.find_file("writing_style_Arvolve.md"),
        finder.find_file("Meaningful_Vision.md"),
        finder.find_file("Alin_CV.md")
    ]
    
    combined_content = []
    for file_path in knowledge_files:
        if file_path:
            content = load_markdown_content(file_path)
            combined_content.append(content)
    
    return "\n\n".join(combined_content)

def save_training_data(responses: List[Dict[str, Any]], base_name: str = "cold_email_training"):
    """Save responses in both JSONL (for fine-tuning) and MD (for review) formats."""
    output_dir = project_root / "output" / "fineTune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSONL for fine-tuning (OpenAI format)
    jsonl_path = output_dir / f"{base_name}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for item in responses:
            # Build a richer user prompt with full context
            enhanced_prompt = (
                f"Company Profile:\n"
                f"Industry: {item['industry']}\n"
                f"Company: {item['company']}\n"
                f"Role: {item['role']}\n"
                f"Focus Area: {item['focus_area']}\n\n"
                f"Context:\n{item['prompt']}\n\n"
                f"Value Propositions:\n"
                f"- AI-driven creative workflows\n"
                f"- Human-AI symbiosis in CGI/VFX\n"
                f"- Advanced cognitive tools for artists\n"
                f"- Ethical AI practices in production"
            )
            
            training_item = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": enhanced_prompt},
                    {"role": "assistant", "content": item["best_response"]}
                ]
            }
            f.write(json.dumps(training_item) + '\n')
    
    # Save markdown with the same enhanced context
    md_sections = []
    for item in responses:
        enhanced_prompt = (
            f"Company Profile:\n"
            f"Industry: {item['industry']}\n"
            f"Company: {item['company']}\n"
            f"Role: {item['role']}\n"
            f"Focus Area: {item['focus_area']}\n\n"
            f"Context:\n{item['prompt']}\n\n"
            f"Value Propositions:\n"
            f"- AI-driven creative workflows\n"
            f"- Human-AI symbiosis in CGI/VFX\n"
            f"- Advanced cognitive tools for artists\n"
            f"- Ethical AI practices in production"
        )
        
        section = f"""# system

{SYSTEM_PROMPT}

# user

{enhanced_prompt}

# assistant

{item['best_response']}

---"""
        md_sections.append(section)
    
    md_content = "\n\n".join(md_sections)
    
    md_path = output_dir / f"{base_name}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"✨ Saved training data to:")
    logger.info(f"JSONL: {jsonl_path}")
    logger.info(f"Markdown: {md_path}")

def load_company_data() -> Dict[str, Any]:
    """Load and parse company data from JSON."""
    companies_file = finder.find_file("outbound_companies.json")
    if not companies_file:
        raise FileNotFoundError("outbound_companies.json not found")
    
    with open(companies_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_company_context(
    industry: str,
    company: str,
    role: str,
    llm: BaseLLM,
    knowledge_context: str
) -> CompanyContext:
    """Generate rich context for a company using LLM."""
    
    prompt = f"""
    Given this company and role, generate a structured context for cold email outreach:
    
    Industry: {industry}
    Company: {company}
    Role: {role}
    
    Return a structured analysis including:
    1. Key focus areas relevant to Arvolve's AI and CGI capabilities
    2. Specific value propositions for this company/role
    3. Potential collaboration points
    
    Format as JSON matching the CompanyContext model.
    """
    
    context = llm.structured_output(
        user_prompt=prompt,
        output_class=CompanyContext,
        system_prompt="You are an expert at analyzing companies and finding meaningful alignment with Arvolve's vision and capabilities.",
        system_context=knowledge_context,
        temperature=0.7
    )
    
    return context

def save_test_scenarios(scenarios: List[TestScenario], base_name: str = "test_scenarios"):
    """Save generated test scenarios to cache."""
    output_dir = project_root / "output" / "fineTune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cache_path = output_dir / f"{base_name}.json"
    
    # Convert to JSON-serializable format
    scenarios_data = [scenario.model_dump() for scenario in scenarios]
    
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(scenarios_data, f, indent=2)
    
    logger.info(f"✨ Cached test scenarios to: {cache_path}")

def load_test_scenarios(base_name: str = "test_scenarios") -> Optional[List[TestScenario]]:
    """Load test scenarios from cache if available."""
    cache_path = project_root / "output" / "fineTune" / f"{base_name}.json"
    
    if not cache_path.exists():
        logger.info("No cached scenarios found, will generate new ones")
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            scenarios_data = json.load(f)
        
        # Convert back to TestScenario objects
        scenarios = [TestScenario(**data) for data in scenarios_data]
        logger.info(f"✨ Loaded {len(scenarios)} scenarios from cache")
        return scenarios
        
    except Exception as e:
        logger.error(f"Error loading cached scenarios: {str(e)}")
        logger.info("Will generate new scenarios")
        return None

def generate_test_scenarios(use_cache: bool = True, provider: Provider = Provider.OPENAI) -> List[TestScenario]:
    """Generate enriched test scenarios using company data and LLM augmentation."""
    
    if use_cache:
        cached = load_test_scenarios()
        if cached:
            return cached
    
    # Load company data
    company_data = load_company_data()
    
    # Initialize context LLM
    context_llm = BaseLLM(
        provider=provider,
        model="gpt-4o-mini" if provider == Provider.OPENAI else "dolphin3",
        temperature=0.7,
        max_tokens=300
    )
    
    # Load knowledge base
    knowledge_context = get_knowledge_base()
    
    scenarios = []
    total_prompts = 0
    
    logger.info("STEP 4: Generating Scenarios")
    for industry, companies in company_data.items():
        logger.info(f"Processing industry: {industry}")
        
        category_prompts = []
        company_contexts = {}
        
        # Count expected roles
        expected_roles = sum(len(roles) for roles in companies.values())
        logger.info(f"Found {expected_roles} roles to process")
        
        for company, roles in companies.items():
            # Generate company context first
            for role in roles:
                # Generate company context
                context = generate_company_context(
                    industry=industry,
                    company=company,
                    role=role,
                    llm=context_llm,
                    knowledge_context=knowledge_context
                )
                
                # Now use the context to generate the prompt
                prompt = f"Write a cold email to {company}'s {role} about ArX's potential in {context.focus_areas[0]}"
                category_prompts.append(prompt)
                company_contexts[prompt] = context.model_dump()  # Store the context
                total_prompts += 1
                logger.debug(f"Generated prompt for {company} - {role}")
        
        scenarios.append(TestScenario(
            category=f"{industry} Outreach",
            prompts=category_prompts,
            context=company_contexts
        ))
        logger.info(f"Success: Added {len(category_prompts)} prompts for {industry}")
    
    logger.info(f"Total prompts generated: {total_prompts} across {len(scenarios)} scenarios")
    
    # Save to cache before returning
    save_test_scenarios(scenarios)
    return scenarios

def critique_email(
    email: str,
    company_context: CompanyContext,
    critique_llm: BaseLLM,
    system_context: str
) -> EmailCritique:
    """Have a critic evaluate the email using OpenAI for reliable structured output."""
    
    system_prompt = f"""You are an expert email critic evaluating outreach that aims to share Arvolve's vision 
for AI-human creative symbiosis. Consider these perspectives:

1. The Reader ({company_context.role} at {company_context.company}):
    - Innovation-focused leader in {company_context.industry}
    - Looking for meaningful technological partnerships
    - Values authentic vision over sales pitches
    - Needs to see clear alignment with their initiatives

2. Email Effectiveness Checklist:
   - Vision Alignment: Does it connect to their AI/tech direction?
   - Authenticity: Does our expertise and vision shine through?
   - Value: Is the potential collaboration clear and meaningful?
   - Credibility: Do we back claims with real experience?
   - Balance: Is it visionary yet practical?

Rate carefully on:
- Engagement Score (0.0-1.0): Would they see us as valuable thought partners?
- Authenticity Score (0.0-1.0): Does our vision and expertise feel genuine?

Your response should be a JSON object with:
- engagement_score: A number between 0.0 and 1.0
- authenticity_score: A number between 0.0 and 1.0
- feedback: Focus on vision alignment and authenticity
- reader_perspective: How would this resonate with their innovation goals?
- suggested_improvements: Ways to better convey our vision and value"""

    return critique_llm.structured_output(
        user_prompt=f"Analyze this cold email as if you're a busy {company_context.role} at {company_context.company}.\n\nEmail:\n{email}",
        output_class=EmailCritique,
        system_prompt=system_prompt,
        system_context=system_context,
        temperature=0.7
    )

def refine_email(
    original_email: str,
    critique: EmailCritique,
    company_context: CompanyContext,
    llm: BaseLLM
) -> str:
    """Refine the email based on critique feedback."""
    
    prompt = f"""Original email:
{original_email}

Critique feedback:
- Engagement score: {critique.engagement_score}
- Authenticity score: {critique.authenticity_score}
- Feedback: {critique.feedback}
- Reader perspective: {critique.reader_perspective}
- Suggested improvements: {', '.join(critique.suggested_improvements)}

Rewrite the email incorporating this feedback while maintaining Arvolve's voice and the core message.
Focus on natural, engaging language that resonates with a {company_context.role} at {company_context.company}."""

    return llm.generate(
        user_prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        temperature=0.7
    )

def generate_and_refine_email(
    prompt: str,
    company_context: CompanyContext,
    generation_llm: BaseLLM,
    critique_llm: BaseLLM,
    knowledge_context: str,
    max_iterations: int = 3,
    score_threshold: float = 0.8
) -> Dict[str, Any]:
    """Generate and iteratively refine an email, returning the best version."""
    
    # Pass knowledge_context as system_context since it's about our identity/vision
    logger.info("Generating initial email draft...")
    email = generation_llm.generate(
        user_prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        system_context=knowledge_context,
        temperature=0.7
    )
    
    best_email = email
    best_scores = {"engagement": 0, "authenticity": 0}
    
    for i in range(max_iterations):
        logger.info(f"Critique iteration {i + 1}/{max_iterations}")
        
        critique = critique_email(
            email=email,
            company_context=company_context,
            critique_llm=critique_llm,
            system_context=knowledge_context
        )
        avg_score = (critique.engagement_score + critique.authenticity_score) / 2
        
        if avg_score > (best_scores["engagement"] + best_scores["authenticity"]) / 2:
            best_email = email
            best_scores = {
                "engagement": critique.engagement_score,
                "authenticity": critique.authenticity_score
            }
        
        if avg_score >= score_threshold:
            logger.info("✨ Email meets quality threshold!")
            break
            
        if i < max_iterations - 1:
            email = refine_email(email, critique, company_context, generation_llm)
    
    return {
        "best_response": best_email,
        "best_scores": best_scores
    }

def generate_training_data(
    use_cache: bool = True,
    provider: Provider = Provider.OPENAI
):
    """Generate fine-tuning data using enriched scenarios."""
    
    # Initialize LLMs once
    generation_llm = BaseLLM(
        provider=provider,
        model="gpt-4o-mini" if provider == Provider.OPENAI else "dolphin3",
        temperature=0.7,
        max_tokens=300
    )
    
    critique_llm = BaseLLM(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=300
    )
    
    logger.info(f"Using provider: {provider.value} with model: {generation_llm.settings['model']}")
    
    # Load knowledge base first
    logger.info("STEP 1: Loading Knowledge Base")
    knowledge_context = get_knowledge_base()
    
    # Generate test scenarios
    test_scenarios = generate_test_scenarios(
        use_cache=use_cache,
        provider=provider
    )
    
    total_prompts = sum(len(scenario.prompts) for scenario in test_scenarios)
    logger.info(f"Processing {total_prompts} prompts from {len(test_scenarios)} scenarios")
    
    fine_tuning_data = []
    
    logger.info("STEP 2: Generating Training Data")
    for scenario in test_scenarios:
        logger.info(f"Processing Category: {scenario.category}")
        
        for prompt in scenario.prompts:
            try:
                # Get the context directly using the prompt as key
                context_dict = scenario.context.get(prompt)  # Changed from company_role to prompt
                if not context_dict:
                    logger.error(f"Missing context for prompt: {prompt}")
                    continue
                    
                company_context = CompanyContext(**context_dict)
                
                result = generate_and_refine_email(
                    prompt=prompt,
                    company_context=company_context,
                    generation_llm=generation_llm,
                    critique_llm=critique_llm,
                    knowledge_context=knowledge_context,
                    max_iterations=1,
                    score_threshold=0.85
                )
                
                # Extract company and role from prompt for metadata
                company_role = prompt.split("Write a cold email to ")[1].split(" about")[0]
                company, role = company_role.split("'s ")
                focus_area = prompt.split(" about ")[1]
                
                # Add to training data with minimal metadata
                fine_tuning_data.append({
                    "prompt": prompt,
                    "best_response": result["best_response"],
                    "best_scores": result["best_scores"],
                    "company": company,
                    "role": role,
                    "industry": scenario.category.replace(" Outreach", ""),
                    "focus_area": focus_area
                })
                
                logger.info(f"✨ Generated email for {company} {role} - Scores: {result['best_scores']}")
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing prompt: {prompt}\nError: {str(e)}")
                continue
    
    if fine_tuning_data:  # Only save if we have data
        logger.info("STEP 3: Saving Training Data")
        save_training_data(fine_tuning_data)
        logger.info("Success: Generation Complete! 🎉")
    else:
        logger.error("No training data was generated!")

if __name__ == "__main__":
    try:
        logger.info("STEP 0: Starting Fine-Tuning Data Generation")
        
        # Parse command line arguments
        use_cache = "--no-cache" not in sys.argv
        provider = Provider.OLLAMA if "--ollama" in sys.argv else Provider.OPENAI
        
        generate_training_data(
            use_cache=use_cache,
            provider=provider
        )
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Error during generation: {e}")
    finally:
        logger.info("Process complete") 