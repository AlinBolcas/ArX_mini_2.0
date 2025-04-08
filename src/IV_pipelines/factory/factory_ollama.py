import os
import sys
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import subprocess # Added for running commands
import shutil # For directory operations

# Ensure utils can be imported (handles potential path issues)
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"[FactoryOllama] Added project root to sys.path: {project_root}")


# Local imports after path correction
try:
    from src.I_integrations.ollama_API import OllamaWrapper
    from src.IV_pipelines.factory import factory_utils as utils # Import our new utils
    from src.VI_utils.utils import printColoured # For colored printing
except ImportError as e:
    print(f"[FactoryOllama Critical Error] Failed to import necessary modules: {e}")
    print("Please ensure factory_utils.py and ollama_API.py are accessible and project structure is correct.")
    sys.exit(1) # Exit if core components are missing


# Basic logging setup (can be configured further)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# --- Constants ---
# List of common Python built-in modules that shouldn't be in requirements.txt
# Expanded this list
BUILT_IN_MODULES = [
    'os', 'sys', 'time', 're', 'json', 'math', 'random', 'datetime', 'tkinter',
    'argparse', 'collections', 'csv', 'glob', 'hashlib', 'heapq', 'itertools',
    'logging', 'multiprocessing', 'pathlib', 'pickle', 'shutil', 'socket',
    'sqlite3', 'statistics', 'subprocess', 'threading', 'unittest', 'urllib',
    'uuid', 'xml', 'zipfile', 'io', 'contextlib', 'functools'
]

AGENT_COLORS = {
    "SYSTEM": "magenta",
    "PLANNER": "cyan",
    "CODER": "blue",
    "NAMER": "yellow",
    "SAVER": "white",
    "TESTER": "red",
    "REFINER": "magenta", # Using SYSTEM color for refinement steps
}


class FactoryOllama:
    """
    An autonomous coding factory using Ollama local LLMs.
    Iteratively plans, codes, tests, and refines based on a goal.
    """
    # Class constants for prompts
    PLANNER_SYSTEM_MSG = """
You are an expert AI system architect and Python programmer. Your goal is to take a high-level objective
and create a detailed, step-by-step technical plan for a Python script.
Focus on: Modules, functions, classes, logic flow, necessary libraries, and potential challenges.
IMPORTANT: Include a section at the end listing all required pip packages, like this:

Required Packages:
- requests
- numpy
- pandas

If no external packages are needed, state 'Required Packages: None'.
Respond ONLY with the plan, formatted clearly.
"""
    CODER_SYSTEM_MSG = """
You are an expert AI Python programmer. Your task is to write clean, functional, and well-commented Python code
based *strictly* on the provided plan.
Include necessary imports. Place ALL code within a single ```python ... ``` block.
Respond ONLY with the code block. Do not add explanations before or after the block.
"""
    NAMER_SYSTEM_MSG = """
You are an AI assistant specializing in clear and concise naming conventions.
Based on the provided context (goal or plan), suggest a short, descriptive, snake_case filename ending in '.py'.
Respond ONLY with the filename.
"""
    REFINER_SYSTEM_MSG = """
You are an expert AI debugger and Python programmer.
You will receive the original goal, the previous plan, the failed code, and the resulting error trace.
Analyze the error and the code. Provide a step-by-step analysis of the issue and clear SUGGESTIONS for how to fix it.

IMPORTANT: If the error is a 'ModuleNotFoundError' for a package 'X', your suggestions MUST explicitly state to potentially add 'X' to the list of required packages. Example: "Suggestion: Add 'requests' to the required packages list."

Focus ONLY on the analysis and actionable suggestions. Respond ONLY with your analysis and suggestions.
"""

    # Added System message for Planner acting as Supervisor
    SUPERVISOR_SYSTEM_MSG = """
You are an expert AI system architect and Python programmer acting as a SUPERVISOR.
You will receive the original goal, the previous plan, the failed code, the error trace, and analysis/suggestions from a Refiner agent.

Your tasks are:
1.  Analyze all the provided information.
2.  Decide the best course of action (e.g., revise the plan, confirm code is okay but needs requirements, identify unfixable issue).
3.  If a plan revision is needed, generate a concise, actionable REVISED PLAN focusing *only* on the necessary changes for the CODER.
4.  If the error is a ModuleNotFoundError or similar dependency issue:
    a. Verify the Refiner correctly identified the missing package(s).
    b. Ensure the required package(s) are listed correctly in the 'Required Packages' section below.
    c. Your revised plan should state: "Code logic appears correct. Ensure listed requirements are installed and retry." Do NOT instruct the coder to perform installation steps.
5.  If the issue seems genuinely unresolvable with the current approach, state that clearly.

IMPORTANT: Ensure your revised plan instructs the CODER to produce a COMPLETE, functional script if code changes *are* needed.
IMPORTANT: Include a section listing required pip packages if the plan changes them, like this:

Required Packages:
- package1
- package2

Respond ONLY with your decision and the revised plan (if applicable).
"""

    def __init__(
        self,
        goal: str = "Create a simple python script that prints 'Hello World'.",
        model: str = "codellama:latest", # Recommend code-specific model
        temperature: float = 0.5, # Slightly lower temp for more deterministic coding
        max_tokens: int = 4096,
        auto_pull: bool = True,
        max_retries: int = 3, # Max refinement loops
        verbose: bool = False # Added verbose toggle for streaming responses
    ):
        """
        Initialize the Ollama Factory.

        Args:
            goal: The overall objective for the code generation.
            model: The Ollama model to use (e.g., "codellama:latest").
            temperature: LLM temperature for generation.
            max_tokens: Max tokens for LLM responses.
            auto_pull: Whether to automatically pull the Ollama model if not present.
            max_retries: (Currently unused) Number of times to potentially retry on failure.
            verbose: When True, streams agent responses in real-time.
        """
        self.goal = goal
        self.max_retries = max_retries
        self.verbose = verbose # Store verbose setting

        # State tracking
        self.current_plan = ""
        self.current_code = ""
        self.current_filename = "generated_code.py"
        self.project_name = Path(self.current_filename).stem # Default project name
        self.requirements = []
        self.refinement_analysis = ""
        self.current_version = 0
        self.test_results = ""
        self.last_error = ""
        self.run_history = []
        self.run_context = f"Initial Goal: {self.goal}\n"
        self.project_base_dir = Path("data/output/code") # Changed back to 'code' per user request

        logger.info(f"Initializing Ollama Factory with model: {model}, Max Retries: {self.max_retries}")
        print(printColoured(f"🎯 Initial Goal: {self.goal}", AGENT_COLORS["SYSTEM"]))

        try:
            # We will set the system message per-call based on the agent role
            self.api = OllamaWrapper(
                model=model,
                auto_pull=auto_pull,
                system_message="", # Set dynamically per agent
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info("OllamaWrapper initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OllamaWrapper: {e}")
            raise

        logger.info(f"Factory Goal set to: {self.goal}")

    def _add_history(self, agent_name: str, output: Any, color: str = "grey"):
        """Adds a step to the factory's history and prints it."""
        self.run_history.append((agent_name, output))
        print(printColoured(f"[{agent_name.upper()}] Output recorded.", color))
        logger.debug(f"[{agent_name.upper()}] Recorded output: {str(output)[:100]}...") # Log snippet

    def _call_llm(self, agent_name: str, system_message: str, user_prompt: str, **kwargs) -> str:
        """Helper function to call the LLM with specific role and prompt."""
        color = AGENT_COLORS.get(agent_name.upper(), "grey")
        print(printColoured(f"\n🔄 Calling {agent_name.upper()} Agent...", color))
        try:
            # Temporarily set system message for this specific call
            original_system_msg = self.api.system_message
            self.api.system_message = system_message

            # Enable streaming if verbose mode is on
            use_stream = self.verbose
            response = ""

            if use_stream:
                # Stream the response and display in real-time
                print(printColoured(f"\n📝 {agent_name.upper()} Response (streaming):", color))
                stream_response = self.api.chat_completion(
                    user_prompt=user_prompt,
                    stream=True,
                    **kwargs
                )
                
                full_response = ""
                for chunk in stream_response:
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print("\n") # Add newline after streaming completes
                response = full_response
            else:
                # Get full response at once (non-streaming)
                response = self.api.chat_completion(
                    user_prompt=user_prompt,
                    stream=False, # Get full response for processing
                    **kwargs
                )

            # Restore original system message
            self.api.system_message = original_system_msg

            response_content = response.strip()
            self._add_history(agent_name, response_content, color)
            logger.info(f"{agent_name.upper()} Agent call successful.")
            return response_content
        except Exception as e:
            logger.error(f"Error during {agent_name.upper()} Agent call: {e}")
            self.last_error = f"{agent_name.upper()} Error: {e}"
            self._add_history(f"{agent_name} Error", str(e), "red")
            return "" # Return empty string on error

    def _extract_requirements(self):
        """Parses the current plan to extract required pip packages."""
        if not self.current_plan:
            logger.warning("Cannot extract requirements, no plan available.")
            return

        logger.info("Extracting requirements from plan...")
        # Use regex to find the 'Required Packages:' section and list items
        req_section_match = re.search(r"Required Packages:\s*\n(.*?)(?:\n\n|\Z)", self.current_plan, re.DOTALL | re.IGNORECASE)
        extracted_reqs = []

        if req_section_match:
            package_lines = req_section_match.group(1).strip()
            if package_lines.lower() != 'none':
                # Find lines starting with '-' or '*'
                potential_packages = re.findall(r"^[\s]*[-\*][\s]+(\S+)", package_lines, re.MULTILINE)
                for pkg in potential_packages:
                    # Basic cleaning: remove version constraints for now, handle complex cases later if needed
                    clean_pkg = re.split(r'[=<>~]', pkg)[0].strip()
                    if clean_pkg:
                        extracted_reqs.append(clean_pkg)

        # Filter out built-in modules that should not be in requirements
        # Use the constant list defined at the top
        filtered_reqs = [req for req in extracted_reqs if req.lower() not in [m.lower() for m in BUILT_IN_MODULES]]
        
        if len(filtered_reqs) != len(extracted_reqs):
            removed = set(extracted_reqs) - set(filtered_reqs)
            logger.info(f"Filtered out built-in modules from requirements: {removed}")
            self._add_history("Requirements Filtered", f"Removed built-in modules: {removed}")
            
        if filtered_reqs:
            self.requirements = list(set(filtered_reqs)) # Remove duplicates
            logger.info(f"Extracted requirements: {self.requirements}")
            self._add_history("Requirements Extracted", self.requirements)
        else:
            logger.info("No requirements specified in the plan.")
            self.requirements = [] # Ensure it's empty if none found
            self._add_history("Requirements Extracted", "None")

    def planner(self, current_goal_or_refinement_prompt: str) -> bool:
        """Generates the initial plan."""
        agent_name = "PLANNER"
        color = AGENT_COLORS[agent_name]
        plan_response = self._call_llm(
            agent_name,
            self.PLANNER_SYSTEM_MSG,
            current_goal_or_refinement_prompt,
            temperature=max(0.2, self.api.default_options.get('temperature', 0.6) - 0.2),
            max_tokens=self.api.default_options.get('num_predict', 4096) // 2
        )
        if plan_response:
            self.current_plan = plan_response
            print(printColoured("📝 Plan Generated:", color))
            print(self.current_plan)
            self.run_context += f"\n---\nGenerated Plan:\n{self.current_plan}\n---\n"
            
            # Extract requirements after generating the plan
            self._extract_requirements()
            
            return True
        else:
            print(printColoured(f"❌ Failed to generate plan.", "red"))
            return False

    def supervisor_planner(self, refinement_context: str) -> bool:
        """Acts as the supervisor, deciding the next step after a failed test."""
        agent_name = "SUPERVISOR_PLANNER"
        color = AGENT_COLORS["PLANNER"] # Use Planner color
        
        print(printColoured(f"\n🤔 Calling {agent_name} Agent to review failure...", color))

        prompt = f"""ORIGINAL GOAL:
{self.goal}

PREVIOUS PLAN:
{self.current_plan}

FAILED CODE (VERSION {self.current_version}):
```python
{self.current_code}
```

ERROR / TEST OUTPUT:
{self.last_error}

REFINER ANALYSIS & SUGGESTIONS:
{refinement_context}

Based on all the above, decide the next course of action and, if needed, provide a REVISED PLAN for the CODER.
"""

        supervisor_response = self._call_llm(
            agent_name,
            self.SUPERVISOR_SYSTEM_MSG,
            prompt,
            temperature=max(0.3, self.api.default_options.get('temperature', 0.6) - 0.1),
            max_tokens=self.api.default_options.get('num_predict', 4096) // 2
        )

        if supervisor_response:
            # The supervisor response *is* the new plan or decision
            self.current_plan = supervisor_response 
            print(printColoured("📝 Supervisor Decision / Revised Plan:", color))
            print(self.current_plan)
            self.run_context += f"\n---\nSupervisor Plan (v{self.current_version}):\n{self.current_plan}\n---\n"
            
            # IMPORTANT: Extract requirements based on the *new* plan from the supervisor
            self._extract_requirements()
            
            # Check if supervisor decided the issue is unresolvable (simple check)
            if "unresolvable" in supervisor_response.lower() or "cannot fix" in supervisor_response.lower():
                print(printColoured(" Supervisors deems the issue unresolvable.", "red"))
                self.last_error = f"Supervisor: Issue deemed unresolvable.\n{supervisor_response}"
                return False # Indicate failure to proceed
                
            return True # Proceed with the new plan
        else:
            print(printColoured(f"❌ Failed to get decision from Supervisor Planner.", "red"))
            self.last_error = "Supervisor Planner Error: No response."
            return False

    def coder(self, coder_retries: int = 2) -> bool:
        """Generates code based on the current plan with internal retries."""
        if not self.current_plan:
            logger.error("Cannot generate code without a plan.")
            self.last_error = "Coding Error: No plan available."
            return False

        agent_name = "CODER"
        color = AGENT_COLORS[agent_name]

        for attempt in range(coder_retries):
            print(printColoured(f"-- Coder Attempt {attempt + 1}/{coder_retries} --", color))
            # Context includes the latest plan and potentially the goal/history for broader context
            prompt = f"""Context:\n{self.run_context}\n\nCURRENT PLAN:\n{self.current_plan}\n\n
Write the COMPLETE Python code based *only* on the CURRENT PLAN. 
Put ALL code in a single \`\`\`python ... \`\`\` block.
Your code must be a FULLY FUNCTIONAL implementation.
IMPORTANT: Return the ENTIRE code, not just a utility or helper function.
IMPORTANT: DO NOT write code that tries to modify requirements.txt files or other non-functional snippets.
IMPORTANT: If a package cannot be installed, try to use alternative packages or standard library modules.
"""

            code_response = self._call_llm(
                agent_name,
                self.CODER_SYSTEM_MSG,
                prompt,
                temperature=self.api.default_options.get('temperature', 0.6),
                max_tokens=self.api.default_options.get('num_predict', 4096)
            )

            if not code_response:
                print(printColoured(f"❌ Failed to get response from Coder Agent (Attempt {attempt + 1}).", "red"))
                continue # Retry if LLM call failed

            extracted_codes = utils.extract_code(code_response)
            if extracted_codes:
                potential_code = extracted_codes[0]
                # Basic Syntax Check
                try:
                    compile(potential_code, '<string>', 'exec')
                    self.current_code = potential_code
                    print(printColoured("💻 Code Generated and Syntax OK (showing snippet):", color))
                    print(self.current_code[:200] + "...") # Show snippet
                    self.run_context += f"\n---\nGenerated Code (snippet):\n{self.current_code[:200]}...\n---\n"
                    return True # Success
                except SyntaxError as e:
                    logger.warning(f"Generated code failed syntax check (Attempt {attempt + 1}): {e}")
                    self._add_history("Coder Syntax Error", str(e), "yellow")
                    # Continue to retry
            else:
                logger.warning(f"No code block found in the Coder response (Attempt {attempt + 1}).")
                print(printColoured("❌ Coder Response (no code block found):", color))
                print(code_response) # Print response for debugging
                # Continue to retry

        # If loop finishes without success
        logger.error(f"Coder failed to produce valid code after {coder_retries} attempts.")
        self.current_code = ""
        self.last_error = "Coding Error: Failed to produce valid code after retries."
        self._add_history("Coder Final Error", self.last_error, "red")
        return False

    def namer(self) -> bool:
        """Suggests a filename for the main script and sets the project name."""
        agent_name = "NAMER"
        color = AGENT_COLORS[agent_name]

        context = self.current_plan if self.current_plan else self.goal
        # Ask for a suitable primary script filename
        prompt = f"CONTEXT:\n{context}\n\nSuggest a concise, descriptive, snake_case filename for the *main Python script* ending in '.py'. Respond ONLY with the filename."

        filename_response = self._call_llm(
            agent_name,
            self.NAMER_SYSTEM_MSG,
            prompt,
            temperature=0.2,
            max_tokens=50
        )

        if filename_response:
            # Basic cleaning
            filename = re.sub(r'[^\w\-_.]', '_', filename_response)
            filename = filename.replace('`', '').replace('"', '').replace("'", "")
            if not filename.endswith('.py'):
                filename += '.py'
            filename = filename.lower()
            self.current_filename = filename
            # Use the filename stem as the project name
            self.project_name = Path(self.current_filename).stem
            print(printColoured(f"🏷️ Suggested Filename: {self.current_filename}", color))
            print(printColoured(f"🏷️ Project Name: {self.project_name}", color))
            return True
        else:
            # Keep default filename and project name if suggestion fails
            self.project_name = Path(self.current_filename).stem
            print(printColoured(f"⚠️ Using default filename: {self.current_filename}", color))
            print(printColoured(f"⚠️ Using default project name: {self.project_name}", color))
            self._add_history("Namer Info", f"Failed to get suggestion, using defaults: {self.current_filename} / {self.project_name}", color)
            return False # Indicate failure but allow process to continue

    def saver(self) -> Tuple[bool, Optional[str]]:
        """Saves the current project state (code, requirements) to a versioned directory."""
        agent_name = "SAVER"
        color = AGENT_COLORS[agent_name]
        
        # Determine versioned project directory path
        versioned_project_dir = self.project_base_dir / f"{self.project_name}_v{self.current_version}"
        versioned_project_dir.mkdir(parents=True, exist_ok=True)
        
        print(printColoured(f"\n💾 Saving Project State to {versioned_project_dir}...", color))

        if not self.current_code:
            self.last_error = "Saving Error: No code available to save."
            print(printColoured(f"❌ {self.last_error}", "red"))
            self._add_history("Saver Error", self.last_error, "red")
            return False, None

        # Save the main code file
        code_file_path = versioned_project_dir / self.current_filename
        try:
            with open(code_file_path, "w", encoding="utf-8") as f:
                f.write(self.current_code)
            print(printColoured(f"   Code saved to: {code_file_path}", color))
            self._add_history(agent_name, f"Code file saved to {code_file_path}", color)
        except Exception as e:
            self.last_error = f"Saving Error: Failed to save code file '{code_file_path}': {e}"
            print(printColoured(f"❌ {self.last_error}", "red"))
            self._add_history("Saver Error", self.last_error, "red")
            return False, None
        
        # Save requirements.txt for this version
        req_file_path = versioned_project_dir / "requirements.txt"
        try:
            req_content = "\n".join(self.requirements)
            with open(req_file_path, "w", encoding="utf-8") as f:
                f.write(req_content)
            if self.requirements:
                print(printColoured(f"   Requirements saved to: {req_file_path}", color))
                self._add_history(agent_name, f"Requirements saved to {req_file_path}", color)
            else:
                print(printColoured(f"   Requirements file (empty) saved to: {req_file_path}", color))
                self._add_history(agent_name, f"Empty requirements saved to {req_file_path}", color)
        except Exception as e:
             print(printColoured(f"   Warning: Failed to save requirements.txt to '{req_file_path}': {e}", "yellow"))
             # Continue even if reqs saving fails, but log it
             self._add_history("Saver Warning", f"Failed to save {req_file_path}: {e}", "yellow")


        print(printColoured(f"✅ Project state saved to: {versioned_project_dir}", color))
        return True, str(versioned_project_dir)

    def tester(self, project_dir: str) -> bool:
        """Tests the main script within the specified project directory using a venv sandbox."""
        agent_name = "TESTER"
        color = AGENT_COLORS[agent_name]
        project_path = Path(project_dir)
        main_script_path = project_path / self.current_filename
        requirements_path = project_path / "requirements.txt"

        print(printColoured(f"\n🧪 Testing Code in Project Sandbox: {project_path}...", color))

        if not main_script_path.is_file():
            self.last_error = f"Testing Error: Main script '{main_script_path}' not found in project directory."
            print(printColoured(f"❌ {self.last_error}", "red"))
            self._add_history("Tester Error", self.last_error, "red")
            return False

        # Read requirements from the project directory's requirements.txt
        current_project_reqs = []
        if requirements_path.is_file():
            try:
                with open(requirements_path, "r", encoding="utf-8") as f:
                    current_project_reqs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                logger.info(f"Read {len(current_project_reqs)} requirements from {requirements_path}")
            except Exception as e:
                logger.warning(f"Could not read requirements file {requirements_path}: {e}")
                # Continue without requirements if file read fails
        
        # Ensure requirements list passed to test_code_in_venv is correct (could be empty)
        reqs_to_pass = current_project_reqs

        # Execute in venv - test_code_in_venv now needs adaptation
        # For now, let's assume utils.test_code_in_venv can handle a project dir context
        # We might need to modify test_code_in_venv itself later.
        success, output = utils.test_code_in_venv(
            project_dir=str(project_path), # Pass project dir 
            main_script_name=self.current_filename, # Pass main script name
            requirements=reqs_to_pass # Pass requirements read from file
            # code_content=self.current_code, # No longer needed if test_code reads from file
            # base_filename=self.current_filename, # No longer needed
            # version=self.current_version # No longer needed
        )
        
        # Fallback logic (testing without requirements) might need adjustment
        # if testing inherently uses the requirements.txt inside the venv setup.
        # For now, let's assume the initial test handles it.
        # TODO: Re-evaluate fallback logic if test_code_in_venv changes significantly.
        
        self.test_results = output
        self._add_history(agent_name, {"success": success, "output": output}, color)

        if success:
            print(printColoured("✅ Code Test Successful!", color))
            print(printColoured("Output:", "yellow"))
            print(output)
            self.last_error = "" # Clear last error on success
            return True
        else:
            print(printColoured("❌ Code Test Failed.", color))
            print(printColoured("Error/Output:", "yellow"))
            print(output)
            self.last_error = f"Testing Error:\n{output}" # Store the specific error
            # Update long-term context with the error
            self.run_context += f"\n---\nTesting Failed:\n{self.last_error}\n---\n"
            return False

    def refiner(self) -> bool:
        """Analyzes the error and generates a refined plan/instructions."""
        agent_name = "REFINER"
        color = AGENT_COLORS[agent_name]

        if not self.last_error:
            logger.warning("Refiner called without a preceding error. Skipping.")
            return False # Cannot refine without an error

        print(printColoured("\n🤔 Analyzing Error and Refining Plan...", color))

        # Construct prompt with all relevant context for refinement
        prompt = f"""ORIGINAL GOAL:
{self.goal}

PREVIOUS PLAN:
{self.current_plan}

FAILED CODE:
```python
{self.current_code}
```

ERROR / TEST OUTPUT:
{self.last_error}

Analyze the error in the context of the plan and code. Provide step-by-step reasoning.
Then, generate a REVISED PLAN or specific, clear INSTRUCTIONS for the CODER to fix the issue.

IMPORTANT: If there's a problem with external dependencies like 'os' or 'tkinter', note that these are built-in Python modules and should NOT be added to requirements. Focus on making the code work with standard library or alternative packages.

IMPORTANT: Ensure your plan makes the CODER return a COMPLETE implementation, not just utility functions or partial fixes.

Focus ONLY on the necessary changes. Respond ONLY with your analysis and revised plan/instructions.
"""
        refinement_response = self._call_llm(
            agent_name,
            self.REFINER_SYSTEM_MSG,
            prompt,
             # Use a moderate temperature for creative problem solving but still grounded
            temperature=max(0.4, self.api.default_options.get('temperature', 0.6)),
            max_tokens=self.api.default_options.get('num_predict', 4096) // 2
        )

        if refinement_response:
            self.current_plan = refinement_response # Refinement output becomes the new plan
            print(printColoured("📝 Refinement/Revised Plan Generated:", color))
            print(self.current_plan)
            self.run_context += f"\n---\nRefinement Analysis & Plan:\n{self.current_plan}\n---\n"
            self.last_error = "" # Clear error after successful refinement

            # Filter out built-in modules
            added_packages = re.findall(r"Add '(\S+?)' to the required packages list", refinement_response, re.IGNORECASE)
            # Further refine: Use suggestions pattern instead
            added_packages = re.findall(r"Suggestion: Add '(\S+?)' to the required packages list", refinement_response, re.IGNORECASE)
            if added_packages:
                newly_added = []
                for pkg in added_packages:
                    clean_pkg = pkg.strip("'\"") # Clean quotes
                    # Filter out built-in modules using the constant
                    if clean_pkg and clean_pkg.lower() not in [m.lower() for m in BUILT_IN_MODULES]:
                        # DEPRECATED: Requirements are now handled by the supervisor planner
                        # self.requirements.append(clean_pkg)
                        # newly_added.append(clean_pkg)
                        logger.info(f"Refiner suggested adding requirement: {clean_pkg} (will be handled by supervisor)")
                        pass # Let supervisor handle it based on revised plan
                if newly_added: # This block likely won't run now, keep for potential future logic
                    logger.info(f"Refiner added requirements (DEPRECATED LOGIC): {newly_added}")
                    # self._add_history("Requirements Updated by Refiner", newly_added)
            
            return True
        else:
            print(printColoured(f"❌ Failed to generate refinement analysis.", "red"))
            # Keep the last error if refinement fails
            return False

    def run_factory(self):
        """
        Executes the full Plan -> Code -> Name -> Save -> Test -> [Refine -> Code -> Save -> Test]... pipeline.
        """
        print(printColoured(f"\n🚀🚀🚀 Starting Autonomous Factory Run 🚀🚀🚀", AGENT_COLORS["SYSTEM"]))
        self._add_history("SYSTEM", f"Starting run with goal: {self.goal}", AGENT_COLORS["SYSTEM"])

        # Initial Planning
        if not self.planner(f"GOAL: {self.goal}\n\nGenerate the initial plan."):
            print(printColoured("❌ Factory stopped during initial planning.", "red"))
            return False # Stop if initial planning fails

        # Suggest Filename (only needs to happen once, gets base name)
        self.namer() # Suggests self.current_filename

        for attempt in range(self.max_retries + 1): # +1 for the initial attempt
            self.current_version = attempt + 1 # Start versioning from v1
            print(printColoured(f"\n--- Iteration {self.current_version} / {self.max_retries + 1} ---", AGENT_COLORS["SYSTEM"]))

            # Coding based on the current plan (initial or revised)
            if not self.coder():
                print(printColoured(f"❌ Factory stopped during coding (Iteration {self.current_version}).", "red"))
                # Decide if we should try refining the *plan* or stop
                # For now, stopping if coder fails fundamentally.
                return False

            # Save the generated code (now includes version)
            save_success, saved_file_path = self.saver()
            if not save_success:
                print(printColoured(f"❌ Factory stopped during saving (Iteration {self.current_version}).", "red"))
                return False # Stop if saving fails

            # Test the generated code
            test_success = self.tester(saved_file_path)
            if test_success:
                print(printColoured(f"\n🎉🎉🎉 Factory Run Completed Successfully! 🎉🎉🎉", AGENT_COLORS["SYSTEM"]))
                return True # Exit the loop and function successfully

            # --- Refinement Stage (if test failed and retries remain) ---
            if attempt < self.max_retries:
                print(printColoured(f"\n--- Starting Refinement (Attempt {self.current_version}) ---", AGENT_COLORS["REFINER"]))
                
                # 1. Call Refiner (Critic/Debugger)                
                if not self.refiner():
                    print(printColoured(f"❌ Factory stopped: Failed to generate refinement analysis (Iteration {self.current_version}).", "red"))
                    return False # Stop if refinement analysis fails
                
                # 2. Call Planner (Supervisor) to decide next step and potentially revise plan
                if not self.supervisor_planner(self.refinement_analysis):
                     print(printColoured(f"❌ Factory stopped: Supervisor Planner failed or deemed issue unresolvable (Iteration {self.current_version}).", "red"))
                     return False # Stop if supervisor fails or decides to stop

                # 3. Loop continues to coder with the new plan (requirements extracted by supervisor_planner)
                # Requirements are extracted within supervisor_planner based on its output
            else:
                # Max retries reached
                print(printColoured(f"\n🚫 Maximum retries ({self.max_retries}) reached. Code still failing.", AGENT_COLORS["SYSTEM"]))
                print(printColoured(f"Final Error:\n{self.last_error}", "red"))
                return False # Indicate failure after max retries

        # Should not be reached if logic is correct, but acts as a fallback
        print(printColoured("\n🏁 Factory run finished (reached end unexpectedly).", AGENT_COLORS["SYSTEM"]))
        return False


if __name__ == "__main__":
    print(printColoured("===== Ollama Autonomous Code Factory =====", "blue"))

    # Default goal if user doesn't provide one
    default_goal = "Create a simple python script that uses tk ui and loads in a list of images browsed on either mac/windows and generates an video mp4 file with options for fps, quality, and saves it with a given name in the same location as the image frames."
    
    # Print the prompt and then get input
    print(printColoured("Enter the goal for the code factory (press Enter for default goal): ", "cyan"), end="")
    project_goal = input().strip()
    
    # Use default if no input provided
    if not project_goal:
        project_goal = default_goal
        print(printColoured(f"Using default goal: {project_goal}", "yellow"))
    
    if project_goal:
        # Ask for verbose mode
        print(printColoured("Enable verbose streaming mode? (y/n): ", "cyan"), end="")
        verbose_mode = input().strip().lower() == 'y'
        
        # Choose model
        print(printColoured("Enter model name (press Enter for gemma3:12b): ", "cyan"), end="")
        model_name = input().strip() or "gemma3:12b"
        
        # Example: Use a capable coding model
        factory = FactoryOllama(goal=project_goal, model=model_name, max_retries=2, verbose=verbose_mode)
        run_successful = factory.run_factory()

        print(printColoured("\n===== Factory Run Summary =====", "blue"))
        if run_successful:
             print(printColoured("✅ Overall Status: Success", "green"))
        else:
             print(printColoured("❌ Overall Status: Failed", "red"))
             if factory.last_error:
                 print(printColoured(f"Last Error:\n{factory.last_error}", "red"))

        print(printColoured("\n===== Full Run History =====", "blue"))
        for i, (agent, output) in enumerate(factory.run_history):
             color = AGENT_COLORS.get(agent.split(" ")[0].upper(), "grey") # Get color based on agent name
             print(printColoured(f"{i+1}. [{agent.upper()}]", color))
             # Print snippet of output for brevity
             print(f"   Output: {str(output)[:300]}{'...' if len(str(output)) > 300 else ''}\n")

    else:
        print(printColoured("No goal provided. Exiting.", "yellow"))
