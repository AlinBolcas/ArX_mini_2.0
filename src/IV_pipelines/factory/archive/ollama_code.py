#!/usr/bin/env python
from pathlib import Path
import sys
import re
import os
from typing import Optional, List, Dict, Any, Iterator, Union

# Fix module imports by adding project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.I_integrations.ollama_API import OllamaWrapper
from src.VI_utils.utils import printColoured

class OllamaCodeEditor:
    """Efficient code editing system using Ollama."""
    
    def __init__(
        self,
        model: str = "codellama:latest",
        system_message: str = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        auto_pull: bool = True
    ):
        """Initialize the code editor with Ollama integration."""
        
        # Default system message if none provided
        if system_message is None:
            system_message = """
            You are an expert Python code editor. Your task is to help modify, improve, and debug Python code files.
            
            IMPORTANT INSTRUCTIONS:
            1. When modifying code, you MUST provide the COMPLETE FILE content, not just the changes.
            2. All code must be placed in Python code blocks like: ```python [code here] ```
            3. Your code must be runnable and complete - don't leave placeholders or TODOs.
            4. Include helpful comments to explain complex logic.
            5. Always maintain the original functionality while improving the code.
            6. The user will give you instructions on how to modify the code - follow them precisely.
            
            First provide a brief explanation of your changes, then the complete code implementation.
            """
        
        # Initialize Ollama API
        self.api = OllamaWrapper(
            model=model,
            auto_pull=auto_pull,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize message history
        self.message_history = []
        self.current_file = None
        self.current_file_content = None
    
    def read_file(self, file_path: str) -> str:
        """Read content from a file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return ""
    
    def write_file(self, file_path: str, content: str) -> bool:
        """Write content to a file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"Error writing to file {file_path}: {e}")
            return False
    
    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from markdown formatted text."""
        code_blocks = []
        pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
        matches = re.finditer(pattern, text)
        
        for match in matches:
            code_blocks.append(match.group(1).strip())
        
        return code_blocks
    
    def set_file(self, file_path: str) -> None:
        """Set the current file to edit and load its content."""
        self.current_file = file_path
        self.current_file_content = self.read_file(file_path)
        
        # Add file context to message history with clear instructions
        if self.current_file_content:
            self.message_history = [
                {"role": "system", "content": self.api.system_message},
                {"role": "user", "content": f"I'm working on this Python file: '{file_path}'. Here is the current content:\n\n```python\n{self.current_file_content}\n```\n\nI'll give you instructions on how to modify this file. Always respond with the complete updated file content inside a Python code block."}
            ]
        else:
            self.message_history = [
                {"role": "system", "content": self.api.system_message},
                {"role": "user", "content": f"I'm creating a new Python file: '{file_path}'. Please help me implement it based on my instructions. Always respond with the complete file content inside a Python code block."}
            ]
    
    def get_suggested_filename(self, description: str) -> str:
        """Ask the LLM to suggest a suitable filename based on the description."""
        print(printColoured("\nGenerating appropriate filename...", "magenta"))
        
        prompt = f"""Based on this description: "{description}", 
        suggest a short, clear Python filename (including .py extension).
        The filename should be descriptive, use snake_case, and be under 30 characters.
        Respond with ONLY the filename, nothing else."""
        
        result = self.api.chat_completion(
            user_prompt=prompt,
            stream=False,
            message_history=[],
            max_tokens=50,
            temperature=0.3  # Lower temperature for more predictable naming
        )
        
        # Clean the result to ensure it's just a filename
        filename = result.strip().replace('`', '').replace('"', '').replace("'", "")
        
        # Ensure it has .py extension
        if not filename.endswith('.py'):
            filename += '.py'
            
        # Sanitize the filename (remove special characters, spaces)
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        print(printColoured(f"\nSuggested filename: {filename}", "green"))
        return filename
    
    def edit_loop(self, file_path: Optional[str] = None) -> None:
        """Run an interactive editing loop for the specified file."""
        if file_path:
            self.set_file(file_path)
        
        if not self.current_file:
            file_path = input("Enter the path to the file you want to edit: ")
            self.set_file(file_path)
            
        print(printColoured(f"\nEditing file: {self.current_file}", "magenta"))
        if self.current_file_content:
            print(printColoured("\nCurrent file content:", "yellow"))
            print(self.current_file_content)
        else:
            print(printColoured("\nThis is a new file.", "yellow"))
        
        # Main editing loop
        while True:
            user_input = input(printColoured("\nWhat changes would you like to make? (type 'exit' to quit): ", "cyan"))
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            # Check for @new command to create a new file based on description
            new_file_match = re.search(r'@new\s+(.*)', user_input)
            if new_file_match:
                description = new_file_match.group(1).strip()
                if not description:
                    print(printColoured("\nPlease provide a description after @new", "red"))
                    continue
                    
                # Get LLM to suggest a filename based on the description
                suggested_filename = self.get_suggested_filename(description)
                
                # Add path to the filename - default to data/output/code/
                dir_path = os.path.dirname(self.current_file) if self.current_file else "data/output/code"
                new_file_path = os.path.join(dir_path, suggested_filename)
                
                # Confirm with user
                confirm = input(printColoured(f"\nCreate new file at {new_file_path}? (y/n): ", "magenta"))
                if confirm.lower() != 'y':
                    print(printColoured("\nFile creation cancelled.", "yellow"))
                    continue
                
                print(printColoured(f"\nCreating new file: {new_file_path}", "yellow"))
                
                # Set up the new file messaging context
                self.current_file = new_file_path
                self.current_file_content = ""
                
                # Initialize message history for the new file
                self.message_history = [
                    {"role": "system", "content": self.api.system_message},
                    {"role": "user", "content": f"I'm creating a new Python file: '{new_file_path}' to {description}. Please provide a complete implementation with proper comments and documentation. Return the code in a Python code block."}
                ]
                
                # Generate code based on the description
                print(printColoured("\nGenerating initial code...", "magenta"))
                
                response_text = ""
                # Direct API call with proper message history for consistent context
                stream = self.api.chat_completion(
                    user_prompt="",  # Empty because we already set it in message_history
                    stream=True,
                    message_history=self.message_history,
                    max_tokens=4096,
                    temperature=0.7
                )
                
                # Display streaming response
                print(printColoured("\nResponse:", "magenta"))
                for chunk in stream:
                    print(printColoured(chunk, "blue"), end="", flush=True)
                    response_text += chunk
                
                # Add assistant response to history
                self.message_history.append({"role": "assistant", "content": response_text})
                
                # Extract code blocks
                code_blocks = self.extract_code_blocks(response_text)
                
                if code_blocks:
                    new_content = code_blocks[-1]
                    # Save the generated code automatically
                    if self.write_file(self.current_file, new_content):
                        print(printColoured(f"\n✅ Code successfully generated and saved to {self.current_file}", "green"))
                        self.current_file_content = new_content
                    else:
                        print(printColoured("\n❌ Failed to save code", "red"))
                else:
                    print(printColoured("\n⚠️ No code blocks found in the response. Creating empty file instead.", "yellow"))
                    self.write_file(self.current_file, "")
                
                continue  # Start fresh with the new file
            
            # Check for @[FILE_NAME] pattern
            file_name_match = re.search(r'@(\S+)', user_input)
            if file_name_match:
                file_name = file_name_match.group(1)
                file_content = self.read_file(file_name)
                if file_content:
                    user_input = user_input.replace(f'@{file_name}', f"\n\nHere is the content of {file_name}:\n\n```python\n{file_content}\n```\n")
                else:
                    print(printColoured(f"\nFile {file_name} not found or empty.", "red"))
                    continue
            
            # Add user instruction to history with clear formatting expectations
            instruction = f"Please modify the code according to these instructions: {user_input}\n\nProvide the COMPLETE updated file content in a Python code block."
            self.message_history.append({"role": "user", "content": instruction})
            
            # Get response from Ollama
            print(printColoured("\nThinking...", "magenta"))
            result = ""
            stream = self.api.chat_completion(
                user_prompt=instruction,
                stream=True,
                message_history=self.message_history[:-1],  # Exclude the last message we just added
                max_tokens=4096,
                temperature=0.6
            )
            
            # Display streaming response
            print(printColoured("\nResponse:", "magenta"))
            for chunk in stream:
                print(printColoured(chunk, "blue"), end="", flush=True)
                result += chunk
            
            # Add assistant response to history
            self.message_history.append({"role": "assistant", "content": result})
            
            # Extract code blocks with more robust pattern matching
            code_blocks = self.extract_code_blocks(result)
            
            if code_blocks:
                # Use the last code block as the new content
                new_content = code_blocks[-1]
                
                # Simple approval process
                print(printColoured("\n\nProposed new code:", "magenta"))
                print(printColoured(new_content, "cyan"))
                
                # Ask for confirmation
                confirm = input(printColoured("\nApply these changes? (y/n): ", "magenta"))
                if confirm.lower() == 'y':
                    # Save changes
                    if self.write_file(self.current_file, new_content):
                        print(printColoured(f"\nChanges saved to {self.current_file}", "green"))
                        self.current_file_content = new_content
                    else:
                        print(printColoured("\nFailed to save changes", "red"))
                else:
                    print(printColoured("\nChanges discarded", "yellow"))
            else:
                print(printColoured("\nNo code blocks found in the response. Please try again with different instructions.", "yellow"))

def main():
    """Main function to run the Ollama Code Editor with hardcoded settings."""
    print(printColoured("🚀 Ollama Code Editor", "cyan"))
    print(printColoured("===================", "cyan"))
    
    GOAL = ""
    # Hardcoded settings - ensure default file is in the correct directory
    target_file = "data/output/code/generated_code.py" # Changed path and filename
    model = "llama3.2-vision:11b" 
    temperature = 0.65
    
    # Initialize editor
    editor = OllamaCodeEditor(
        model=model,
        temperature=temperature,
        auto_pull=True
    )
    
    # Empty file by default for wip_code.py
    if not os.path.exists(target_file):
        print(printColoured(f"\nEnsuring directory exists and creating empty {target_file} file...", "yellow"))
        # Ensure the directory exists before writing
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, "w") as f:
            f.write("")
    
    # Start editing loop with hardcoded file
    editor.edit_loop(target_file)
    
    print(printColoured("\nThank you for using Ollama Code Editor!", "cyan"))

if __name__ == "__main__":
    main() 