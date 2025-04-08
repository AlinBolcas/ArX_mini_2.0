from pathlib import Path
import sys
import re
import os

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.I_integrations.ollama_API import OllamaWrapper
from src.VI_utils.utils import printColoured

sys_message = """
You are an expert coder, assist with all user requests with effciently written code. 
Always response in markdown python block.
"""

def get_code(code_path=None):
    # Get the path of the current script
    if code_path == None:
        code_path = __file__
    try:
        with open(code_path, 'r') as f:
            source_code = f.read()
        
        return source_code
    
    except FileNotFoundError:
        print("Error: Script not found.")
        return None

def save_response_as_code(response):
    import re

    # Fix module imports by adding project root to path
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    code_blocks = []
    start_index = 0
    while True:
        start_index = response.find("```python", start_index)
        if start_index == -1:
            break
        end_index = response.find("```", start_index + len("```python"))
        if end_index == -1:
            code_blocks.append(response[start_index:])
            break
        # Include the backticks in the extracted block for clarity
        code_block = response[start_index:end_index + 3]
        code_blocks.append(code_block)
        start_index = end_index + 3

    # Extract and concatenate Python code blocks, preserving indentation
    python_code = ""
    for block in code_blocks:
        lines = block.splitlines()
        if len(lines) > 1:  # Ensure there are at least two lines to include the backticks
            content_lines = lines[1:-1]  # Skip the initial and final backtick markers
            stripped_content = "\n".join(content_lines)
            python_code += f"{stripped_content}\n\n"

    # Save the Python code to a file named 'code.py' in the correct location
    path_to_code = "data/output/code/generated_code.py"
    
    # Ensure the directory exists before writing
    os.makedirs(os.path.dirname(path_to_code), exist_ok=True) 
    
    with open(path_to_code, "w") as f:
        f.write(python_code.strip())

    return path_to_code

if __name__ == "__main__":

    this_code = get_code()
    memory_code = get_code("src/II_textGen/memory.py")
    rag_code = get_code("src/II_textGen/rag.py")
    ollama_code = get_code("src/I_integrations/ollama_API.py")
    
    # Initialize API
    api = OllamaWrapper(
        model="llama3.2-vision:11b", #deepseek-r1:8b , llama3.2-vision:11b , deepseek-coder:latest , codellama:code, phi4:latest, gemma3:4b
        auto_pull=True,
        system_message=sys_message
    )
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "bye":
            break
        
        prompt = f"""Can you give me an idea or the snippet of how I can create a true file search based on the name when I tag a file name as @file_name:code.py, to actually search for it in the repo than for me having to import it. Here's the full code context: {ollama_code}"""
        
        
        result = ""
        stream = api.chat_completion(
        user_prompt=prompt,
        stream = True,
        # max_tokens=800,
        # temperature=0.6
        )
        print(printColoured("Response:", "magenta"))
        for chunk in stream:
            print(printColoured(chunk,"blue"), end="", flush=True)
            result += chunk

        # Extract Python code blocks from the response string
        code_path = save_response_as_code(result)
        print(printColoured(f"\n\nCode block saved to {code_path}","yellow"))
        

