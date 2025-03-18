# Empty file to make the directory a Python package 

# Package export for the III_agents module

try:
    from .openai_agents import OpenAIAgentsAPI
    __all__ = ["OpenAIAgentsAPI"]
except ImportError:
    # If the OpenAI Agents SDK is not installed, we still want
    # the package to be importable, but using the class will 
    # raise appropriate errors
    __all__ = [] 