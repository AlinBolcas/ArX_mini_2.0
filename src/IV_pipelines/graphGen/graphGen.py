"""
Graph Generation Utility for creating visualizations using Graphviz.
Produces system architecture diagrams with transparent backgrounds and clear structure.
"""
import os
import tempfile
import graphviz
import logging
import json
import re
import time
import datetime
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

# Import OpenAI Responses API
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
try:
    from src.I_integrations.openai_responses_API import OpenAIResponsesAPI
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from I_integrations.openai_responses_API import OpenAIResponsesAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphGen:
    """
    Graph generation utility using Graphviz for high-quality visualizations.
    Specializes in system architecture diagrams with consistent styling.
    """
    
    def __init__(self):
        """Initialize the graph generator with Arvolve-inspired blue-integrated color scheme."""
        # Blue-based color scheme with accents for different component types
        self.colors = {
            'background': 'transparent',     # Transparent background
            'central': '#00BFFF',            # Bright blue for central node
            'primary_blue': '#3B82F6',       # Medium blue for primary connections (API/TextGen)
            'secondary_blue': '#1E3A8A',     # Deep blue for secondary nodes
            'purple': '#9333EA',             # Purple for tools/knowledge diamonds
            'deep_purple': '#6D28D9',        # Deep purple for UI components
            'bright_green': '#10B981',       # Bright green for internal services
            'orange': '#F97316',             # Orange for external services and inputs
            'red': '#EF4444',                # Red for output
            'edge': '#AAAAAA80',             # Light gray semi-transparent for edges
            'edge_primary': '#00BFFF',       # Bright blue for important edges
            'text': '#ffffff'                # White text
        }
        
    def generate_graph(self, input_data: Union[Dict, str], output_path: str) -> Optional[str]:
        """
        Generate a graph visualization using Graphviz.
        
        Args:
            input_data: Data defining the graph structure
            output_path: Where to save the final image (without extension)
            
        Returns:
            Path to the generated image or None if generation failed
        """
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                
            # Generate the graph visualization
            return self._generate_graphviz(input_data, output_path)
        
        except Exception as e:
            logger.error(f"Failed to generate graph: {e}")
            return None
    
    def _generate_graphviz(self, input_data: Union[Dict, str], output_path: str) -> Optional[str]:
        """
        Generate a graph visualization using Graphviz.
        
        Args:
            input_data: Dictionary with nodes and edges or DOT language string
            output_path: Where to save the final image (without extension)
            
        Returns:
            Path to the generated PNG
        """
        try:
            # Handle string input as direct DOT language
            if isinstance(input_data, str):
                # Create a temporary DOT file
                with tempfile.NamedTemporaryFile('w', suffix='.dot', delete=False) as f:
                    f.write(input_data)
                    dot_path = f.name
                
                # Use graphviz command-line to render with transparency
                output_png = f"{output_path}.png"
                cmd = f"dot -Tpng -Gbgcolor=transparent -o {output_png} {dot_path}"
                logger.info(f"Running command: {cmd}")
                os.system(cmd)
                os.unlink(dot_path)  # Clean up
                
                if os.path.exists(output_png):
                    logger.info(f"Graphviz visualization saved to: {output_png}")
                    return output_png
                else:
                    logger.error("Failed to generate Graphviz PNG")
                    return None
            
            # Handle dictionary input (standard format)
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary with 'nodes' and 'edges' keys, or a DOT language string")
            
            # Create directed graph (for arrows)
            dot = graphviz.Digraph('graph', comment='System Architecture', format='png')
            
            # Set graph attributes for visibility and transparency
            graph_attrs = input_data.get('graph_attrs', {})
            
            # Core graph attributes
            dot.attr(
                fontname='Helvetica,Arial,sans-serif',
                fontcolor=self.colors['text'],
                fontsize='14',
                rankdir='TB',  # Top to bottom layout
                bgcolor='transparent',  # Transparent background
                dpi='300',
                splines='true',  # Curved edges
                overlap='false',
                concentrate='false', # Don't merge edges (better visibility)
                nodesep='0.7',
                ranksep='1.2'
            )
            
            # Node defaults - more prominent nodes
            dot.attr('node',
                shape='box',
                style='filled,rounded',
                fillcolor=self.colors['secondary_blue'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='12',
                margin='0.2',
                height='0.6',
                width='0.8',
                penwidth='1.2'  # Slightly thicker borders for visibility
            )
            
            # Edge defaults - make edges much more visible
            dot.attr('edge',
                color=self.colors['edge_primary'],  # Brighter edges by default
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='10',
                penwidth='1.5',   # Thicker edges for better visibility
                arrowsize='0.9'   # Larger arrow heads
            )
            
            # Create ranks based on the architecture tiers
            ranks = {
                "api_tier": ["api_integrations"],
                "service_tier": ["openai", "replicate", "tripo", "web_crawling"],
                "middle_tier": ["promptgen", "tools"],
                "processing_tier": ["textgen", "utils", "librarian", "rag_context"],
                "knowledge_tier": ["knowledge"],
                "memory_tier": ["short_term", "long_term"],
                # UI tier removed to allow custom horizontal layout
            }
            
            # Create invisible constraints to position nodes in their tiers
            for rank_name, nodes in ranks.items():
                with dot.subgraph(name=f"cluster_{rank_name}") as s:
                    s.attr(rank="same")
                    for node in nodes:
                        s.node(node)
            
            # Create a special horizontal layout for UI flow
            with dot.subgraph(name="cluster_ui_horizontal") as s:
                s.attr(rank="same")
                s.attr(rankdir="LR")  # Force left-to-right inside this subgraph
                s.node("user_input")
                s.node("ui_platform")
                s.node("output")
                # Add invisible edges to enforce ordering
                s.edge("user_input", "ui_platform", style="invis", weight="10")
                s.edge("ui_platform", "output", style="invis", weight="10")
            
            # Add nodes with appropriate colors and shapes
            for node in input_data.get('nodes', []):
                node_id = node.get('id')
                if not node_id:
                    continue
                    
                label = node.get('label', node_id).replace('\\n', '\n')  # Handle line breaks
                shape = node.get('shape', 'box').lower()
                
                # Match the colors based on node type and shape
                attrs = {
                    'label': label,
                    'shape': shape,
                    'style': 'filled,rounded'
                }
                
                # Assign colors based on shape and node ID to match the example image
                if shape == 'diamond':
                    if node_id in ['api_integrations', 'textgen']:
                        attrs['fillcolor'] = self.colors['primary_blue']
                    else:  # tools, knowledge
                        attrs['fillcolor'] = self.colors['purple']
                elif shape == 'circle':
                    if node_id in ['promptgen', 'rag_context', 'librarian']:
                        attrs['fillcolor'] = self.colors['bright_green']
                    else:  # openai, replicate, tripo, web_crawling, user_input
                        attrs['fillcolor'] = self.colors['orange']
                elif shape == 'box':
                    if node_id == 'output':
                        attrs['fillcolor'] = self.colors['red']
                    else:  # utils, ui_platform, short_term, long_term
                        attrs['fillcolor'] = self.colors['deep_purple']
                
                # Add the node with its attributes
                dot.node(node_id, **attrs)
            
            # Add edges with proper direction and visibility
            for edge in input_data.get('edges', []):
                source = edge.get('from')
                target = edge.get('to')
                if not source or not target:
                    continue
                    
                # Assign edge color and weight based on the connection type
                edge_attrs = {}
                
                # Main flow is brighter
                if (source == 'api_integrations' and target in ['tools', 'openai']) or \
                   (source == 'openai' and target in ['promptgen', 'textgen', 'librarian']) or \
                   (source == 'tools' and target == 'textgen') or \
                   (source == 'textgen' and target in ['knowledge', 'ui_platform']) or \
                   (source == 'knowledge' and target in ['short_term', 'long_term']) or \
                   (source == 'ui_platform' and target == 'output'):
                    edge_attrs['color'] = self.colors['edge_primary']
                    edge_attrs['penwidth'] = '1.8'
                else:
                    edge_attrs['color'] = self.colors['edge'] 
                    edge_attrs['penwidth'] = '1.2'
                
                # Add the edge
                dot.edge(source, target, **edge_attrs)
            
            # Generate the PNG directly
            output_png = f"{output_path}.png"
            
            # First render to a dot file
            dot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dot')
            dot_file.close()
            dot.save(dot_file.name)
            
            # Then use the command-line tool for better transparency support
            cmd = f"dot -Tpng -Gbgcolor=transparent {dot_file.name} -o {output_png}"
            logger.info(f"Running command: {cmd}")
            result = os.system(cmd)
            
            # Clean up the temporary file
            os.unlink(dot_file.name)
            
            if result == 0 and os.path.exists(output_png):
                logger.info(f"Graphviz visualization saved to: {output_png}")
                return output_png
            else:
                logger.error("Failed to generate Graphviz PNG")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate Graphviz visualization: {e}")
            return None

    def _generate_flat_graphviz(self, input_data: Dict, output_path: str) -> Optional[str]:
        """
        Generate a flat graph visualization using Graphviz 'dot' engine.
        
        Args:
            input_data: Dictionary with nodes and edges
            output_path: Where to save the final image (without extension)
            
        Returns:
            Path to the generated PNG
        """
        try:
            # Create directed graph with dot engine (hierarchical layout)
            dot = graphviz.Digraph('flat_graph', comment='Flat System Architecture', format='png', engine='dot')
            
            # Set graph attributes for visibility and transparency
            dot.attr(
                fontname='Helvetica,Arial,sans-serif',
                fontcolor=self.colors['text'],
                fontsize='14',
                rankdir='LR',  # Left to right layout for flat representation
                bgcolor='transparent',
                dpi='300',
                splines='spline',  # Curved splines for smooth rounded arrows
                overlap='false',
                nodesep='0.8',
                ranksep='1.5',
                concentrate='true',  # Concentrate edges for cleaner appearance
                sep='+15',  # More separation to avoid crowding
                esep='+5'   # Edge separation
            )
            
            # Node defaults
            dot.attr('node',
                shape='box',
                style='filled,rounded',
                fillcolor=self.colors['secondary_blue'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='12',
                margin='0.2',
                height='0.6',
                width='0.8',
                penwidth='1.0',
                fixedsize='false'  # Allow nodes to size properly for content
            )
            
            # Edge defaults - make edges visible but less prominent
            dot.attr('edge',
                color=self.colors['edge'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='10',
                penwidth='1.2',
                arrowsize='0.8',
                minlen='1.5',      # Minimum edge length
                arrowhead='normal',
                tailport='e',      # East port for output (right side)
                headport='w'       # West port for input (left side)
            )
            
            # Add nodes with appropriate colors and shapes
            for node in input_data.get('nodes', []):
                node_id = node.get('id')
                if not node_id:
                    continue
                    
                label = node.get('label', node_id).replace('\\n', '\n')
                shape = node.get('shape', 'box').lower()
                
                # Assign colors based on shape and node ID to match the example image
                attrs = {
                    'label': label,
                    'shape': shape,
                    'style': 'filled,rounded'
                }
                
                if shape == 'diamond':
                    if node_id in ['api_integrations', 'textgen']:
                        attrs['fillcolor'] = self.colors['primary_blue']
                    else:  # tools, knowledge
                        attrs['fillcolor'] = self.colors['purple']
                elif shape == 'circle':
                    if node_id in ['promptgen', 'rag_context', 'librarian']:
                        attrs['fillcolor'] = self.colors['bright_green']
                    else:  # openai, replicate, tripo, web_crawling, user_input
                        attrs['fillcolor'] = self.colors['orange']
                elif shape == 'box':
                    if node_id == 'output':
                        attrs['fillcolor'] = self.colors['red']
                    else:  # utils, ui_platform, short_term, long_term
                        attrs['fillcolor'] = self.colors['deep_purple']
                
                # Add the node with its attributes
                dot.node(node_id, **attrs)
            
            # Add edges
            for edge in input_data.get('edges', []):
                source = edge.get('from')
                target = edge.get('to')
                if not source or not target:
                    continue
                
                # Assign edge color and weight based on the connection type
                edge_attrs = {}
                
                # Main flow is brighter
                if source == 'api_integrations' or target == 'textgen' or source == 'textgen':
                    edge_attrs['color'] = self.colors['edge_primary']
                    edge_attrs['penwidth'] = '1.5'
                else:
                    edge_attrs['color'] = self.colors['edge']
                    edge_attrs['penwidth'] = '1.2'
                
                # Port positioning based on node position in hierarchy
                if source == 'api_integrations':
                    edge_attrs['tailport'] = 'e'  # Output from right side
                elif source in ['knowledge']:
                    # For knowledge node with multiple connections
                    if target in ['short_term', 'long_term']:
                        edge_attrs['tailport'] = 'ne' if target == 'short_term' else 'se'
                        edge_attrs['headport'] = 'w'
                    else:
                        edge_attrs['tailport'] = 'e'
                elif source == 'openai' and target in ['promptgen', 'textgen', 'librarian', 'rag_context']:
                    # Better positioning for OpenAI's multiple outputs
                    if target == 'promptgen':
                        edge_attrs['tailport'] = 'ne'
                    elif target == 'textgen':
                        edge_attrs['tailport'] = 'e'
                    elif target == 'librarian':
                        edge_attrs['tailport'] = 'se'
                    elif target == 'rag_context':
                        edge_attrs['tailport'] = 's'
                elif source in ['user_input', 'ui_platform']:
                    # For the UI flow which should be horizontal
                    edge_attrs['tailport'] = 'e'
                    edge_attrs['headport'] = 'w'
                
                # Add the edge
                dot.edge(source, target, **edge_attrs)
            
            # Generate the PNG directly
            output_png = f"{output_path}_flat.png"
            
            # First render to a dot file
            dot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.dot')
            dot_file.close()
            dot.save(dot_file.name)
            
            # Then use the command-line tool for better transparency support
            cmd = f"dot -Tpng -Gbgcolor=transparent {dot_file.name} -o {output_png}"
            logger.info(f"Running command: {cmd}")
            result = os.system(cmd)
            
            # Clean up the temporary file
            os.unlink(dot_file.name)
            
            if result == 0 and os.path.exists(output_png):
                logger.info(f"Flat visualization saved to: {output_png}")
                return output_png
            else:
                logger.error("Failed to generate flat Graphviz PNG")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate flat Graphviz visualization: {e}")
            return None

    def validate_graph_data(self, data: Dict) -> bool:
        """
        Validate that the input data has the correct structure for graph generation.
        
        Args:
            data: Dictionary with nodes and edges
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check if data is a dictionary
            if not isinstance(data, dict):
                logger.error("Input data is not a dictionary")
                return False
                
            # Check if nodes exist and are a list
            if 'nodes' not in data or not isinstance(data['nodes'], list):
                logger.error("Input data missing 'nodes' list")
                return False
                
            # Check if edges exist and are a list
            if 'edges' not in data or not isinstance(data['edges'], list):
                logger.error("Input data missing 'edges' list")
                return False
                
            # Check if any nodes are defined
            if not data['nodes']:
                logger.error("Input data has empty 'nodes' list")
                return False
                
            # Check if any edges are defined
            if not data['edges']:
                logger.error("Input data has empty 'edges' list")
                return False
            
            # Check each node has an id
            for i, node in enumerate(data['nodes']):
                if 'id' not in node:
                    logger.error(f"Node at index {i} is missing 'id' field")
                    return False
            
            # Check each edge has from and to
            for i, edge in enumerate(data['edges']):
                if 'from' not in edge:
                    logger.error(f"Edge at index {i} is missing 'from' field")
                    return False
                if 'to' not in edge:
                    logger.error(f"Edge at index {i} is missing 'to' field")
                    return False
            
            # Validate that all edges reference valid nodes
            valid_node_ids = {node['id'] for node in data['nodes']}
            for i, edge in enumerate(data['edges']):
                if edge['from'] not in valid_node_ids:
                    logger.error(f"Edge at index {i} references non-existent source node '{edge['from']}'")
                    return False
                if edge['to'] not in valid_node_ids:
                    logger.error(f"Edge at index {i} references non-existent target node '{edge['to']}'")
                    return False
            
            # If we get here, the data is valid
            return True
            
        except Exception as e:
            logger.error(f"Error validating graph data: {e}")
            return False
    
    def generate_from_prompt(self, prompt: str, output_dir: str) -> tuple:
        """
        Generate a graph from a user prompt using OpenAI Responses API.
        
        Args:
            prompt: User's prompt describing the graph to generate
            output_dir: Directory to save the output files
            
        Returns:
            Tuple of (path_to_standard_graph, path_to_flat_graph) or (None, None) if generation failed
        """
        try:
            # Create OpenAI Responses API client
            openai_api = OpenAIResponsesAPI(
                model="gpt-4o", 
                temperature=0.7
            )
            
            # Define system prompt
            system_prompt = """
            You are an expert graph generation assistant. Your task is to create a detailed graph structure
            based on the user's description. The output must be in the proper format for GraphGen visualization.
            
            Follow these rules:
            1. Always create nodes with unique 'id' fields and descriptive 'label' fields.
            2. Each node must have a 'shape' field that is one of: 'box', 'circle', or 'diamond'.
            3. Each edge must have 'from' and 'to' fields that reference valid node IDs.
            4. For complex labels with line breaks, use \\n in the label text.
            5. Only respond with a valid JSON object containing 'nodes' and 'edges' arrays.
            
            Node shapes should be assigned based on their purpose:
            - 'diamond' for core/central components and high-level abstractions
            - 'circle' for services, inputs, or processing elements
            - 'box' for utilities, outputs, or UI elements
            
            DO NOT include any explanation, prose or markdown. ONLY output the JSON object.
            """
            
            # Print status
            print(f"\n=== Generating Graph from Prompt ===")
            print(f"Prompt: {prompt}")
            print(f"Generating graph structure with OpenAI...")
            
            # Generate graph structure
            max_attempts = 3
            for attempt in range(1, max_attempts + 1):
                print(f"\nAttempt {attempt}/{max_attempts}...")
                
                # Get structured output from OpenAI
                result = openai_api.structured_response(
                    user_prompt=f"""
                    Please generate a graph structure for the following description:
                    
                    {prompt}
                    
                    Create a detailed system architecture representation with appropriate nodes and edges.
                    Make sure to use descriptive labels and appropriate node shapes.
                    
                    Remember to create a complete, properly formatted JSON object with 'nodes' and 'edges' arrays.
                    Do not include any explanation, just the JSON.
                    """,
                    system_prompt=system_prompt,
                    temperature=0.7 if attempt > 1 else 0.2,  # Increase temperature on retry for more variance
                )
                
                # Check if we got an error
                if isinstance(result, dict) and 'error' in result:
                    print(f"Error from OpenAI: {result['error']}")
                    print("Retrying with different parameters...")
                    continue
                
                # Validate the response
                if self.validate_graph_data(result):
                    print("✅ Generated valid graph structure!")
                    break
                else:
                    print("❌ Generated invalid graph structure, retrying...")
            else:
                # If we get here, we failed to generate a valid graph
                print("Failed to generate valid graph structure after multiple attempts.")
                return None, None
            
            # Create sanitized filename from prompt
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_name = self._sanitize_filename(prompt)
            output_basename = f"{sanitized_name}_{timestamp}"
            output_path = os.path.join(output_dir, output_basename)
            
            # Generate graphs
            print(f"\nGenerating visualization...")
            standard_graph = self.generate_graph(result, output_path)
            
            print(f"\nGenerating flat visualization...")
            flat_graph = self._generate_flat_graphviz(result, output_path)
            
            if standard_graph and flat_graph:
                print(f"\n✅ Graphs generated successfully!")
                print(f"Standard graph: {standard_graph}")
                print(f"Flat graph: {flat_graph}")
                return standard_graph, flat_graph
            else:
                print(f"\n❌ Failed to generate one or both graphs.")
                return None, None
                
        except Exception as e:
            logger.error(f"Error generating graph from prompt: {e}")
            print(f"Error: {e}")
            return None, None
    
    def _sanitize_filename(self, text: str) -> str:
        """
        Create a safe filename from text.
        
        Args:
            text: Input text to convert to a filename
            
        Returns:
            Sanitized filename
        """
        # Extract first 30 characters
        short_text = text[:30].strip()
        
        # Convert to lowercase and replace spaces with underscores
        sanitized = short_text.lower().replace(' ', '_')
        
        # Remove any characters that aren't alphanumeric, underscore, or hyphen
        sanitized = re.sub(r'[^a-z0-9_-]', '', sanitized)
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = "graph"
            
        return sanitized

if __name__ == "__main__":
    # Set up project paths
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent.parent  # Go up to project root
    
    # Create output directory
    output_dir = project_root / "data" / "output" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create graph generator
    graph_gen = GraphGen()
    
    # ArX mini system architecture data that matches the image exactly
    arx_mini_data = {
        "nodes": [
            # Core diamond nodes (blue and purple)
            {"id": "api_integrations", "label": "API\nIntegrations", "shape": "diamond"},
            {"id": "textgen", "label": "TextGen", "shape": "diamond"},
            {"id": "tools", "label": "Tools", "shape": "diamond"},
            {"id": "knowledge", "label": "Knowledge", "shape": "diamond"},
            
            # Circular service nodes (orange and green)
            {"id": "openai", "label": "OpenAI", "shape": "circle"},
            {"id": "replicate", "label": "Replicate", "shape": "circle"},
            {"id": "tripo", "label": "Tripo", "shape": "circle"},
            {"id": "web_crawling", "label": "Web\nCrawling", "shape": "circle"},
            {"id": "promptgen", "label": "PromptGen", "shape": "circle"},
            {"id": "rag_context", "label": "RAG\nContext", "shape": "circle"},
            {"id": "librarian", "label": "Librarian", "shape": "circle"},
            {"id": "user_input", "label": "User\nInput", "shape": "circle"},
            
            # Rectangle nodes (purple, red)
            {"id": "utils", "label": "Utils", "shape": "box"},
            {"id": "ui_platform", "label": "UI\nPlatform", "shape": "box"},
            {"id": "short_term", "label": "Short Term", "shape": "box"},
            {"id": "long_term", "label": "Long Term", "shape": "box"},
            {"id": "output", "label": "Output", "shape": "box"}
        ],
        "edges": [
            # API Integration connections
            {"from": "api_integrations", "to": "openai"},
            {"from": "api_integrations", "to": "replicate"},
            {"from": "api_integrations", "to": "tripo"},
            {"from": "api_integrations", "to": "web_crawling"},
            
            # Integration tools connections
            {"from": "replicate", "to": "tools"},
            {"from": "tripo", "to": "tools"},
            {"from": "web_crawling", "to": "tools"},
            
            # OpenAI connections
            {"from": "openai", "to": "promptgen"},
            {"from": "openai", "to": "textgen"},
            {"from": "openai", "to": "librarian"},
            {"from": "openai", "to": "rag_context"},
            
            # Middle tier connections
            {"from": "promptgen", "to": "tools"},
            {"from": "tools", "to": "textgen"},
            {"from": "utils", "to": "tools"},  # Reversed - utils used by tools
            
            # Knowledge and context connections
            {"from": "librarian", "to": "knowledge"},   # Librarian provides info to knowledge
            {"from": "rag_context", "to": "knowledge"}, # RAG context extracts from knowledge
            {"from": "rag_context", "to": "textgen"},
            
            # Knowledge memory - no incoming arrows
            {"from": "knowledge", "to": "short_term"},
            {"from": "knowledge", "to": "long_term"},
            
            # Output flow
            {"from": "textgen", "to": "ui_platform"},
            {"from": "user_input", "to": "ui_platform"},
            {"from": "ui_platform", "to": "output"}
        ]
    }
    
    # Generate the ArX mini diagram
    print("\n=== Generating ArX Mini Architecture Diagram ===")
    graphviz_output = graph_gen.generate_graph(
        input_data=arx_mini_data,
        output_path=str(output_dir / "arx_mini_architecture")
    )
    
    if graphviz_output:
        print(f"Diagram generation successful! Output saved to: {graphviz_output}")
    else:
        print("Diagram generation failed.")
    
    # Generate flat version of the diagram
    print("\n=== Generating Flat ArX Mini Architecture Diagram ===")
    flat_output = graph_gen._generate_flat_graphviz(
        input_data=arx_mini_data,
        output_path=str(output_dir / "arx_mini_architecture")
    )
    
    if flat_output:
        print(f"Flat diagram generation successful! Output saved to: {flat_output}")
    else:
        print("Flat diagram generation failed.")
    
    print("\nStandard examples completed. Check the output directory for results.")
    
    # Interactive prompt-based graph generation
    print("\n=== Interactive Graph Generation ===")
    print("Enter a description of the graph you want to create, or 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter graph description: ")
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        if not user_input.strip():
            print("Please enter a description.")
            continue
        
        # Generate graphs from user prompt
        standard_graph, flat_graph = graph_gen.generate_from_prompt(user_input, str(output_dir))
        
        # Wait for user to continue
        input("\nPress Enter to continue...") 