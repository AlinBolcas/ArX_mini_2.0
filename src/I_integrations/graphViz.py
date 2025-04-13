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
from openai_API import OpenAIWrapper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphGen:
    """
    Graph generation utility using Graphviz for high-quality visualizations.
    Specializes in system architecture diagrams with consistent styling.
    """
    
    def __init__(self):
        """
        Initialize the graph generator with a consistent color scheme for visualizations.
        
        Color Meaning Convention:
        - Blue: Core/central components, main processing engines, primary modules
        - Purple: Tools, knowledge components, libraries, code modules, plugins
        - Green: Processing services, transformation agents, content generation
        - Orange: Inputs, external services, data sources, user interactions
        - Red: Outputs, results, final deliverables, produced artifacts
        - Yellow: Decision points, evaluation components, filtering nodes
        - Teal: Data transformation, enrichment services, augmentation
        - Gray: Support infrastructure, background services, passive components
        """
        # Core color scheme - ~15% MORE VIBRANT/SATURATED PALETTE
        self.colors = {
            # Background
            'background': 'transparent',     # Transparent background
            
            # Core component colors - MORE VIBRANT BLUES
            'central': '#3060B0',            # Brighter navy blue (was #285095)
            'primary_blue': '#2565C0',       # Brighter royal blue (was #1A4B91)
            'secondary_blue': '#154590',     # Brighter dark navy blue (was #0F3570)
            'light_blue': '#4090E0',         # Brighter moderate blue (was #3178C0)
            
            # Function & purpose colors - MORE VIBRANT
            'purple': '#7D40D0',             # Brighter purple (was #6935B5)
            'deep_purple': '#6035B0',        # Brighter deep purple (was #502990)
            'bright_green': '#35A575',       # Brighter green (was #2A8B63)
            'teal': '#3095A0',               # Brighter teal (was #227A84)
            'yellow': '#D0A030',             # Brighter gold (was #B38728)
            'orange': '#E06040',             # Brighter orange (was #C05032)
            'red': '#D54545',                # Brighter red (was #B73A3A)
            'gray': '#6A7180',               # Brighter gray (was #5A616F)
            
            # Edge colors - Adjusted alpha for visibility if needed
            'edge': '#8A8A8A70',             # Darker gray semi-transparent (Alpha adjusted slightly)
            'edge_primary': '#2565C0A0',     # Brighter blue semi-transparent (Alpha adjusted)
            'edge_decision': '#D0A03090',    # Brighter gold semi-transparent (Alpha adjusted)
            'edge_feedback': '#D5454590',    # Brighter red semi-transparent (Alpha adjusted)
            
            # Text color
            'text': '#ffffff'                # White text for readability
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
            
            # Core graph attributes - Increased font size
            dot.attr(
                fontname='Helvetica,Arial,sans-serif',
                fontcolor=self.colors['text'],
                fontsize='30',  # Increased from '14'
                rankdir='TB',  # Top to bottom layout
                bgcolor='transparent',  # Transparent background
                dpi='300',
                splines='true',  # Curved edges
                overlap='false',
                concentrate='false', # Don't merge edges (better visibility)
                nodesep='0.7',
                ranksep='1.2'
            )
            
            # Node defaults - more prominent nodes - Increased font size
            dot.attr('node',
                shape='box',
                style='filled,rounded',
                fillcolor=self.colors['secondary_blue'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='26',  # Increased from '12'
                margin='0.2',
                height='0.6',
                width='0.8',
                penwidth='1.2'  # Slightly thicker borders for visibility
            )
            
            # Edge defaults - make edges much more visible - Increased font size
            dot.attr('edge',
                color=self.colors['edge_primary'],  # Brighter edges by default
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='22',  # Increased from '10'
                penwidth='1.5',   # Thicker edges for better visibility
                arrowsize='0.9'   # Larger arrow heads
            )
            
            # Get the existing node IDs from the input data
            existing_node_ids = {node.get('id') for node in input_data.get('nodes', [])}
            
            # Only use rank constraints for ArX mini architecture if those nodes exist
            if 'api_integrations' in existing_node_ids and 'textgen' in existing_node_ids:
                # This appears to be the ArX mini architecture or something similar
                # Use the predefined rank structure
                ranks = {
                    "api_tier": ["api_integrations"],
                    "service_tier": ["openai", "replicate", "tripo", "web_crawling"],
                    "middle_tier": ["promptgen", "tools"],
                    "processing_tier": ["textgen", "utils", "librarian", "rag_context"],
                    "knowledge_tier": ["knowledge"],
                    "memory_tier": ["short_term", "long_term"],
                }
                
                # Create invisible constraints to position nodes in their tiers
                # but only for nodes that actually exist in the input data
                for rank_name, nodes in ranks.items():
                    # Filter to only include nodes that exist in the input data
                    existing_nodes = [node for node in nodes if node in existing_node_ids]
                    
                    # Only create a rank constraint if there are nodes to include
                    if existing_nodes:
                        with dot.subgraph(name=f"cluster_{rank_name}") as s:
                            s.attr(rank="same")
                            for node in existing_nodes:
                                s.node(node)
                
                # Check if UI flow nodes exist
                ui_nodes = ["user_input", "ui_platform", "output"]
                existing_ui_nodes = [node for node in ui_nodes if node in existing_node_ids]
                
                # Only create UI flow if relevant nodes exist
                if len(existing_ui_nodes) >= 2:  # Need at least 2 nodes for a flow
                    with dot.subgraph(name="cluster_ui_horizontal") as s:
                        s.attr(rank="same")
                        s.attr(rankdir="LR")  # Force left-to-right inside this subgraph
                        
                        for node in existing_ui_nodes:
                            s.node(node)
                        
                        # Add invisible edges to enforce ordering if all three nodes exist
                        if "user_input" in existing_ui_nodes and "ui_platform" in existing_ui_nodes:
                            s.edge("user_input", "ui_platform", style="invis", weight="10")
                        if "ui_platform" in existing_ui_nodes and "output" in existing_ui_nodes:
                            s.edge("ui_platform", "output", style="invis", weight="10")
            else:
                # For non-ArX mini graphs, use a more generic approach
                
                # Group nodes by shape for potential ranking
                shape_groups = {"diamond": [], "circle": [], "box": []}
                
                # Collect nodes by shape
                for node in input_data.get('nodes', []):
                    node_id = node.get('id')
                    shape = node.get('shape', 'box').lower()
                    if shape in shape_groups:
                        shape_groups[shape].append(node_id)
                
                # Create ranks for shapes if they have multiple nodes
                for shape, nodes in shape_groups.items():
                    if len(nodes) > 1:
                        with dot.subgraph(name=f"cluster_{shape}_group") as s:
                            s.attr(rank="same")
                            for node in nodes:
                                s.node(node)
            
            # Add nodes with appropriate colors and shapes
            for node in input_data.get('nodes', []):
                node_id = node.get('id')
                if not node_id:
                    continue
                    
                label = node.get('label', node_id).replace('\\n', '\n')  # Handle line breaks
                shape = node.get('shape', 'box').lower()
                
                # Apply consistent visual conventions for node styling
                attrs = {
                    'label': label,
                    'shape': shape,
                    'style': 'filled,rounded'
                }
                
                # Apply the consistent visual convention for node colors
                # Using the same rules across all visualizations ensures readability
                if shape == 'diamond':
                    # Diamond nodes are now primarily BLUE (central concepts, regions, main modules)
                    if node_id.lower() in ['brain', 'central', 'main', 'core'] or 'region' in node_id.lower():
                        # Central bright blue for brain/central node
                        attrs['fillcolor'] = self.colors['central']
                    elif any(term in node_id.lower() for term in ['api', 'engine', 'framework', 'pipeline', 'architecture']):
                        # Medium blue for primary framework components
                        attrs['fillcolor'] = self.colors['primary_blue']
                    elif any(term in node_id.lower() for term in ['area', 'module', 'component']):
                        # Deep blue for secondary areas/modules
                        attrs['fillcolor'] = self.colors['secondary_blue']
                    else:
                        # Purple only for knowledge/tools diamonds
                        attrs['fillcolor'] = self.colors['purple']
                
                elif shape == 'circle':
                    # Circles get a more balanced mix, favoring blue over orange
                    if any(term in node_id.lower() for term in ['region', 'area', 'component', 'module']):
                        # Secondary blue for region/area nodes
                        attrs['fillcolor'] = self.colors['secondary_blue']
                    elif any(term in node_id.lower() for term in ['gen', 'process', 'transform', 'librarian']):
                        # Green for generative/processing nodes
                        attrs['fillcolor'] = self.colors['bright_green']
                    elif any(term in node_id.lower() for term in ['enhance', 'enrich', 'augment']):
                        # Teal for enhancement processes
                        attrs['fillcolor'] = self.colors['teal']
                    elif any(term in node_id.lower() for term in ['eval', 'assess', 'filter', 'select']):
                        # Yellow for evaluation/decision points
                        attrs['fillcolor'] = self.colors['yellow']
                    elif any(term in node_id.lower() for term in ['input', 'extern', 'user', 'source']):
                        # Orange for external inputs
                        attrs['fillcolor'] = self.colors['orange']
                    else:
                        # Light blue as default for other circles
                        attrs['fillcolor'] = self.colors['light_blue']
                
                elif shape == 'box':
                    # Box nodes - more variety, fewer purples
                    if node_id.lower() == 'output' or 'result' in node_id.lower():
                        # Red specifically for outputs
                        attrs['fillcolor'] = self.colors['red']
                    elif 'database' in node_id.lower() or 'storage' in node_id.lower():
                        # Deep purple for databases
                        attrs['fillcolor'] = self.colors['deep_purple']
                    elif any(term in node_id.lower() for term in ['ui', 'interface', 'platform']):
                        # Light blue for UI/interface components
                        attrs['fillcolor'] = self.colors['light_blue']
                    elif any(term in node_id.lower() for term in ['support', 'infra', 'background', 'config']):
                        # Gray for infrastructure/support
                        attrs['fillcolor'] = self.colors['gray']
                    else:
                        # Purple as fallback for other boxes
                        attrs['fillcolor'] = self.colors['purple']
                
                # Add the node with its attributes
                dot.node(node_id, **attrs)
            
            # Add edges with proper direction and visibility
            for edge in input_data.get('edges', []):
                source = edge.get('from')
                target = edge.get('to')
                if not source or not target:
                    continue
                    
                # Assign edge color and weight based on connection type and purpose
                edge_attrs = {}
                
                # Determine the flow type to apply appropriate visual styling
                # Get source and target node information
                source_node = next((n for n in input_data.get('nodes', []) if n.get('id') == source), None)
                target_node = next((n for n in input_data.get('nodes', []) if n.get('id') == target), None)
                source_shape = source_node.get('shape', '').lower() if source_node else ''
                target_shape = target_node.get('shape', '').lower() if target_node else ''
                
                # Identify main flow paths using general rules
                is_main_flow = False
                is_decision_flow = False
                is_feedback_flow = False
                
                # Main flow typically involves core system components
                if any(term in source.lower() for term in ['api', 'core', 'main', 'central', 'engine']):
                    is_main_flow = True
                # Main flow often ends at output/results nodes
                elif target_shape == 'circle' and 'output' in target.lower():
                    is_main_flow = True
                # Diamond-to-diamond connections represent key architectural flows
                elif source_shape == 'diamond' and target_shape == 'diamond':
                    is_main_flow = True
                # Flow to output node is usually main flow
                elif 'output' in target.lower() or 'result' in target.lower():
                    is_main_flow = True
                    
                # Decision flows involve evaluation or filtering nodes
                if (source_shape == 'circle' and 
                    any(term in source.lower() for term in ['decide', 'evaluate', 'filter', 'select'])):
                    is_decision_flow = True
                    
                # Feedback flows typically go backward in the processing chain
                # Identify by checking if the source appears later in process than target
                source_idx = -1
                target_idx = -1
                for i, node in enumerate(input_data.get('nodes', [])):
                    if node.get('id') == source:
                        source_idx = i
                    elif node.get('id') == target:
                        target_idx = i
                # If source appears after target in the nodes list, it may be feedback
                # This is a heuristic that often works since nodes are typically defined in processing order
                if source_idx > target_idx and source_idx >= 0 and target_idx >= 0:
                    # Additional check: if target is a core component and source is a later stage
                    if (target_shape == 'diamond' and 
                        any(term in source.lower() for term in ['output', 'result', 'feedback'])):
                        is_feedback_flow = True
                        
                # Apply visual styling based on flow type
                if is_main_flow:
                    edge_attrs['color'] = self.colors['edge_primary']
                    edge_attrs['penwidth'] = '1.8'
                elif is_decision_flow:
                    edge_attrs['color'] = self.colors['edge_decision']
                    edge_attrs['penwidth'] = '1.5'
                elif is_feedback_flow:
                    edge_attrs['color'] = self.colors['edge_feedback']
                    edge_attrs['penwidth'] = '1.5'
                    edge_attrs['style'] = 'dashed'
                else:
                    edge_attrs['color'] = self.colors['edge']
                    edge_attrs['penwidth'] = '1.2'
                
                # Add the edge with its attributes
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
        Applies consistent visual conventions for shapes and colors across domains.
        
        Args:
            input_data: Dictionary with nodes and edges
            output_path: Where to save the final image (without extension)
            
        Returns:
            Path to the generated PNG
        """
        try:
            # Create directed graph with dot engine (hierarchical layout)
            dot = graphviz.Digraph('flat_graph', comment='System Architecture', format='png', engine='dot')
            
            # Set graph attributes for visibility and transparency - Increased font size
            dot.attr(
                fontname='Helvetica,Arial,sans-serif',
                fontcolor=self.colors['text'],
                fontsize='24',  # Increased from '14'
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
            
            # Node defaults - Increased font size
            dot.attr('node',
                shape='box',
                style='filled,rounded',
                fillcolor=self.colors['secondary_blue'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='20',  # Increased from '12'
                margin='0.2',
                height='0.6',
                width='0.8',
                penwidth='1.0',
                fixedsize='false'  # Allow nodes to size properly for content
            )
            
            # Edge defaults - make edges visible but less prominent - Increased font size
            dot.attr('edge',
                color=self.colors['edge'],
                fontcolor=self.colors['text'],
                fontname='Helvetica,Arial,sans-serif',
                fontsize='18',  # Increased from '10'
                penwidth='1.2',
                arrowsize='0.8',
                minlen='1.5',      # Minimum edge length
                arrowhead='normal',
                tailport='e',      # East port for output (right side)
                headport='w'       # West port for input (left side)
            )
            
            # Get the existing node IDs from the input data
            existing_node_ids = {node.get('id') for node in input_data.get('nodes', [])}
            
            # Add nodes with appropriate colors and shapes following visual conventions
            for node in input_data.get('nodes', []):
                node_id = node.get('id')
                if not node_id:
                    continue
                    
                label = node.get('label', node_id).replace('\\n', '\n')
                shape = node.get('shape', 'box').lower()
                
                # Set up node attributes with consistent visual conventions
                attrs = {
                    'label': label,
                    'shape': shape,
                    'style': 'filled,rounded'
                }
                
                # Apply color conventions based on shape and purpose
                # VISUAL CONVENTION:
                # - Diamonds: Primarily blue for core components, with purple for knowledge/tools
                # - Circles: Mix of blues, greens, and occasional orange
                # - Boxes: Mix of blues, purples, and occasional red for outputs
                if shape == 'diamond':
                    # Diamond nodes are now primarily BLUE (central concepts, regions, main modules)
                    if node_id.lower() in ['brain', 'central', 'main', 'core'] or 'region' in node_id.lower():
                        # Central bright blue for brain/central node
                        attrs['fillcolor'] = self.colors['central']
                    elif any(term in node_id.lower() for term in ['api', 'engine', 'framework', 'pipeline', 'architecture']):
                        # Medium blue for primary framework components
                        attrs['fillcolor'] = self.colors['primary_blue']
                    elif any(term in node_id.lower() for term in ['area', 'module', 'component']):
                        # Deep blue for secondary areas/modules
                        attrs['fillcolor'] = self.colors['secondary_blue']
                    else:
                        # Purple only for knowledge/tools diamonds
                        attrs['fillcolor'] = self.colors['purple']
                
                elif shape == 'circle':
                    # Circles get a more balanced mix, favoring blue over orange
                    if any(term in node_id.lower() for term in ['region', 'area', 'component', 'module']):
                        # Secondary blue for region/area nodes
                        attrs['fillcolor'] = self.colors['secondary_blue']
                    elif any(term in node_id.lower() for term in ['gen', 'process', 'transform', 'librarian']):
                        # Green for generative/processing nodes
                        attrs['fillcolor'] = self.colors['bright_green']
                    elif any(term in node_id.lower() for term in ['enhance', 'enrich', 'augment']):
                        # Teal for enhancement processes
                        attrs['fillcolor'] = self.colors['teal']
                    elif any(term in node_id.lower() for term in ['eval', 'assess', 'filter', 'select']):
                        # Yellow for evaluation/decision points
                        attrs['fillcolor'] = self.colors['yellow']
                    elif any(term in node_id.lower() for term in ['input', 'extern', 'user', 'source']):
                        # Orange for external inputs
                        attrs['fillcolor'] = self.colors['orange']
                    else:
                        # Light blue as default for other circles
                        attrs['fillcolor'] = self.colors['light_blue']
                
                elif shape == 'box':
                    # Box nodes - more variety, fewer purples
                    if node_id.lower() == 'output' or 'result' in node_id.lower():
                        # Red specifically for outputs
                        attrs['fillcolor'] = self.colors['red']
                    elif 'database' in node_id.lower() or 'storage' in node_id.lower():
                        # Deep purple for databases
                        attrs['fillcolor'] = self.colors['deep_purple']
                    elif any(term in node_id.lower() for term in ['ui', 'interface', 'platform']):
                        # Light blue for UI/interface components
                        attrs['fillcolor'] = self.colors['light_blue']
                    elif any(term in node_id.lower() for term in ['support', 'infra', 'background', 'config']):
                        # Gray for infrastructure/support
                        attrs['fillcolor'] = self.colors['gray']
                    else:
                        # Purple as fallback for other boxes
                        attrs['fillcolor'] = self.colors['purple']
                
                # Add the node with its attributes
                dot.node(node_id, **attrs)
            
            # Add edges applying intelligent routing logic
            for edge in input_data.get('edges', []):
                source = edge.get('from')
                target = edge.get('to')
                if not source or not target:
                    continue
                
                # Assign edge color and weight based on the connection type
                edge_attrs = {}
                
                # Highlight main flow paths with brighter edges
                # Main flow typically involves core components and primary data paths
                is_main_flow = False
                
                # Check if this is part of the main flow
                if any(x in source for x in ['api', 'core', 'main', 'central']) or \
                   any(x in target for x in ['output', 'result', 'final']):
                    is_main_flow = True
                
                # Any diamond-to-diamond connection is likely part of main flow
                source_node = next((n for n in input_data.get('nodes', []) if n.get('id') == source), None)
                target_node = next((n for n in input_data.get('nodes', []) if n.get('id') == target), None)
                if source_node and target_node and \
                   source_node.get('shape', '').lower() == 'diamond' and \
                   target_node.get('shape', '').lower() == 'diamond':
                    is_main_flow = True
                
                if is_main_flow:
                    edge_attrs['color'] = self.colors['edge_primary']
                    edge_attrs['penwidth'] = '1.5'
                else:
                    edge_attrs['color'] = self.colors['edge']
                    edge_attrs['penwidth'] = '1.2'
                
                # Intelligent edge routing based on node positioning
                # This improves readability by controlling where edges connect to nodes
                
                # Diamond nodes - connect to points based on flow direction
                if source_node and source_node.get('shape', '').lower() == 'diamond':
                    edge_attrs['tailport'] = 'e'  # Connect from east side (right)
                
                # Circle nodes - connect based on function
                if source_node and source_node.get('shape', '').lower() == 'circle':
                    if 'input' in source.lower():
                        edge_attrs['tailport'] = 'e'  # Input flows right
                    elif any(x in source.lower() for x in ['process', 'agent', 'service']):
                        # Processing components connect based on target position
                        if target_node and target_node.get('shape', '').lower() == 'diamond':
                            edge_attrs['tailport'] = 'e'  # To diamonds, connect right
                
                # Always ensure edges to output nodes connect smoothly
                if target_node and (target_node.get('shape', '').lower() == 'circle' and 'output' in target.lower()):
                    edge_attrs['headport'] = 'w'  # Output nodes receive from west side (left)
                
                # Add the edge with its attributes
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
            llm = OpenAIWrapper(
                model="gpt-4o-mini", 
                temperature=0.7
            )
            
            # Define system prompt with clear visual convention guidelines
            system_prompt = """
            You are an expert graph generation assistant. Your task is to create a detailed graph structure
            based on the user's description. The output must be in the proper format for GraphGen visualization.
            
            Follow these strict visual conventions for graph components:
            
            SHAPE CONVENTIONS:
            1. DIAMOND shape: Use for core/central components, high-level modules, and key abstractions
               - Examples: Main systems, core frameworks, central concepts, primary modules, main processes
            
            2. CIRCLE shape: Use for services, processes, actions, and transformative components
               - Examples: API services, processing units, operations, agents, functions, activities
            
            3. BOX shape: Use for data, utilities, resources, and supporting elements
               - Examples: Databases, tools, libraries, static resources, outputs, UI elements
            
            COLOR CONVENTIONS - YOU MUST USE ALL OF THESE COLORS IN A BALANCED WAY:
            1. BLUE (primary_blue): Core system components and central modules
               - SPECIFIC EXAMPLES: Main Engine, Central Framework, Core API, Processing Pipeline, Architecture Backbone
               - ALWAYS use blue for 20-30% of nodes, especially for the most central diamond-shaped nodes
            
            2. PURPLE: Tools, knowledge resources, code modules, and capabilities
               - SPECIFIC EXAMPLES: Tools Framework, Knowledge Repository, Code Generator, Plugin System, Resource Library
               - Use purple for both diamond and box shapes that provide capabilities or store information
            
            3. GREEN: Internal processing services and transformation agents
               - SPECIFIC EXAMPLES: Data Processor, Content Generator, Format Converter, Text Analyzer, Data Transformer
               - ALWAYS use green for circle-shaped nodes that transform or process information
            
            4. ORANGE: External services and input sources
               - SPECIFIC EXAMPLES: User Input, External API, Third-party Service, Data Source, Client Request
               - Use orange primarily for circle-shaped nodes that provide input or external functionality
            
            5. RED: Output and final results
               - SPECIFIC EXAMPLES: Output Display, Final Result, Generated Content, Visualization Output
               - ALWAYS use red for the final output nodes, typically box-shaped
            
            IMPORTANT: A good graph should have a MIX of colors - approximately 20-30% blue, 20-25% purple, 
            15-20% green, 20-30% orange, and 5-10% red nodes. Do not over-use any single color.
            
            CONNECTION CONVENTIONS:
            - Use directional arrows to show data/process flow
            - Primary flows should be emphasized with brighter/thicker edges
            - Related components should be visually grouped together
            
            FORMATTING REQUIREMENTS:
            1. Always create nodes with unique 'id' fields and descriptive 'label' fields
            2. Each node must have a 'shape' field that is one of: 'box', 'circle', or 'diamond'
            3. Each edge must have 'from' and 'to' fields that reference valid node IDs
            4. For complex labels with line breaks, use \\n in the label text
            5. Only respond with a valid JSON object containing 'nodes' and 'edges' arrays
            
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
                
                # Get structured output from OpenAI with specific visual guidance
                result = llm.structured_output(
                    user_prompt=f"""
                    Please generate a detailed and comprehensive graph structure for the following description:
                    
                    {prompt}
                    
                    IMPORTANT: Create a highly detailed and exhaustive graph that thoroughly explores all aspects and components of the described system.
                    
                    Create at least 15-25 nodes to ensure comprehensive coverage of all concepts, and ensure proper connectivity between all relevant nodes.
                    
                    Follow these color and shape conventions:
                    1. Use DIAMOND shapes for core/central components and primary modules (use DARK BLUE colors for these)
                    2. Use CIRCLE shapes for services, processes, and transformative activities (use a mix of DARK BLUE, DARK GREEN, and DEEP ORANGE)
                    3. Use BOX shapes for data stores, utilities, and output components (mix of DARK PURPLE, DEEP RED for outputs)
                    
                    IMPORTANT: All colors should be DARK enough to ensure good contrast with white text labels.
                    
                    Be creative yet logical in depicting the complete structure. The goal is to create a diagram that is both visually balanced and informationally comprehensive.
                    
                    Remember to create a complete, properly formatted JSON object with 'nodes' and 'edges' arrays.
                    Each node must have id, label, and shape fields. Each edge must have from and to fields.
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
                    # Print the LLM response so user can see it
                    print("\nLLM Response (JSON structure):")
                    print(json.dumps(result, indent=2))
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
            
            # Generate graphs with retry logic
            max_render_attempts = 3
            standard_graph = None
            flat_graph = None
            
            # Try to generate standard graph with retries
            for render_attempt in range(1, max_render_attempts + 1):
                print(f"\nGenerating visualization (attempt {render_attempt}/{max_render_attempts})...")
                try:
                    standard_graph = self.generate_graph(result, output_path)
                    if standard_graph:
                        print(f"✅ Standard graph generation successful!")
                        break
                except Exception as e:
                    print(f"Error generating standard graph (attempt {render_attempt}): {e}")
                    
                if render_attempt < max_render_attempts:
                    print("Retrying standard graph generation...")
                    time.sleep(1)  # Small delay before retry
            
            # Try to generate flat graph with retries
            for render_attempt in range(1, max_render_attempts + 1):
                print(f"\nGenerating flat visualization (attempt {render_attempt}/{max_render_attempts})...")
                try:
                    flat_graph = self._generate_flat_graphviz(result, output_path)
                    if flat_graph:
                        print(f"✅ Flat graph generation successful!")
                        break
                except Exception as e:
                    print(f"Error generating flat graph (attempt {render_attempt}): {e}")
                    
                if render_attempt < max_render_attempts:
                    print("Retrying flat graph generation...")
                    time.sleep(1)  # Small delay before retry
            
            if standard_graph and flat_graph:
                print(f"\n✅ Both graphs generated successfully!")
                print(f"Standard graph: {standard_graph}")
                print(f"Flat graph: {flat_graph}")
                return standard_graph, flat_graph
            else:
                print(f"\n⚠️ Warning: Only generated {'standard' if standard_graph else 'flat'} graph.")
                return standard_graph, flat_graph
                
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
            {"from": "knowledge", "to": "rag_context"}, # Knowledge feeds into RAG context
            {"from": "rag_context", "to": "textgen"},
            
            # Knowledge memory - no incoming arrows
            {"from": "short_term", "to": "knowledge"},
            {"from": "long_term", "to": "knowledge"},
            
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