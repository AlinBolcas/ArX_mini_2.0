"""Graph management and visualization for ArX knowledge system."""
import graphviz
from pathlib import Path
import logging
import re
from typing import List, Dict, Optional, Set, Union
import json

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self):
        """Initialize graph manager."""
        self.base_colors = {
            'root': '#00BFFF',      # Bright blue for ArX
            'primary': '#3B82F6',   # Medium blue for direct connections
            'secondary': '#1E3A8A', # Deep blue for other nodes
            'edge': '#ffffff80',    # Semi-transparent white
            'edge_primary': '#60A5FA90'  # Light blue for ArX edges
        }
        
        # Initialize graph state
        self.nodes: Set[str] = {'arx'}
        self.edges: Set[tuple] = set()
        self.node_attributes: Dict[str, Dict] = {
            'arx': {
                'shape': 'circle',
                'style': 'filled',
                'fillcolor': self.base_colors['root'],
                'fontcolor': 'white',
                'width': '1.0',
                'height': '1.0',
                'root': 'true'
            }
        }

    def merge_relationships(self, new_dot_content: str) -> None:
        """Merge new relationships into existing graph state."""
        try:
            # Extract relationships from new DOT content
            new_relationships = self._extract_relationships(new_dot_content)
            
            # Add new nodes and edges to existing state
            for rel in new_relationships:
                source = self._clean_node_name(rel['source'])
                target = self._clean_node_name(rel['target'])
                
                # Add nodes if they don't exist
                for node in [source, target]:
                    if node not in self.nodes:
                        self.nodes.add(node)
                        # Style based on domain (extracted from DOT subgraphs)
                        self.node_attributes[node] = {
                            'fillcolor': self._get_domain_color(node, new_dot_content),
                            'fontsize': '10'
                        }
                
                # Add edge if it doesn't exist
                edge = (source, target)
                if edge not in self.edges:
                    self.edges.add(edge)

        except Exception as e:
            logger.error(f"Failed to merge relationships: {e}")

    def create_visualization(self, dot_content: str, output_path: Path) -> Optional[str]:
        """Generate a high-resolution radial visualization from DOT content."""
        try:
            # Merge new relationships into graph state
            self.merge_relationships(dot_content)
            
            # Create new directed graph with twopi layout
            dot = graphviz.Graph(
                'knowledge_graph',
                comment='ArX Knowledge Graph',
                engine='twopi'
            )

            # Graph attributes for radial layout
            dot.attr(
                layout='twopi',
                ranksep='2',
                overlap='false',
                splines='curved',
                bgcolor='#0A0A0F',
                fontname='Helvetica,Arial,sans-serif',
                dpi='300'
            )

            # Node defaults
            dot.attr('node',
                shape='rect',
                style='filled,rounded',
                fillcolor=self.base_colors['secondary'],
                fontcolor='white',
                fontname='Helvetica,Arial,sans-serif',
                fontsize='11',
                margin='0.2',
                penwidth='0',
                radius='5'
            )

            # Edge defaults
            dot.attr('edge',
                color=self.base_colors['edge'],
                fontcolor='white',
                fontname='Helvetica,Arial,sans-serif',
                fontsize='9',
                penwidth='1.2',
                arrowsize='0.6'
            )

            # Add all nodes and edges from graph state
            for node in self.nodes:
                attrs = self.node_attributes.get(node, {})
                dot.node(node, node.replace('_', ' ').title(), **attrs)

            for source, target in self.edges:
                is_arx_edge = source.lower() == 'arx' or target.lower() == 'arx'
                dot.edge(source, target,
                    color=self.base_colors['edge_primary'] if is_arx_edge else self.base_colors['edge'],
                    penwidth=str(1.5 if is_arx_edge else 1.0),
                    weight=str(2 if is_arx_edge else 1)
                )

            # Save with timestamp
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dot.render(str(output_path), format='png', cleanup=True)
            
            # Save graph state
            self._save_graph_state(output_path.parent / 'graph_state.json')
            
            return str(output_path) + '.png'

        except Exception as e:
            logger.error(f"Failed to visualize graph: {e}")
            return None

    def _save_graph_state(self, path: Path) -> None:
        """Save current graph state to JSON file."""
        state = {
            'nodes': list(self.nodes),
            'edges': list(self.edges),
            'node_attributes': self.node_attributes
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=4)

    def _load_graph_state(self, path: Path) -> None:
        """Load graph state from JSON file."""
        if path.exists():
            with open(path, 'r') as f:
                state = json.load(f)
                self.nodes = set(state['nodes'])
                self.edges = set(tuple(edge) for edge in state['edges'])
                self.node_attributes = state['node_attributes']

    def _extract_relationships(self, dot_content: str) -> List[Dict[str, str]]:
        """Extract relationships from DOT content."""
        relationships = []
        
        # Clean up the content to focus on relationship lines
        lines = [line.strip() for line in dot_content.split('\n') 
                if '--' in line and not line.startswith('#')]
        
        for line in lines:
            if '[' in line:  # Line with attributes
                parts = line.split('--')
                if len(parts) == 2:
                    source = parts[0].strip()
                    target_attrs = parts[1].split('[')
                    target = target_attrs[0].strip()
                    attrs = target_attrs[1].strip(']').strip() if len(target_attrs) > 1 else ''
                    
                    relationships.append({
                        'source': source,
                        'target': target,
                        'attrs': attrs
                    })
            else:  # Simple relationship line
                parts = line.split('--')
                if len(parts) == 2:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    
                    relationships.append({
                        'source': source,
                        'target': target,
                        'attrs': ''
                    })
        
        return relationships

    def _clean_node_name(self, name: str) -> str:
        """Clean node names for graphviz compatibility."""
        clean = re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        clean = re.sub(r'_+', '_', clean)
        return clean.strip('_')

    def _get_domain_color(self, node: str, dot_content: str) -> str:
        """Determine node color based on its domain in the DOT content."""
        if node.lower() == 'arx':
            return self.base_colors['root']
        elif 'cluster_technical' in dot_content and node in dot_content:
            return self.base_colors['secondary']
        elif 'cluster_creative' in dot_content and node in dot_content:
            return self.base_colors['primary']
        else:
            return self.base_colors['secondary'] 