"""Neuron-like graph visualization with brain-inspired structure."""
import graphviz
from pathlib import Path
import random
import logging

from typing import List, Dict, Set, Optional, Tuple

logger = logging.getLogger(__name__)

class NeuronGraph:
    def __init__(self):
        """Initialize neuron graph generator."""
        self.base_colors = {
            'central': '#00BFFF',      # Bright blue for central node
            'primary': '#3B82F6',      # Medium blue for primary connections
            'secondary': '#1E3A8A',    # Deep blue for secondary nodes
            'tertiary': '#6D28D9',     # Purple for tertiary nodes
            'edge': '#ffffff80',       # Semi-transparent white
            'edge_primary': '#60A5FA90' # Light blue for primary edges
        }
        
        # Initialize graph state
        self.nodes: Set[str] = {'brain'}
        self.edges: Set[tuple] = set()
        self.node_attributes: Dict[str, Dict] = {
            'brain': {
                'shape': 'circle',
                'style': 'filled',
                'fillcolor': self.base_colors['central'],
                'fontcolor': 'white',
                'width': '1.2',
                'height': '1.2',
                'central': 'true'
            }
        }

    def generate_random_neuron(self, 
                               primary_nodes: int = 6, 
                               secondary_nodes: int = 12,
                               tertiary_nodes: int = 18) -> None:
        """Generate a random neuron-like structure."""
        try:
            # Generate primary nodes connected to central "brain" node
            for i in range(primary_nodes):
                node_name = f"region_{i+1}"
                self.nodes.add(node_name)
                self.node_attributes[node_name] = {
                    'fillcolor': self.base_colors['primary'],
                    'fontsize': '12'
                }
                self.edges.add(('brain', node_name))
                
                # Generate secondary nodes connected to each primary node
                secondary_count = random.randint(1, 3)
                for j in range(secondary_count):
                    sec_node = f"{node_name}_area_{j+1}"
                    self.nodes.add(sec_node)
                    self.node_attributes[sec_node] = {
                        'fillcolor': self.base_colors['secondary'],
                        'fontsize': '10'
                    }
                    self.edges.add((node_name, sec_node))
                    
                    # Generate tertiary nodes
                    tertiary_count = random.randint(1, 4)
                    for k in range(tertiary_count):
                        tert_node = f"{sec_node}_neuron_{k+1}"
                        self.nodes.add(tert_node)
                        self.node_attributes[tert_node] = {
                            'fillcolor': self.base_colors['tertiary'],
                            'fontsize': '8'
                        }
                        self.edges.add((sec_node, tert_node))
            
            # Add some cross-connections for complexity
            all_nodes = list(self.nodes)
            cross_connections = random.randint(5, 10)
            for _ in range(cross_connections):
                source = random.choice(all_nodes)
                target = random.choice(all_nodes)
                if source != target and (source, target) not in self.edges and (target, source) not in self.edges:
                    self.edges.add((source, target))
                    
            logger.info(f"Generated neuron graph with {len(self.nodes)} nodes and {len(self.edges)} connections")
            
        except Exception as e:
            logger.error(f"Failed to generate neuron graph: {e}")

    def create_visualization(self, output_path: Path) -> Optional[str]:
        """Generate a high-resolution radial visualization of the neuron graph."""
        try:
            # Create new directed graph with twopi layout
            dot = graphviz.Graph(
                'neuron_graph',
                comment='Neuron-like Knowledge Graph',
                engine='twopi'
            )

            # Graph attributes for radial layout
            dot.attr(
                layout='twopi',
                ranksep='2.5',
                overlap='false',
                splines='curved',
                bgcolor='#0A0A0F',
                fontname='Helvetica,Arial,sans-serif',
                dpi='300'
            )

            # Node defaults
            dot.attr('node',
                shape='ellipse',
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
                # Format node labels for better readability
                label = node.replace('_', ' ').title()
                if 'neuron' in node:
                    label = label.split('Neuron')[0].strip()
                dot.node(node, label, **attrs)

            for source, target in self.edges:
                is_central_edge = source.lower() == 'brain' or target.lower() == 'brain'
                dot.edge(source, target,
                    color=self.base_colors['edge_primary'] if is_central_edge else self.base_colors['edge'],
                    penwidth=str(1.5 if is_central_edge else 1.0),
                    weight=str(2 if is_central_edge else 1)
                )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Render and save the graph
            dot.render(str(output_path), format='png', cleanup=True)
            
            return str(output_path) + '.png'

        except Exception as e:
            logger.error(f"Failed to visualize neuron graph: {e}")
            return None


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    graph = NeuronGraph()
    graph.generate_random_neuron(primary_nodes=8, secondary_nodes=15, tertiary_nodes=24)
    output_file = Path("output/neuron_graph")
    result = graph.create_visualization(output_file)
    
    if result:
        print(f"Graph visualization saved to: {result}")
    else:
        print("Failed to generate graph")