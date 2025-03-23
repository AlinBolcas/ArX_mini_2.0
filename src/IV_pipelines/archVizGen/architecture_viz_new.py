import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any, Set, Optional, Tuple
import sys
import re
import graphviz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up project root and add to path properly
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  # Go up to src directory

# Add src directory to path to enable imports
sys.path.insert(0, str(project_root.parent))  # Add project root to path

# Import OpenAI API module with correct path
from src.I_integrations.openai_API import OpenAIAPI

class RepoArchitectureViz:
    def __init__(self, output_dir=None):
        self.project_root = project_root.parent  # Set to actual project root
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = self.project_root / "data" / "output" / "architecture_viz"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI API for architecture analysis
        self.api = OpenAIAPI(
            model="gpt-4o-mini",
            temperature=0.2
        )
        
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
        # Neural graph inspired color scheme
        self.base_colors = {
            'central': '#00BFFF',      # Bright blue for central node
            'primary': '#3B82F6',      # Medium blue for primary connections
            'secondary': '#1E3A8A',    # Deep blue for secondary nodes
            'tertiary': '#6D28D9',     # Purple for tertiary nodes
            'edge': '#ffffff80',       # Semi-transparent white
            'edge_primary': '#60A5FA90' # Light blue for primary edges
        }
        
        # Initialize graph state for Graphviz implementation
        self.nodes: Set[str] = set()
        self.edges: Set[tuple] = set()
        self.node_attributes: Dict[str, Dict] = {}
        
        # Store module imports for dependency analysis
        self.module_imports = {}
        
    def analyze_folder_structure(self, ignore_dirs=None) -> Dict[str, Any]:
        """
        Analyze the repository folder structure in detail
        
        Args:
            ignore_dirs: List of directory names to ignore
        """
        if ignore_dirs is None:
            ignore_dirs = ['__pycache__', '.git', '.github', 'venv', 'env', 'node_modules']
            
        structure = {}
        file_counts = {}
        file_types = {}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            # Skip directories with special prefixes
            if any(p.startswith('__') for p in Path(root).parts):
                continue
                
            rel_path = Path(root).relative_to(self.project_root)
            path_str = str(rel_path)
            
            if path_str == '.':  # Root directory
                current_dict = structure
            else:
                # Navigate to the correct nested dictionary
                current_dict = structure
                for part in rel_path.parts:
                    current_dict = current_dict.setdefault(part, {})
                
                # Count files by directory
                if files:
                    file_counts[path_str] = len(files)
            
            # Add files (ignoring hidden files)
            if files:
                current_dict['_files'] = []
                for f in files:
                    if not f.startswith('.') and not f.startswith('__'):
                        current_dict['_files'].append(f)
                        
                        # Count file types
                        ext = os.path.splitext(f)[1]
                        file_types[ext] = file_types.get(ext, 0) + 1
                        
                        # Analyze Python files for imports
                        if f.endswith('.py'):
                            file_path = os.path.join(root, f)
                            try:
                                self._analyze_imports(file_path, path_str)
                            except Exception as e:
                                print(f"Error analyzing imports in {file_path}: {e}")
                
        # Add file statistics to the structure
        structure['_metadata'] = {
            'file_counts': file_counts,
            'file_types': file_types,
            'directory_count': len(file_counts),
            'total_files': sum(file_counts.values())
        }
        
        return structure
    
    def _analyze_imports(self, file_path: str, module_path: str):
        """Analyze Python file imports to build dependency graph"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Track imports for this module
        imports = set()
        
        # Match standard imports - both 'import x' and 'from x import y'
        import_pattern = r'(?:from\s+([\w.]+)\s+import\s+[\w.,\s*]+)|(?:import\s+([\w.,\s]+))'
        
        for match in re.finditer(import_pattern, content):
            from_import, regular_import = match.groups()
            
            if from_import:
                imports.add(from_import.split('.')[0])  # Add only top-level package
            
            if regular_import:
                # Handle multiple imports like "import os, sys, re"
                for imp in regular_import.split(','):
                    # Extract base package name, handling 'as' aliases
                    base_import = imp.strip().split(' as ')[0].split('.')[0]
                    imports.add(base_import)
        
        # Store the imports for this module
        module_name = os.path.basename(file_path)
        self.module_imports[f"{module_path}/{module_name}"] = list(imports)
            
    def identify_major_components(self, structure: Dict) -> Dict[str, List[str]]:
        """
        Identify major components and categorize them
        
        Returns a dictionary mapping component types to lists of components
        """
        components = {
            'core': [],
            'utility': [],
            'data': [],
            'interface': [],
            'integrations': [],
            'tests': []
        }
        
        # First level breakdown - src, data, tests, etc.
        for key in structure:
            if key == '_metadata' or key == '_files':
                continue
                
            if key == 'src':
                # Second level breakdown within src
                src_structure = structure[key]
                for src_key in src_structure:
                    if src_key == '_files':
                        continue
                    
                    # Categorize based on naming conventions and content
                    if src_key.startswith('I_'):
                        components['integrations'].append(f"src/{src_key}")
                    elif src_key.startswith('VI_'):
                        components['interface'].append(f"src/{src_key}")
                    elif src_key.startswith('U_'):
                        components['utility'].append(f"src/{src_key}")
                    else:
                        components['core'].append(f"src/{src_key}")
            elif key == 'data':
                components['data'].append('data')
            elif key == 'tests':
                components['tests'].append('tests')
            else:
                # Default categorization for other top-level directories
                lower_key = key.lower()
                if any(term in lower_key for term in ['test', 'spec']):
                    components['tests'].append(key)
                elif any(term in lower_key for term in ['data', 'assets', 'resources']):
                    components['data'].append(key)
                elif any(term in lower_key for term in ['ui', 'interface', 'web']):
                    components['interface'].append(key)
                elif any(term in lower_key for term in ['util', 'helper', 'tool']):
                    components['utility'].append(key)
                else:
                    components['core'].append(key)
        
        return components

    def find_dependencies(self, components: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Find dependencies between components based on imports"""
        dependencies = {comp: set() for comp_type in components.values() for comp in comp_type}
        
        # Analyze import relationships
        for module, imports in self.module_imports.items():
            # Get the component this module belongs to
            module_component = None
            for comp_type, comps in components.items():
                for comp in comps:
                    if module.startswith(comp):
                        module_component = comp
                        break
                if module_component:
                    break
            
            if not module_component:
                continue
                
            # For each import, find which component it belongs to
            for imported in imports:
                for comp_type, comps in components.items():
                    for comp in comps:
                        # Check if the import matches a component
                        comp_basename = os.path.basename(comp)
                        if imported == comp_basename or imported.lower() == comp_basename.lower():
                            if comp != module_component:  # Don't add self-references
                                dependencies[module_component].add(comp)
        
        return dependencies

    def read_repo_docs(self) -> str:
        """Read relevant documentation from the repository"""
        docs = []
        
        # Common documentation files to look for
        doc_files = [
            "README.md",
            "docs/README.md",
            "ARCHITECTURE.md",
            "docs/ARCHITECTURE.md",
            "DESIGN.md",
            "docs/DESIGN.md",
            "TODO.md",
            "TODO.txt"
        ]
        
        for doc_file in doc_files:
            file_path = self.project_root / doc_file
            if file_path.exists():
                print(f"Found documentation: {doc_file}")
                docs.append(f"# {doc_file}\n{file_path.read_text()}")
        
        if not docs:
            print("No documentation files found in standard locations.")
            
        return "\n\n---\n\n".join(docs)

    def prepare_graphviz_data(self, components: Dict[str, List[str]], dependencies: Dict[str, Set[str]]):
        """
        Prepare graph data for Graphviz visualization using neuron_graph style
        
        Args:
            components: Dictionary mapping component types to component names
            dependencies: Dependencies between components
        """
        # Clear existing graph data
        self.nodes = set()
        self.edges = set()
        self.node_attributes = {}
        
        # Add a central "repo" node
        central_node = "repository"
        self.nodes.add(central_node)
        self.node_attributes[central_node] = {
            'shape': 'circle',
            'style': 'filled',
            'fillcolor': self.base_colors['central'],
            'fontcolor': 'white',
            'width': '1.2',
            'height': '1.2',
            'central': 'true'
        }
        
        # Add component nodes by type
        for comp_type, comps in components.items():
            for comp in comps:
                # Use basename as node name for better readability
                name = os.path.basename(comp) if '/' in comp else comp
                self.nodes.add(name)
                
                # Set attributes based on component type
                if comp_type == 'core':
                    color = self.base_colors['primary']
                    size = '12'
                    # Link core components directly to central node
                    self.edges.add((central_node, name))
                elif comp_type in ['utility', 'integrations']:
                    color = self.base_colors['secondary']
                    size = '10'
                    # Utility components typically connect to central node
                    self.edges.add((central_node, name))
                else:
                    color = self.base_colors['tertiary']
                    size = '9'
                
                self.node_attributes[name] = {
                    'fillcolor': color,
                    'fontsize': size
                }
        
        # Add edges based on dependencies
        for source, targets in dependencies.items():
            source_name = os.path.basename(source) if '/' in source else source
            if source_name in self.nodes:
                for target in targets:
                    target_name = os.path.basename(target) if '/' in target else target
                    if target_name in self.nodes and source_name != target_name:
                        # Skip if we already have an edge from central to this node
                        if not ((source_name == central_node and target_name in self.nodes) or 
                                (target_name == central_node and source_name in self.nodes)):
                            self.edges.add((source_name, target_name))
        
        logger.info(f"Graph prepared with {len(self.nodes)} nodes and {len(self.edges)} edges")

    def create_graphviz_visualization(self, output_path: Optional[Path] = None) -> Optional[str]:
        """
        Generate a high-resolution radial visualization of the repository architecture.
        
        Args:
            output_path: Optional custom output path
            
        Returns:
            Path to the generated PNG file
        """
        try:
            if not output_path:
                output_path = self.output_dir / "repo_architecture_radial"
            
            # Create new graph with twopi layout
            dot = graphviz.Graph(
                'repo_architecture',
                comment='Repository Architecture Graph',
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

            # Add all nodes
            for node in self.nodes:
                attrs = self.node_attributes.get(node, {})
                # Format node labels for better readability
                label = node.replace('_', ' ').title()
                dot.node(node, label, **attrs)

            # Add all edges
            for source, target in self.edges:
                is_central_edge = source.lower() == 'repository' or target.lower() == 'repository'
                dot.edge(source, target,
                    color=self.base_colors['edge_primary'] if is_central_edge else self.base_colors['edge'],
                    penwidth=str(1.5 if is_central_edge else 1.0),
                    weight=str(2 if is_central_edge else 1)
                )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Render and save the graph
            dot.render(str(output_path), format='png', cleanup=True)
            
            logger.info(f"Radial visualization saved to: {output_path}.png")
            return str(output_path) + '.png'

        except Exception as e:
            logger.error(f"Failed to visualize architecture graph: {e}")
            return None

    def create_mermaid_visualization(self, mermaid_code: str):
        """Save Mermaid diagram code to file"""
        mermaid_path = self.output_dir / "architecture.mmd"
        mermaid_path.write_text(mermaid_code)
        print(f"Mermaid diagram saved to: {mermaid_path}")
        
        # Also save a clean version with just the mermaid code for rendering
        clean_code = self.extract_mermaid_code(mermaid_code)
        if clean_code:
            clean_path = self.output_dir / "architecture_clean.mmd"
            clean_path.write_text(clean_code)
            print(f"Clean Mermaid code saved to: {clean_path}")
    
    def extract_mermaid_code(self, text: str) -> str:
        """Extract mermaid code from text that might contain markdown blocks"""
        if "```mermaid" in text:
            try:
                code = text.split("```mermaid")[1].split("```")[0].strip()
                return code
            except IndexError:
                pass
        
        if "```" in text:
            try:
                code = text.split("```")[1].strip()
                if code.startswith("graph ") or code.startswith("flowchart "):
                    return code
            except IndexError:
                pass
        
        # If we can't extract it properly, return the original
        return text

    def generate_architecture_description(self, folder_structure: Dict, docs: str, components: Dict[str, List[str]]) -> str:
        """Generate a logical flow diagram of the repository architecture"""
        
        # Create a more precise input prompt with actual component names
        component_list = []
        for category, comps in components.items():
            if comps:
                component_list.append(f"{category.upper()} COMPONENTS:")
                for comp in comps:
                    component_list.append(f"- {comp}")
                component_list.append("")
        
        component_text = "\n".join(component_list)
        
        input_content = f"""
        Create a Mermaid diagram showing the logical architecture and relationships of this repository.
        
        IMPORTANT: Use ONLY the ACTUAL component names from the repository that I'll list below.
        DO NOT invent generic component names like "Core Processing" or "TextGen" - use the real module names.
        
        ACTUAL REPOSITORY COMPONENTS:
        {component_text}
        
        Key Requirements:
        1. The diagram MUST use the actual component names specified above, not generic or made-up components.
        2. Show ONLY main components and their relationships, keep it readable (max 15-20 components).
        3. Use colors to distinguish component types:
           - Blue (#3B82F6) for core components
           - Indigo (#1E3A8A) for utility components
           - Purple (#6D28D9) for data components
           - Light Blue (#60A5FA) for interface components
           - Cyan (#00BFFF) for integration components
        4. Show logical flow and dependencies between components
        
        Repository Documentation:
        {docs}
        
        Repository Structure:
        {json.dumps(folder_structure, indent=2)}
        
        Create a diagram that shows the logical flow and component relationships using the ACTUAL module names.
        Use flowchart TD and include appropriate styling.
        """

        # Get completion
        mermaid_code = self.api.chat_completion(
            user_prompt=input_content,
            system_prompt="""You are an expert software architect specializing in visualization.
            Create a Mermaid diagram based ONLY on the ACTUAL components listed in the repository structure.
            NEVER invent generic components - use only the real component names provided.
            Your diagram should accurately represent the repository's structure and dependencies,
            focusing on the most important components for clarity."""
        )
        
        # Post-process to check for made-up component names and fix them
        return self._validate_mermaid_diagram(mermaid_code, components)
        
    def _validate_mermaid_diagram(self, mermaid_code: str, components: Dict[str, List[str]]) -> str:
        """Validate and fix the mermaid diagram to ensure it uses actual component names"""
        # Extract clean mermaid code
        clean_code = self.extract_mermaid_code(mermaid_code)
        
        # Flatten component list for validation
        all_components = []
        for comp_list in components.values():
            all_components.extend(comp_list)
        
        # Extract node IDs from the mermaid diagram
        node_pattern = r'([A-Za-z0-9_]+)\s*\[([^\]]+)\]'
        nodes = re.findall(node_pattern, clean_code)
        
        # Check if any node label doesn't correspond to a real component
        valid_nodes = []
        invalid_nodes = []
        
        for node_id, node_label in nodes:
            # Clean up the label
            label = node_label.strip().replace('"', '').replace("'", "")
            
            # Check if this component name exists in our actual components
            is_valid = False
            for comp in all_components:
                comp_name = os.path.basename(comp)
                if label == comp or label == comp_name or comp in label:
                    is_valid = True
                    break
            
            if is_valid:
                valid_nodes.append((node_id, label))
            else:
                invalid_nodes.append((node_id, label))
        
        # If we found invalid nodes, warn about them
        if invalid_nodes:
            print("\nWarning: The generated diagram contains components that don't match the actual repository:")
            for node_id, label in invalid_nodes:
                print(f"- {label} (node ID: {node_id})")
            
            print("\nValid components from the repository:")
            for comp in all_components:
                print(f"- {comp}")
        
        return clean_code

    def visualize(self):
        """Main method to generate all visualizations"""
        print("Analyzing repository architecture...")
        
        try:
            # 1. Analyze folder structure
            folder_structure = self.analyze_folder_structure()
            
            # 2. Read repository documentation
            docs = self.read_repo_docs()
            
            # 3. Identify major components
            components = self.identify_major_components(folder_structure)
            
            # 4. Find dependencies between components
            dependencies = self.find_dependencies(components)
            
            # 5. Generate GraphViz visualization (neuron-graph style)
            print("\nGenerating radial architecture diagram...")
            self.prepare_graphviz_data(components, dependencies)
            graphviz_output = self.create_graphviz_visualization()
            if graphviz_output:
                print(f"Radial diagram created at: {graphviz_output}")
            
            # 6. Generate Mermaid diagram
            print("\nGenerating Mermaid architecture diagram...")
            mermaid_code = self.generate_architecture_description(folder_structure, docs, components)
            self.create_mermaid_visualization(mermaid_code)
            
            print(f"\nArchitecture visualizations saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    viz = RepoArchitectureViz()
    viz.visualize()