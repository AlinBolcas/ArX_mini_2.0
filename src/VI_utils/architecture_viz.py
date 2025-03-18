import os
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any, Set, Tuple
import sys
import re

# Set up project root and add to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import OpenAI API module
from src.I_integrations.openai_API import OpenAIAPI

class RepoArchitectureViz:
    def __init__(self, output_dir=None):
        self.project_root = project_root
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = project_root / "data" / "output" / "architecture_viz"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI API for architecture analysis
        self.api = OpenAIAPI(
            model="gpt-4o-mini",
            temperature=0.2
        )
        
        # Set dark theme for matplotlib
        plt.style.use('dark_background')
        
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
        
        # Store module imports for dependency analysis
        self.module_imports = {}
        
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
           - Blue (#1E90FF) for core components
           - Green (#32CD32) for utility components
           - Purple (#800080) for data components
           - Yellow (#FFD700) for interface and integration components
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

    def create_networkx_visualization(self, structure: Dict, components: Dict[str, List[str]], dependencies: Dict[str, Set[str]]):
        """Create a detailed network visualization of the repository architecture"""
        G = nx.DiGraph()
        
        # Add nodes from identified components
        node_types = {}
        
        for comp_type, comps in components.items():
            for comp in comps:
                # Get simplified name for the node (basename of the component path)
                name = os.path.basename(comp) if '/' in comp else comp
                G.add_node(name, type=comp_type, fullpath=comp)
                node_types[name] = comp_type
        
        # Add edges based on dependencies
        for source, targets in dependencies.items():
            source_name = os.path.basename(source) if '/' in source else source
            if source_name in G:
                for target in targets:
                    target_name = os.path.basename(target) if '/' in target else target
                    if target_name in G and source_name != target_name:
                        G.add_edge(source_name, target_name)
        
        # Add additional edges based on common patterns if the graph is sparse
        if len(G.edges()) < len(G.nodes()) / 2:
            self._add_logical_edges(G, node_types)
        
        # Set up layout - use force-directed layout for better visualization
        pos = nx.spring_layout(G, seed=42, k=0.5)
        
        # Set up the plot with dark theme
        plt.figure(figsize=(16, 10), facecolor='#1a1a1a')
        ax = plt.gca()
        ax.set_facecolor('#1a1a1a')
        
        # Define node colors by type
        node_colors = {
            'core': '#2196F3',      # Blue
            'utility': '#4CAF50',   # Green
            'data': '#9C27B0',      # Purple
            'interface': '#FFC107', # Yellow
            'integrations': '#FF9800', # Orange
            'tests': '#F44336'      # Red
        }
        
        # Draw nodes by type
        for node_type in set(node_types.values()):
            nodes = [n for n, t in node_types.items() if t == node_type]
            if nodes:
                nx.draw_networkx_nodes(G, pos,
                                     nodelist=nodes,
                                     node_color=node_colors.get(node_type, '#CCCCCC'),
                                     node_size=2000,
                                     alpha=0.9)
        
        # Draw edges with curved arrows
        nx.draw_networkx_edges(G, pos,
                              edge_color='#404040',
                              arrows=True,
                              arrowstyle='-|>',
                              arrowsize=15,
                              width=1.5,
                              alpha=0.7,
                              connectionstyle='arc3,rad=0.1')
        
        # Add labels with better positioning
        nx.draw_networkx_labels(G, pos,
                              font_size=9,
                              font_weight='bold',
                              font_color='white')
        
        # Add a title and a color legend
        plt.title("Repository Architecture Visualization", color='white', fontsize=16)
        
        # Add legend manually
        legend_elements = []
        import matplotlib.patches as mpatches
        
        # Create legend entries for each component type
        for comp_type, color in node_colors.items():
            if any(t == comp_type for t in node_types.values()):
                legend_elements.append(
                    mpatches.Patch(facecolor=color, edgecolor='white', alpha=0.7, label=comp_type.capitalize())
                )
        
        # Add legend to plot
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.7)
        
        # Save the visualization
        network_path = self.output_dir / 'repo_architecture.png'
        plt.savefig(
            network_path,
            facecolor='#1a1a1a',
            edgecolor='none',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        print(f"Network visualization saved to: {network_path}")
        
        # Create additional detailed visualization with module details
        self._create_detailed_visualization(structure, components)
    
    def _add_logical_edges(self, G, node_types):
        """Add logical edges based on common architectural patterns"""
        node_by_type = {}
        for node, node_type in node_types.items():
            if node_type not in node_by_type:
                node_by_type[node_type] = []
            node_by_type[node_type].append(node)
            
        # Common patterns:
        # 1. Core components depend on data components
        if 'core' in node_by_type and 'data' in node_by_type:
            for core_node in node_by_type['core']:
                for data_node in node_by_type['data']:
                    G.add_edge(core_node, data_node)
        
        # 2. Interface components depend on core components
        if 'interface' in node_by_type and 'core' in node_by_type:
            for interface_node in node_by_type['interface']:
                for core_node in node_by_type['core']:
                    G.add_edge(interface_node, core_node)
        
        # 3. Integration components connect to core components
        if 'integrations' in node_by_type and 'core' in node_by_type:
            for integration_node in node_by_type['integrations']:
                for core_node in node_by_type['core']:
                    G.add_edge(integration_node, core_node)
                    
    def _create_detailed_visualization(self, structure, components):
        """Create a more detailed visualization showing module organization"""
        # Create a new figure
        plt.figure(figsize=(20, 14), facecolor='#1a1a1a')
        ax = plt.gca()
        ax.set_facecolor('#1a1a1a')
        
        # Use a treemap-like visualization
        # First, define areas for different component types
        areas = [
            ('core', (0.1, 0.5, 0.35, 0.4)),  # (left, bottom, width, height)
            ('utility', (0.1, 0.1, 0.35, 0.3)),
            ('data', (0.55, 0.5, 0.35, 0.4)),
            ('interface', (0.55, 0.1, 0.2, 0.3)),
            ('integrations', (0.77, 0.1, 0.13, 0.3)),
            ('tests', (0.55, 0.05, 0.35, 0.03))
        ]
        
        # Draw component areas
        for component_type, rect in areas:
            left, bottom, width, height = rect
            color = {
                'core': '#1E4D8C',
                'utility': '#1C522A',
                'data': '#461D59',
                'interface': '#8C6D1F',
                'integrations': '#8C4A1D',
                'tests': '#8C1D1D'
            }.get(component_type, '#333333')
            
            rect = plt.Rectangle((left, bottom), width, height, facecolor=color, alpha=0.3, edgecolor='white', linewidth=2)
            ax.add_patch(rect)
            
            # Add area label
            ax.text(left + width / 2, bottom + height - 0.02, component_type.upper(), 
                   ha='center', va='top', color='white', fontsize=14, fontweight='bold')
            
            # Add components to the area
            if component_type in components and components[component_type]:
                comps = components[component_type]
                num_comps = len(comps)
                
                if num_comps > 0:
                    # Calculate grid dimensions
                    cols = min(4, num_comps)
                    rows = (num_comps + cols - 1) // cols
                    
                    # Draw components
                    for i, comp in enumerate(comps):
                        row = i // cols
                        col = i % cols
                        
                        # Calculate position
                        comp_width = width / cols
                        comp_height = (height - 0.07) / rows  # Leave space for area label
                        
                        comp_left = left + col * comp_width + 0.01
                        comp_bottom = bottom + (rows - row - 1) * comp_height + 0.02
                        
                        # Draw component box
                        comp_color = {
                            'core': '#2196F3',
                            'utility': '#4CAF50',
                            'data': '#9C27B0',
                            'interface': '#FFC107',
                            'integrations': '#FF9800',
                            'tests': '#F44336'
                        }.get(component_type, '#CCCCCC')
                        
                        comp_rect = plt.Rectangle(
                            (comp_left, comp_bottom), 
                            comp_width - 0.02, 
                            comp_height - 0.04, 
                            facecolor=comp_color, 
                            alpha=0.7,
                            edgecolor='white', 
                            linewidth=1
                        )
                        ax.add_patch(comp_rect)
                        
                        # Get component name (basename)
                        comp_name = os.path.basename(comp) if '/' in comp else comp
                        
                        # Add component label
                        ax.text(
                            comp_left + (comp_width - 0.02) / 2, 
                            comp_bottom + (comp_height - 0.04) / 2, 
                            comp_name, 
                            ha='center', 
                            va='center', 
                            color='white', 
                            fontsize=9,
                            fontweight='bold'
                        )
        
        # Remove axes
        plt.axis('off')
        plt.title("Repository Module Organization", color='white', fontsize=18, pad=20)
        
        # Save the detailed visualization
        detailed_path = self.output_dir / 'repo_detailed_architecture.png'
        plt.savefig(
            detailed_path,
            facecolor='#1a1a1a',
            edgecolor='none',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        
        print(f"Detailed visualization saved to: {detailed_path}")

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
            
            # 5. Generate Mermaid diagram
            print("\nGenerating architecture diagram...")
            mermaid_code = self.generate_architecture_description(folder_structure, docs, components)
            
            # 6. Create visualizations
            self.create_mermaid_visualization(mermaid_code)
            self.create_networkx_visualization(folder_structure, components, dependencies)
            
            print(f"\nArchitecture visualizations saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    viz = RepoArchitectureViz()
    viz.visualize()