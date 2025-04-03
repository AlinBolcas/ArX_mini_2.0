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
import base64
import urllib.request
import urllib.parse

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
            self.output_dir = self.project_root / "data" / "output" / "graphs"
        
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
        
        # Store file-level info and dependencies
        self.python_files = {}  # Maps file paths to their info
        self.file_dependencies = {}  # Maps file paths to their dependencies
        
    def analyze_folder_structure(self, ignore_dirs=None, max_depth=8, analyze_files=True) -> Dict[str, Any]:
        """
        Analyze the repository folder structure in detail
        
        Args:
            ignore_dirs: List of directory names to ignore
            max_depth: Maximum depth to analyze (default is 8)
            analyze_files: Whether to analyze individual files or just directories
        """
        if ignore_dirs is None:
            ignore_dirs = ['__pycache__', '.git', '.github', 'venv', 'env', 'node_modules', 'build', 'dist']
            
        structure = {}
        file_counts = {}
        file_types = {}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]
            
            # Skip directories with special prefixes
            if any(p.startswith('__') for p in Path(root).parts):
                continue
                
            # Calculate path relative to project root
            rel_path = Path(root).relative_to(self.project_root)
            
            # Check depth - skip if too deep
            if len(rel_path.parts) > max_depth and rel_path.parts[0] != '.':
                continue
                
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
                            rel_file_path = os.path.join(path_str, f)
                            try:
                                # Analyze detailed imports for file-level dependencies
                                self._analyze_file_imports(file_path, rel_file_path)
                                
                                # Also add to traditional module imports for backward compatibility
                                self._analyze_imports(file_path, path_str)
                            except Exception as e:
                                print(f"Error analyzing imports in {file_path}: {e}")
                
        # Add file statistics to the structure
        structure['_metadata'] = {
            'file_counts': file_counts,
            'file_types': file_types,
            'directory_count': len(file_counts),
            'total_files': sum(file_counts.values()),
            'python_files': len(self.python_files)
        }
        
        print(f"Analyzed {len(file_counts)} directories with {sum(file_counts.values())} files ({len(self.python_files)} Python files)")
        
        # After gathering all file imports, build file-level dependency graph
        if analyze_files:
            self._build_file_dependencies()
        
        return structure
    
    def _analyze_file_imports(self, file_path: str, rel_file_path: str):
        """Analyze detailed imports for a single Python file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Create file info dictionary
        self.python_files[rel_file_path] = {
            'path': rel_file_path,
            'name': os.path.basename(file_path),
            'imports': [],  # Will store all import statements
            'imported_by': [],  # Will be filled in during dependency resolution
            'module_name': os.path.splitext(os.path.basename(file_path))[0],
            'content_length': len(content.splitlines())
        }
        
        # Match standard imports - both 'import x' and 'from x import y'
        import_pattern = r'(?:from\s+([\w.]+)\s+import\s+[\w.,\s*]+)|(?:import\s+([\w.,\s]+))'
        
        for match in re.finditer(import_pattern, content):
            from_import, regular_import = match.groups()
            
            if from_import:
                self.python_files[rel_file_path]['imports'].append(from_import)
            
            if regular_import:
                # Handle multiple imports like "import os, sys, re"
                for imp in regular_import.split(','):
                    # Extract base package name, handling 'as' aliases
                    base_import = imp.strip().split(' as ')[0]
                    self.python_files[rel_file_path]['imports'].append(base_import)
    
    def _build_file_dependencies(self):
        """Build detailed file-level dependency graph based on imports"""
        print("Building file-level dependency graph...")
        file_count = len(self.python_files)
        dep_count = 0
        
        # First pass: set up empty dependency lists
        for file_path in self.python_files:
            self.file_dependencies[file_path] = set()
        
        # Second pass: resolve imports to actual files
        for file_path, file_info in self.python_files.items():
            for imported in file_info['imports']:
                # Skip standard library and external imports
                if imported in ['os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random',
                               'numpy', 'pandas', 'matplotlib', 'tensorflow', 'torch', 'sklearn',
                               'pathlib', 'typing', 'collections', 'functools']:
                    continue
                
                # Try to match with other Python files in the repo
                for target_path, target_info in self.python_files.items():
                    if file_path == target_path:  # Skip self-dependencies
                        continue
                    
                    # Check if the import matches this file's module name
                    if (imported == target_info['module_name'] or 
                        imported.endswith('.' + target_info['module_name']) or
                        target_path.endswith('/' + imported + '.py')):
                        
                        # Add dependency
                        self.file_dependencies[file_path].add(target_path)
                        target_info['imported_by'].append(file_path)
                        dep_count += 1
                        break
        
        print(f"Found {dep_count} file-level dependencies among {file_count} Python files")
    
    def _analyze_imports(self, file_path: str, module_path: str):
        """Analyze Python file imports to build detailed dependency graph"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Track imports for this module
        imports = set()
        deep_imports = set()  # Track exact imports with submodules
        
        # Match standard imports - both 'import x' and 'from x import y'
        import_pattern = r'(?:from\s+([\w.]+)\s+import\s+[\w.,\s*]+)|(?:import\s+([\w.,\s]+))'
        
        for match in re.finditer(import_pattern, content):
            from_import, regular_import = match.groups()
            
            if from_import:
                # Add full import path
                deep_imports.add(from_import)
                # Also add each level of the import for hierarchical dependencies
                parts = from_import.split('.')
                for i in range(1, len(parts)+1):
                    imports.add('.'.join(parts[:i]))
            
            if regular_import:
                # Handle multiple imports like "import os, sys, re"
                for imp in regular_import.split(','):
                    # Extract base package name, handling 'as' aliases
                    base_import = imp.strip().split(' as ')[0]
                    imports.add(base_import)
                    deep_imports.add(base_import)
                    
                    # Add each level of the import
                    parts = base_import.split('.')
                    for i in range(1, len(parts)+1):
                        imports.add('.'.join(parts[:i]))
        
        # Store the imports for this module
        module_name = os.path.basename(file_path)
        self.module_imports[f"{module_path}/{module_name}"] = {
            'basic': list(imports),
            'deep': list(deep_imports)
        }
    
    def create_file_level_visualization(self):
        """Generate visualization of file-level dependencies in a layered approach"""
        print("Generating file-level dependency visualization...")
        
        # Create new graph with dot layout
        dot = graphviz.Digraph(
            'file_dependencies',
            comment='File-level Dependency Graph',
            engine='dot'
        )
        
        # Graph attributes - using LR (left to right) direction
        dot.attr(
            rankdir='LR',
            ranksep='2.0',
            nodesep='0.7',
            bgcolor='#0A0A0F',
            fontname='Helvetica,Arial,sans-serif',
            dpi='300',
            overlap='false',
            splines='polyline'  # Use straight lines with bends
        )
        
        # Node defaults
        dot.attr('node',
            shape='box',
            style='filled,rounded',
            fillcolor=self.base_colors['secondary'],
            fontcolor='white',
            fontname='Helvetica,Arial,sans-serif',
            fontsize='11',
            margin='0.1',
            height='0.6',
            width='1.5',
            penwidth='0'
        )
        
        # Edge defaults
        dot.attr('edge',
            color=self.base_colors['edge'],
            fontcolor='white',
            fontname='Helvetica,Arial,sans-serif',
            fontsize='9',
            penwidth='1.0',
            arrowsize='0.6'
        )
        
        # Group files by directory structure categories (similar to the image)
        categories = {
            "Core": [],
            "Integrations": [],
            "Data": [],
            "Tools": [],
            "Knowledge": [],
            "UI": [],
            "Pipelines": [],
            "Utils": [],
            "Other": []
        }
        
        # Map each file to a category based on path
        for file_path in self.python_files:
            if "src/I_integrations" in file_path:
                categories["Integrations"].append(file_path)
            elif "src/II_data" in file_path or "data" in file_path:
                categories["Data"].append(file_path)
            elif "src/III_tools" in file_path or "tools" in file_path:
                categories["Tools"].append(file_path)
            elif "src/IV_pipelines" in file_path:
                categories["Pipelines"].append(file_path)
            elif "src/V_knowledge" in file_path:
                categories["Knowledge"].append(file_path)
            elif "src/VI_UI" in file_path or "interface" in file_path or "ui" in file_path:
                categories["UI"].append(file_path)
            elif "src/U_utils" in file_path or "utils" in file_path or "helpers" in file_path:
                categories["Utils"].append(file_path)
            elif "src" in file_path and not any(p in file_path for p in ["integrations", "data", "tools", "pipelines", "knowledge", "UI", "utils"]):
                categories["Core"].append(file_path)
            else:
                categories["Other"].append(file_path)
        
        # Create invisible rank nodes to enforce layering (left to right flow)
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_integrations', style='invis', shape='point')
            
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_pipelines', style='invis', shape='point')
        
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_tools', style='invis', shape='point')
            
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_core', style='invis', shape='point')
            
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_data', style='invis', shape='point')
            
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_knowledge', style='invis', shape='point')
            
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('layer_ui', style='invis', shape='point')
        
        # Connect the invisible nodes to enforce left-right flow
        dot.edge('layer_integrations', 'layer_pipelines', style='invis')
        dot.edge('layer_pipelines', 'layer_tools', style='invis')
        dot.edge('layer_tools', 'layer_core', style='invis')
        dot.edge('layer_core', 'layer_data', style='invis')
        dot.edge('layer_data', 'layer_knowledge', style='invis')
        dot.edge('layer_knowledge', 'layer_ui', style='invis')
        
        # Add files by category
        category_colors = {
            "Core": "#3B82F6",      # Blue
            "Integrations": "#00BFFF", # Light blue
            "Data": "#6D28D9",      # Purple
            "Tools": "#9333EA",     # Bright purple
            "Knowledge": "#1E3A8A", # Dark blue
            "UI": "#F97316",        # Orange
            "Pipelines": "#10B981", # Green
            "Utils": "#A855F7",     # Light purple
            "Other": "#6B7280"      # Gray
        }
        
        # Create clusters for each category with consistent positioning
        category_order = [
            ("Integrations", "layer_integrations"), 
            ("Pipelines", "layer_pipelines"),
            ("Tools", "layer_tools"), 
            ("Core", "layer_core"),
            ("Data", "layer_data"), 
            ("Knowledge", "layer_knowledge"),
            ("UI", "layer_ui")
        ]
        
        for category, layer_node in category_order:
            if not categories[category]:
                continue
                
            # Create a cluster for this category
            with dot.subgraph(name=f"cluster_{category}") as c:
                c.attr(
                    label=category,
                    style='filled',
                    fillcolor='#1a1a2e',
                    fontcolor=category_colors[category],
                    fontsize='16',
                    color='#333355'
                )
                
                # Make this category align with its layer node
                c.node(layer_node, style='invis', shape='point')
                
                # Add files in this category
                for file_path in categories[category]:
                    filename = os.path.basename(file_path)
                    # Color based on number of dependencies
                    dep_count = len(self.file_dependencies.get(file_path, set()))
                    imp_count = len(self.python_files[file_path].get('imported_by', []))
                    
                    # Use diamond shape for core components
                    shape = 'diamond' if category in ["Core", "Integrations", "Pipelines"] else 'ellipse'
                    
                    c.node(
                        file_path, 
                        label=filename,
                        fillcolor=category_colors[category],
                        shape=shape,
                        tooltip=f"{file_path}: Imports {dep_count}, Imported by {imp_count}"
                    )
        
        # Add Utils and Other categories without fixed position
        for category in ["Utils", "Other"]:
            if not categories[category]:
                continue
                
            with dot.subgraph(name=f"cluster_{category}") as c:
                c.attr(
                    label=category,
                    style='filled',
                    fillcolor='#1a1a2e',
                    fontcolor=category_colors[category],
                    fontsize='16',
                    color='#333355'
                )
                
                for file_path in categories[category]:
                    filename = os.path.basename(file_path)
                    c.node(
                        file_path, 
                        label=filename,
                        fillcolor=category_colors[category],
                        tooltip=f"{file_path}"
                    )
        
        # Add edges for dependencies
        for source_file, target_files in self.file_dependencies.items():
            for target_file in target_files:
                dot.edge(source_file, target_file)
        
        # Save the visualization
        output_path = self.output_dir / "file_dependencies"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dot.render(str(output_path), format='png', cleanup=True)
        
        print(f"File-level dependency visualization saved to: {output_path}.png")
        return str(output_path) + '.png'

    def render_mermaid_to_png(self, mermaid_code: str) -> Optional[str]:
        """
        Render the Mermaid diagram to a PNG file using the Mermaid.ink web service API
        
        Returns:
            Path to the generated PNG file, or None if rendering failed
        """
        try:
            # Clean up any potential invalid characters and ensure proper encoding
            mermaid_code = mermaid_code.strip()
            
            # First attempt - save to a simple HTML file that can be viewed in browser
            html_file = self.output_dir / "architecture_diagram.html"
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Repository Architecture</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
  <style>
    body {{ background-color: #0A0A0F; color: white; font-family: Arial, sans-serif; padding: 20px; }}
    .mermaid {{ background-color: #0A0A0F; }}
    h1 {{ color: #3B82F6; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .info {{ margin-bottom: 20px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Repository Architecture Diagram</h1>
    <div class="info">
      <p>This diagram shows the logical structure and relationships between components in the repository.</p>
      <p>To save as an image, right-click on the diagram and select "Save image as..."</p>
    </div>
    <div class="mermaid">
{mermaid_code}
    </div>
  </div>
  <script>
    mermaid.initialize({{
      startOnLoad: true,
      theme: 'dark',
      themeVariables: {{
        primaryColor: '#3B82F6',
        primaryTextColor: '#ffffff', 
        primaryBorderColor: '#1E3A8A',
        lineColor: '#ffffff90',
        secondaryColor: '#6D28D9',
        tertiaryColor: '#00BFFF',
        background: '#0A0A0F'
      }}
    }});
  </script>
</body>
</html>
            """
            
            html_file.write_text(html_content)
            print(f"Mermaid diagram HTML file created for browser viewing: {html_file}")
            
            # Second attempt - Direct API use with simpler approach
            try:
                # Create a simplified Mermaid diagram in SVG format
                # Use a different API that's more reliable
                svg_url = f"https://mermaid.ink/svg/{base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')}"
                output_file = self.output_dir / "architecture_diagram.svg"
                
                # Download the SVG
                urllib.request.urlretrieve(svg_url, output_file)
                print(f"Mermaid diagram rendered to SVG: {output_file}")
                
                # Try to download PNG as well
                png_url = f"https://mermaid.ink/img/{base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')}"
                png_output = self.output_dir / "architecture_diagram.png"
                
                try:
                    urllib.request.urlretrieve(png_url, png_output)
                    print(f"Mermaid diagram rendered to PNG: {png_output}")
                    return str(png_output)
                except Exception as png_e:
                    print(f"Note: Could not download PNG directly, but SVG is available: {e}")
                    return str(output_file)
                    
            except Exception as e:
                print(f"Could not render Mermaid diagram to SVG/PNG: {e}")
                print(f"HTML version is still available at: {html_file}")
                return None
                    
        except Exception as e:
            logger.error(f"Failed to create any Mermaid visualization: {e}")
            return None

    def visualize(self):
        """Main method to generate all visualizations"""
        # Hard-coded depth for repository analysis
        analyze_depth = 8
        print(f"Analyzing repository architecture (depth: {analyze_depth})...")
        
        try:
            # 1. Analyze folder structure and files
            folder_structure = self.analyze_folder_structure(max_depth=analyze_depth, analyze_files=True)
            
            # 2. Read repository documentation
            docs = self.read_repo_docs()
            
            # 3. Create file-level visualization
            print("\nGenerating file-level dependency diagram...")
            file_viz_output = self.create_file_level_visualization()
            if file_viz_output:
                print(f"File-level dependency diagram created at: {file_viz_output}")
            
            # 4. Generate traditional directory-level visualizations
            
            # 4a. Identify major components
            components = self.identify_major_components(folder_structure)
            
            # 4b. Find dependencies between components
            dependencies = self.find_dependencies(components)
            
            # 4c. Generate GraphViz radial visualization
            print("\nGenerating radial architecture diagram...")
            self.prepare_graphviz_data(components, dependencies)
            graphviz_output = self.create_graphviz_visualization()
            if graphviz_output:
                print(f"Radial diagram created at: {graphviz_output}")
            
            # 4d. Generate Mermaid diagram
            print("\nGenerating Mermaid architecture diagram...")
            mermaid_code = self.generate_architecture_description(folder_structure, docs, components)
            self.create_mermaid_visualization(mermaid_code)
            
            # 4e. Generate flat visualization
            print("\nGenerating flat architecture diagram...")
            flat_output_path = self.output_dir / "repo_architecture_flat"
            # Create a flat diagram using dot engine instead of twopi
            dot = graphviz.Graph(
                'repo_architecture_flat',
                comment='Repository Architecture Graph (Flat)',
                engine='dot'
            )
            
            # Set basic attributes for flat layout
            dot.attr(
                rankdir='LR',
                ranksep='1.5',
                nodesep='0.8',
                bgcolor='#0A0A0F',
                fontname='Helvetica,Arial,sans-serif',
                dpi='300'
            )
            
            # Use the same node and edge attributes as the radial diagram
            dot.attr('node',
                shape='box',
                style='filled,rounded',
                fillcolor=self.base_colors['secondary'],
                fontcolor='white',
                fontname='Helvetica,Arial,sans-serif',
                fontsize='11',
                margin='0.2',
                penwidth='0'
            )
            
            dot.attr('edge',
                color=self.base_colors['edge'],
                fontcolor='white',
                fontname='Helvetica,Arial,sans-serif',
                fontsize='9',
                penwidth='1.2',
                arrowsize='0.6'
            )
            
            # Add nodes and edges from the prepared data
            for node in self.nodes:
                attrs = self.node_attributes.get(node, {})
                label = node.replace('_', ' ').title()
                dot.node(node, label, **attrs)
                
            for source, target in self.edges:
                is_central_edge = source.lower() == 'repository' or target.lower() == 'repository'
                dot.edge(source, target,
                    color=self.base_colors['edge_primary'] if is_central_edge else self.base_colors['edge'],
                    penwidth=str(1.5 if is_central_edge else 1.0),
                    weight=str(2 if is_central_edge else 1)
                )
                
            # Render the flat diagram
            dot.render(str(flat_output_path), format='png', cleanup=True)
            print(f"Flat diagram created at: {flat_output_path}.png")
            
            print(f"\nAll architecture visualizations saved to: {self.output_dir}")
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
        """Find detailed dependencies between components based on imports"""
        dependencies = {comp: set() for comp_type in components.values() for comp in comp_type}
        
        # Create a map of module patterns to their components
        module_to_component = {}
        for comp_type, comps in components.items():
            for comp in comps:
                # Add the main component
                module_to_component[comp] = comp
                
                # Add basename matching for easier detection
                basename = os.path.basename(comp) if '/' in comp else comp
                module_to_component[basename] = comp
                
                # Add lowercase version for case-insensitive matching
                module_to_component[basename.lower()] = comp
        
        # Analyze import relationships
        for module, import_data in self.module_imports.items():
            # Extract the basic imports and deep imports
            imports = import_data.get('basic', []) if isinstance(import_data, dict) else import_data
            deep_imports = import_data.get('deep', []) if isinstance(import_data, dict) else []
            
            # Get the component this module belongs to
            module_component = None
            module_path = os.path.dirname(module)
            
            # Try to find the containing component
            for comp in components.values():
                for c in comp:
                    if module.startswith(c) or module_path.startswith(c):
                        module_component = c
                        break
                if module_component:
                    break
            
            if not module_component:
                continue
                
            # For each import, find which component it belongs to
            processed_imports = set()
            
            # First process deep imports for more precise matching
            for imported in deep_imports:
                # Skip if already processed
                if imported in processed_imports:
                    continue
                
                imported_parts = imported.split('.')
                # Try different combinations of the import path
                for i in range(len(imported_parts), 0, -1):
                    partial_import = '.'.join(imported_parts[:i])
                    
                    # Check if this import corresponds to a component
                    target_comp = None
                    
                    # Direct match in our mapping
                    if partial_import in module_to_component:
                        target_comp = module_to_component[partial_import]
                    else:
                        # Try to match against component basenames
                        for comp in components.values():
                            for c in comp:
                                basename = os.path.basename(c)
                                if partial_import == basename or partial_import.endswith(f".{basename}"):
                                    target_comp = c
                                    break
                            if target_comp:
                                break
                    
                    if target_comp and target_comp != module_component:
                        dependencies[module_component].add(target_comp)
                        processed_imports.add(imported)
                        break
            
            # Then process basic imports for fallback
            for imported in imports:
                if imported in processed_imports:
                    continue
                
                # Skip standard library and common packages
                if imported in ['os', 'sys', 're', 'json', 'time', 'datetime', 'math', 'random']:
                    continue
                
                # Check each component
                for comp_type, comps in components.items():
                    for comp in comps:
                        # Check if the import matches a component
                        comp_basename = os.path.basename(comp)
                        if (imported == comp_basename or 
                            imported.lower() == comp_basename.lower() or
                            imported.endswith(comp_basename)):
                            
                            if comp != module_component:  # Don't add self-references
                                dependencies[module_component].add(comp)
                                processed_imports.add(imported)
        
        # Log dependency stats
        total_deps = sum(len(deps) for deps in dependencies.values())
        logger.info(f"Found {total_deps} dependencies between {len(dependencies)} components")
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
            'width': '1.5',
            'height': '1.5',
            'central': 'true'
        }
        
        # Track nodes by type for organized layout
        node_types = {
            'core': [],
            'utility': [],
            'integrations': [],
            'interface': [],
            'data': [],
            'tests': []
        }
        
        # Add component nodes by type
        for comp_type, comps in components.items():
            for comp in comps:
                # Use basename as node name for better readability
                name = os.path.basename(comp) if '/' in comp else comp
                self.nodes.add(name)
                
                # Add to appropriate type category
                if comp_type in node_types:
                    node_types[comp_type].append(name)
                
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
                    'fontsize': size,
                    'type': comp_type
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
        
        # Store node type information for layout
        self.node_types = node_types
        
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
                ranksep='3.0',  # Increased spacing between ranks
                overlap='false',
                splines='curved',
                bgcolor='#0A0A0F',
                fontname='Helvetica,Arial,sans-serif',
                dpi='300',
                pack='true',    # Pack disconnected components closer
                K='1.0',        # Optimal edge length
                start='random'  # Start with a random layout initially
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
                fixedsize='true',  # Consistent node sizes
                width='1.0',       # Default width
                height='0.7'       # Default height
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

            # Add repository node first (central node)
            for node in self.nodes:
                if node.lower() == 'repository':
                    attrs = self.node_attributes.get(node, {})
                    dot.node(node, 'Repository', **attrs)
            
            # Add nodes by type to organize them better
            for type_name, nodes_of_type in self.node_types.items():
                # Skip empty types
                if not nodes_of_type:
                    continue
                    
                # Add each node of this type
                for node in nodes_of_type:
                    if node.lower() == 'repository':
                        continue  # Skip central node, already added
                        
                    attrs = self.node_attributes.get(node, {})
                    
                    # Format node labels for better readability
                    label = node.replace('_', ' ').title()
                    
                    # Set different size based on type
                    if type_name == 'core':
                        width = '1.2'
                        height = '0.8'
                    elif type_name in ['utility', 'integrations']:
                        width = '1.0'
                        height = '0.7'
                    else:
                        width = '0.9'
                        height = '0.6'
                        
                    attrs['width'] = width
                    attrs['height'] = height
                    attrs['group'] = type_name  # Group by type for layout
                    
                    dot.node(node, label, **attrs)
            
            # Add all edges
            for source, target in self.edges:
                is_central_edge = source.lower() == 'repository' or target.lower() == 'repository'
                
                # Attributes based on edge type
                edge_attrs = {
                    'color': self.base_colors['edge_primary'] if is_central_edge else self.base_colors['edge'],
                    'penwidth': str(1.5 if is_central_edge else 1.0),
                    'weight': str(2 if is_central_edge else 1)
                }
                
                # Add constraint for central edges to enforce radial layout
                if is_central_edge:
                    edge_attrs['constraint'] = 'true'
                
                dot.edge(source, target, **edge_attrs)

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
        """Save Mermaid diagram code to file and render to PNG"""
        mermaid_path = self.output_dir / "architecture.mmd"
        mermaid_path.write_text(mermaid_code)
        print(f"Mermaid diagram code saved to: {mermaid_path}")
        
        # Also save a clean version with just the mermaid code for rendering
        clean_code = self.extract_mermaid_code(mermaid_code)
        if clean_code:
            clean_path = self.output_dir / "architecture_clean.mmd"
            clean_path.write_text(clean_code)
            print(f"Clean Mermaid code saved to: {clean_path}")
            
            # Generate PNG from Mermaid diagram
            png_path = self.render_mermaid_to_png(clean_code)
            if png_path:
                print(f"Mermaid diagram rendered to PNG: {png_path}")
    
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

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualization with output to data/output/graphs
    output_dir = project_root.parent / "data" / "output" / "graphs"
    viz = RepoArchitectureViz(output_dir=output_dir)
    
    # Run visualization
    viz.visualize()