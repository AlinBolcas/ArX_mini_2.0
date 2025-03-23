from graphviz import Digraph
from pathlib import Path

def create_pipeline_diagram():
    """Generate a visual representation of the LLM-Introspector pipeline."""
    
    # Create a new directed graph
    dot = Digraph(comment='LLM-Introspector Pipeline', engine='dot')
    
    # Color scheme matching knowledge graph
    COLORS = {
        'root': '#00BFFF',      # Bright blue for core components
        'primary': '#3B82F6',   # Medium blue for primary nodes
        'secondary': '#1E3A8A', # Deep blue for secondary nodes
        'edge': '#ffffff80',    # Semi-transparent white
        'edge_primary': '#60A5FA90',  # Light blue for primary edges
        'bg': '#0A0A0F'         # Dark background
    }
    
    # Graph settings
    dot.attr(
        rankdir='LR',           # Left-to-right layout
        bgcolor=COLORS['bg'],
        fontname='Helvetica,Arial,sans-serif',
        fontcolor='white',
        fontsize='16',
        dpi='300',
        nodesep='0.3',         # Reduced separation
        ranksep='0.5',         # Reduced separation
        margin='0.2',          # Tighter margins
        pad='0.2'              # Minimal padding
    )
    
    # Node defaults
    dot.attr('node',
        shape='rect',
        style='filled,rounded',
        fontname='Helvetica,Arial,sans-serif',
        fontcolor='white',
        fontsize='11',
        margin='0.15',
        height='0.4',
        width='1.8',
        penwidth='0'
    )
    
    # Edge defaults
    dot.attr('edge',
        color=COLORS['edge_primary'],
        fontcolor='white',
        fontname='Helvetica,Arial,sans-serif',
        fontsize='9',
        penwidth='1.2'
    )

    # Create pipeline structure
    with dot.subgraph(name='cluster_pipeline') as pipeline:
        pipeline.attr(label='LLM-Introspector Pipeline', 
                     style='rounded', 
                     bgcolor=COLORS['secondary'],
                     margin='10')
        
        # Input node
        dot.node('input_llm',
            'Language Model\n(GPT-4, LLaMA, etc.)',
            shape='cylinder',
            fillcolor=COLORS['root'])

        # Main process nodes
        stages = ['form_gen', 'analysis', 'style', 'synthesis']
        stage_labels = {
            'form_gen': 'Form Generation\n---\nCognitive Assessment',
            'analysis': 'Self-Analysis\n---\nUnrestricted Examination',
            'style': 'Style Analysis\n---\nExpression Patterns',
            'synthesis': 'Profile Synthesis\n---\nPattern Analysis'
        }
        
        # Create nodes and connect them
        for stage in stages:
            pipeline.node(stage, stage_labels[stage], fillcolor=COLORS['primary'])
            # Connect LLM to each stage
            dot.edge('input_llm', stage, '')
        
        # Connect stages sequentially
        for i in range(len(stages)-1):
            pipeline.edge(stages[i], stages[i+1])

        # Output node
        dot.node('output_files',
            '''Output Files
            ---
            • form.json
            • answers.json
            • style.md
            • profile.md''',
            shape='folder',
            fillcolor=COLORS['primary'])
        
        # Connect final stage to output
        dot.edge('synthesis', 'output_files')

    # Save with specific engine settings
    output_dir = Path(__file__).parent.parent.parent / "output" / "data" / "personaGen"
    output_dir.mkdir(parents=True, exist_ok=True)
    dot.render(
        str(output_dir / "pipeline_diagram"),
        format='png',
        cleanup=True,
        engine='dot'
    )

if __name__ == "__main__":
    create_pipeline_diagram() 