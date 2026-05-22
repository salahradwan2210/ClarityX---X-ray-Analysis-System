import pydot
import os
import platform
import sys

print("Generating chest X-ray system flowchart...")

# Create a new graph
graph = pydot.Dot(graph_type='digraph', format='png')

# Add nodes
nodes = {
    'A': 'Doctor',
    'B': 'Machine Learning',
    'C': 'New User',
    'D': 'Register',
    'E': 'Database',
    'F': 'Existing User',
    'G': 'Sign In',
    'H': 'Dashboard',
    'I': 'Add Patient',
    'J': 'Patient Database',
    'K': 'Patient Form',
    'L': 'Patient Details',
    'M': 'Upload X-ray',
    'N': 'Analyze Image',
    'O': 'Trained Model',
    'P': 'View History',
    'Q': 'Results Processing',
    'R': 'Display Results',
    'S': 'Generate PDF',
    'T': 'Download PDF Report',
    'U': 'No Finding Check (100%)'
}

# Add all nodes to the graph
for node_id, node_label in nodes.items():
    graph.add_node(pydot.Node(node_id, label=node_label))

# Add edges
edges = ['CD', 'DE', 'FG', 'GH', 'HI', 'IJ', 'IK', 'JL', 'LM', 'MN', 'NO', 'OQ', 'JP', 'QR', 'RS', 'ST', 'RU']

for edge in edges:
    graph.add_edge(pydot.Edge(edge[0], edge[1]))

# Write the graph to a file
try:
    output_file = "chest_xray_system.png"
    graph.write_png(output_file)
    print(f"Graph has been saved as '{output_file}'")
    
    # Open the file if it exists and has content
    if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
        if platform.system() == 'Windows':
            os.system(f'start {output_file}')
        elif platform.system() == 'Darwin':  # macOS
            os.system(f'open {output_file}')
        elif platform.system() == 'Linux':
            os.system(f'xdg-open {output_file}')
    else:
        print("\nWarning: The generated file seems too small or doesn't exist.")
        print("This likely means that Graphviz is not properly installed or not in your PATH.")
        print("\nTo fix this issue:")
        print("1. Make sure Graphviz is installed (you installed it with winget).")
        print("2. Add the Graphviz 'bin' directory to your system PATH.")
        print("3. Restart your terminal or computer.")
        print("\nTypical Graphviz bin paths:")
        print(" - Windows: C:\\Program Files\\Graphviz\\bin")
        print(" - macOS: /usr/local/bin (if installed with Homebrew)")
        print(" - Linux: /usr/bin")
        
except Exception as e:
    print(f"Error generating the graph: {str(e)}")
    print("\nThis error likely means that Graphviz is not properly installed or not in your PATH.")
    print("\nTo fix this issue:")
    print("1. Make sure Graphviz is installed (you installed it with winget).")
    print("2. Add the Graphviz 'bin' directory to your system PATH.")
    print("3. Restart your terminal or computer.")
    print("\nTypical Graphviz bin paths:")
    print(" - Windows: C:\\Program Files\\Graphviz\\bin")
    print(" - macOS: /usr/local/bin (if installed with Homebrew)")
    print(" - Linux: /usr/bin")