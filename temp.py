from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt

def display_graph_simple(graph):
    """
    Display the graph using networkx and matplotlib
    Args:
        graph: The LangGraph object
    """
    try:
        # Get graph representation
        graph_data = graph.get_graph()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        nodes = set()
        edges = []
        
        # Extract nodes and edges
        for edge in graph_data.edges:
            nodes.add(edge.source)
            nodes.add(edge.target)
            edges.append((edge.source, edge.target))
        
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Use a deterministic layout
        pos = nx.kamada_kawai_layout(G)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, 
                             node_color='lightblue',
                             node_size=2000)
        
        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20)
        
        # Add node labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graph Workflow")
        plt.axis('off')
        
        # Display in notebook
        display(plt.gcf())
        plt.close()
        
    except Exception as e:
        print(f"Visualization error: {e}")
        # Fallback to text representation
        print("\nGraph Structure:")
        print("Nodes:", [n for n in nodes])
        print("Edges:", edges)

# Usage
try:
    display_graph_simple(graph)
except Exception as e:
    print(f"Could not display graph: {e}")
    # Fallback to basic text output
    print("Graph nodes:", graph.get_graph().nodes)
    print("Graph edges:", [(e.source, e.target) for e in graph.get_graph().edges])

from IPython.display import display
import networkx as nx
import matplotlib.pyplot as plt

def display_graph_vertical(graph):
    """
    Display the graph in a vertical layout with clear arrows
    Args:
        graph: The LangGraph object
    """
    try:
        # Get graph representation
        graph_data = graph.get_graph()
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Extract nodes and edges
        nodes = set()
        edges = []
        for edge in graph_data.edges:
            nodes.add(edge.source)
            nodes.add(edge.target)
            edges.append((edge.source, edge.target))
        
        nodes = list(nodes)
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create custom vertical layout
        pos = {}
        total_nodes = len(nodes)
        
        # Find start and end nodes
        start_nodes = {n for n in G.nodes() if G.in_degree(n) == 0}
        end_nodes = {n for n in G.nodes() if G.out_degree(n) == 0}
        
        # Main path nodes (nodes that aren't start or end)
        middle_nodes = list(set(nodes) - start_nodes - end_nodes)
        
        # Position nodes vertically
        y_spacing = 1.0 / (total_nodes + 1)
        
        # Position start nodes at top
        for i, node in enumerate(start_nodes):
            pos[node] = (0.5, 0.9)
        
        # Position middle nodes in center
        for i, node in enumerate(middle_nodes):
            y_pos = 0.9 - ((i + 1) * y_spacing * 2)
            
            # If it's a conditional node (has multiple outgoing edges)
            if G.out_degree(node) > 1:
                pos[node] = (0.5, y_pos)
                # Position its targets diagonally
                targets = list(G.successors(node))
                for j, target in enumerate(targets):
                    if target not in end_nodes:  # Don't position end nodes yet
                        x_offset = 0.3 * (-1 if j % 2 == 0 else 1)
                        pos[target] = (0.5 + x_offset, y_pos - y_spacing)
            else:
                pos[node] = (0.5, y_pos)
        
        # Position end nodes at bottom
        for i, node in enumerate(end_nodes):
            pos[node] = (0.5, 0.1)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             node_color='lightblue',
                             node_size=2000,
                             edgecolors='black')
        
        # Draw edges with curved arrows for conditional paths
        for edge in G.edges():
            if G.out_degree(edge[0]) > 1:  # Conditional edge
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[edge],
                                     connectionstyle=f'arc3, rad={0.3}',
                                     edge_color='red',
                                     arrows=True,
                                     arrowsize=20,
                                     width=2,
                                     arrowstyle='->',
                                     style='dashed')
            else:  # Regular edge
                nx.draw_networkx_edges(G, pos,
                                     edgelist=[edge],
                                     edge_color='gray',
                                     arrows=True,
                                     arrowsize=20,
                                     width=2,
                                     arrowstyle='->')
        
        # Add node labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Graph Workflow")
        plt.axis('off')
        
        # Display in notebook
        display(plt.gcf())
        plt.close()
        
    except Exception as e:
        print(f"Visualization error: {e}")
        # Fallback to text representation
        print("\nGraph Structure:")
        print("Nodes:", [n for n in nodes])
        print("Edges:", edges)

# Usage
try:
    display_graph_vertical(graph)
except Exception as e:
    print(f"Could not display graph: {e}")
    # Fallback to basic text output
    print("Graph nodes:", graph.get_graph().nodes)
    print("Graph edges:", [(e.source, e.target) for e in graph.get_graph().edges])

import networkx as nx
import matplotlib.pyplot as plt

def display_langgraph(graph):
    # Get graph data
    graph_data = graph.get_graph()
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add all nodes and edges from the data
    for node_id, node_data in graph_data.nodes.items():
        G.add_node(node_id)
    
    for edge in graph_data.edges:
        G.add_edge(edge.source, edge.target)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define fixed positions for better layout
    pos = {
        '__start__': (0.5, 1.0),
        'load_index': (0.5, 0.8),
        'get_input': (0.5, 0.2),
        'retrieval': (0.5, 0.6),
        'llm_chain': (0.5, 0.4),
        '__end__': (0.8, 0.3)  # End node slightly offset
    }
    
    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in ['__start__', '__end__']:
            node_colors.append('lightgray')
            node_sizes.append(1500)
        else:
            node_colors.append('lightblue')
            node_sizes.append(2000)
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=node_sizes)
    
    # Draw regular edges
    regular_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if not any(e.source == u and e.target == v and e.conditional 
                             for e in graph_data.edges)]
    
    # Draw conditional edges
    conditional_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if any(e.source == u and e.target == v and e.conditional 
                             for e in graph_data.edges)]
    
    # Draw regular edges with black solid lines
    nx.draw_networkx_edges(G, pos,
                          edgelist=regular_edges,
                          edge_color='black',
                          arrows=True,
                          arrowsize=20)
    
    # Draw conditional edges with red dashed lines
    nx.draw_networkx_edges(G, pos,
                          edgelist=conditional_edges,
                          edge_color='red',
                          style='dashed',
                          arrows=True,
                          arrowsize=20,
                          connectionstyle='arc3,rad=0.3')
    
    # Add labels
    labels = {node: node.replace('__', '') for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10)
    
    plt.title("LangGraph Workflow")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

# Usage
display_langgraph(graph)

