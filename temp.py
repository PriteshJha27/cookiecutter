import openai
import networkx as nx
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = "your-api-key"

def extract_triples_gpt4(input_text):
    """
    Extract entities and relationships from text using GPT-4.
    
    Example prompt: 
        Extract entities and relationships from the following text.
        Present the output in a triplet format as (Entity1, Relationship, Entity2).
        Use concise and clear relationships. Do not repeat entities or relationships unnecessarily.

        Example Input:
        "Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year."
        
        Example Output:
        1. (Visa, achieved, $8.9 billion in revenue)
        2. (Visa, grew by, 10%)
        3. (Visa, performance, Q3 2024)

    Args:
        input_text (str): Input text to extract triples from.
    
    Returns:
        list of tuples: A list of extracted triples in the form (Entity1, Relationship, Entity2).
    """
    prompt = f"""
    Extract entities and relationships from the following text. 
    Present the output in a triplet format: (Entity1, Relationship, Entity2). 
    Use concise and clear relationships. Do not repeat entities or relationships unnecessarily.
    
    Input: {input_text}
    """
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=200,
        temperature=0
    )
    triplets_text = response.choices[0].text.strip()
    
    # Parse triplets into a list of tuples
    triplets = []
    for line in triplets_text.split("\n"):
        if line.startswith("(") and line.endswith(")"):
            triplets.append(eval(line))  # Convert string to tuple
    return triplets

def create_knowledge_graph(triplets):
    """
    Create a knowledge graph using NetworkX.

    Args:
        triplets (list of tuples): List of triples in the form (Entity1, Relationship, Entity2).
    
    Returns:
        networkx.DiGraph: Directed graph of the entities and relationships.
    """
    G = nx.DiGraph()
    for sub, rel, obj in triplets:
        G.add_node(sub, label='Entity')
        G.add_node(obj, label='Entity')
        G.add_edge(sub, obj, label=rel)
    return G

def visualize_knowledge_graph(G):
    """
    Visualize the knowledge graph.

    Args:
        G (networkx.DiGraph): Directed graph of entities and relationships.
    """
    pos = nx.spring_layout(G)  # Layout for nodes
    edge_labels = nx.get_edge_attributes(G, 'label')
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=18, font_weight='bold', node_size=3000, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Knowledge Graph Visualization")
    plt.show()

# Example usage
if __name__ == "__main__":
    input_text = "Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year."
    
    # Step 1: Extract triplets using GPT-4
    triplets = extract_triples_gpt4(input_text)
    
    # Step 2: Create a knowledge graph
    G = create_knowledge_graph(triplets)
    
    # Step 3: Visualize the knowledge graph
    visualize_knowledge_graph(G)




# ----------------------------------------------------------------------------------------------------------------------------------------------


def add_theme_to_graph(G, triplets_with_themes):
    """
    Add themes as edge attributes to the knowledge graph.

    Args:
        G (networkx.DiGraph): The knowledge graph.
        triplets_with_themes (list of tuples): List of triples with themes, in the form
            [(Entity1, Relationship, Entity2, Theme), ...].
    """
    for triplet in triplets_with_themes:
        sub, rel, obj, theme = triplet
        if G.has_edge(sub, obj):
            G[sub][obj]['theme'] = theme  # Add theme to existing edge
        else:
            G.add_edge(sub, obj, label=rel, theme=theme)  # Create edge with theme

# Example usage
if __name__ == "__main__":
    # Example triplets with themes
    triplets_with_themes = [
        ("Visa", "achieved", "$8.9 billion in revenue", "Financial Performance"),
        ("Visa", "partners with", "ICICI Bank", "Growth Strategies"),
        ("Visa", "faces", "Regulatory Challenges", "Risks Identified")
    ]

    # Create a new graph and add themes
    G = nx.DiGraph()
    add_theme_to_graph(G, triplets_with_themes)

    # Display the graph edges with themes
    for edge in G.edges(data=True):
        print(edge)




# ----------------------------------------------------------------------------------------------------------------------------------------------


import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Define the data (entities, relationships, themes)
triplets_with_themes = [
    ("Visa", "achieved", "$8.9 billion in revenue", "Financial Performance"),
    ("Visa", "partners with", "ICICI Bank", "Growth Strategies"),
    ("Visa", "faces", "Regulatory Challenges", "Risk Identified"),
    ("Visa Direct", "grew by", "10%", "Financial Performance"),
    ("Consumers", "increased", "Tap-to-Pay adoption", "Consumer Trends"),
    ("Visa", "launched", "AI-driven payment solutions", "Innovation and Technology"),
    ("Visa Direct", "enables", "cross-border payments", "Growth Strategies"),
    ("Visa", "expanded to", "Asia", "Growth Strategies"),
]

# Step 2: Build the Knowledge Graph
def build_knowledge_graph(triplets_with_themes):
    G = nx.DiGraph()  # Create a directed graph
    for entity1, relation, entity2, theme in triplets_with_themes:
        # Add nodes with type attribute
        G.add_node(entity1, type="Entity")
        G.add_node(entity2, type="Entity")
        # Add edge and include the relationship and theme as attributes
        G.add_edge(entity1, entity2, relation=relation, theme=theme)
    return G

# Step 3: Visualize the Knowledge Graph
def visualize_knowledge_graph(G):
    pos = nx.spring_layout(G)  # Generate positions for nodes
    edge_labels = nx.get_edge_attributes(G, "relation")  # Get edge labels (relation)
    plt.figure(figsize=(12, 8))  # Set figure size
    nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=10, node_size=3000, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Knowledge Graph Visualization")
    plt.show()

# Step 4: Query Functions
def find_entities_by_theme(G, theme):
    """
    Find all unique entities associated with a given theme.
    """
    entities = set()
    for u, v, data in G.edges(data=True):
        if data.get("theme") == theme:
            entities.add(u)
            entities.add(v)
    return entities

def find_shortest_path(G, start, end):
    """
    Find the shortest relationship path between two entities.
    """
    try:
        path = nx.shortest_path(G, source=start, target=end)
        return path
    except nx.NetworkXNoPath:
        return None

def find_entities_connected_to(G, entity):
    """
    Find all entities directly connected to a given entity.
    """
    neighbors = list(G.neighbors(entity))
    return neighbors

def find_relationships_by_entity(G, entity):
    """
    Find all relationships involving a given entity.
    """
    relationships = [
        (u, v, data["relation"]) for u, v, data in G.edges(data=True) if u == entity or v == entity
    ]
    return relationships

# Main Execution
if __name__ == "__main__":
    # Build the knowledge graph
    G = build_knowledge_graph(triplets_with_themes)

    # Example Queries
    print("\n1. Entities related to 'Financial Performance':")
    print(find_entities_by_theme(G, "Financial Performance"))

    print("\n2. Shortest path between 'Visa' and 'cross-border payments':")
    print(find_shortest_path(G, "Visa", "cross-border payments"))

    print("\n3. Entities directly connected to 'Visa Direct':")
    print(find_entities_connected_to(G, "Visa Direct"))

    print("\n4. Relationships involving 'Visa':")
    print(find_relationships_by_entity(G, "Visa"))

    # Visualize the knowledge graph
    visualize_knowledge_graph(G)



# ----------------------------------------------------------------------------------------------------------------------------------------------

import openai

# GPT-4 API setup (replace with your key)
openai.api_key = "your-api-key"

def extract_triplets_from_text(document_text):
    """
    Extract triplets from text using GPT-4.

    Example prompt:
        You are an advanced language model trained for information extraction. 
        Your task is to extract relationships in the form of triplets from the given text. 
        Format each triplet as: (Entity1, Relationship, Entity2).
        
    Guidelines:
        1. "Entity1" and "Entity2" should be key nouns (e.g., people, organizations, concepts).
        2. The "Relationship" should be a verb or phrase describing how "Entity1" and "Entity2" are related.
        3. Avoid duplicating triplets.
        4. Focus on meaningful and concise relationships.

    Example:
        Input: 
            "Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year. 
            Visa Direct transactions grew by 41%. Visa partnered with ICICI Bank to enable cross-border payments in India. 
            The company launched AI-driven payment solutions."
        Output: 
            1. (Visa, achieved, $8.9 billion in revenue)
            2. (Visa, grew by, 10%)
            3. (Visa Direct, grew by, 41%)
            4. (Visa, partnered with, ICICI Bank)
            5. (Visa, launched, AI-driven payment solutions)
    
    Args:
        document_text (str): The input text to extract triplets from.

    Returns:
        list of tuples: A list of extracted triplets in the form (Entity1, Relationship, Entity2).
    """
    prompt = f"""
    You are an advanced language model trained for information extraction. 
    Your task is to extract relationships in the form of triplets from the given text. 
    Format each triplet as: (Entity1, Relationship, Entity2).

    ### Guidelines ###
    1. "Entity1" and "Entity2" should be key nouns (e.g., people, organizations, concepts).
    2. The "Relationship" should be a verb or phrase describing how "Entity1" and "Entity2" are related.
    3. Avoid duplicating triplets.
    4. Focus on meaningful and concise relationships.

    ### Example ###
    Input: Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year.
    Visa Direct transactions grew by 41%. Visa partnered with ICICI Bank to enable cross-border payments in India. 
    The company launched AI-driven payment solutions.

    ### Output ###
    1. (Visa, achieved, $8.9 billion in revenue)
    2. (Visa, grew by, 10%)
    3. (Visa Direct, grew by, 41%)
    4. (Visa, partnered with, ICICI Bank)
    5. (Visa, launched, AI-driven payment solutions)

    Input: {document_text}
    Output:
    """
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=300,
        temperature=0
    )
    
    # Parse GPT-4 response
    extracted_text = response.choices[0].text.strip()
    triplets = []
    for line in extracted_text.split("\n"):
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            triplets.append(eval(line))  # Convert string to tuple
    return triplets

# Example usage
if __name__ == "__main__":
    # Replace with the content of your document
    document_text = (
        "Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year. "
        "Visa Direct transactions grew by 41%. Visa partnered with ICICI Bank to enable cross-border payments in India. "
        "The company launched AI-driven payment solutions."
    )

    # Extract triplets
    triplets = extract_triplets_from_text(document_text)

    # Print triplets
    print("Extracted Triplets:")
    for triplet in triplets:
        print(triplet)


# ----------------------------------------------------------------------------------------------------------------------------------------------
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import networkx as nx
import matplotlib.pyplot as plt

# Step 1: Initialize OpenAI API
llm = OpenAI(temperature=0, openai_api_key="your-api-key", model="gpt-4")

# Step 2: Define the Prompt Template
triplet_extraction_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Extract entities and relationships from the following text in the form of triplets: (Entity1, Relationship, Entity2).
    
    ### Guidelines ###
    1. Entity1 and Entity2 should be meaningful nouns (e.g., people, organizations, concepts).
    2. The Relationship should be a verb or a short phrase connecting the two entities.
    3. Avoid duplications and ensure relationships are concise and actionable.
    
    ### Example ###
    Input: Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year. Visa Direct transactions grew by 41%.
    Output:
    1. (Visa, achieved, $8.9 billion in revenue)
    2. (Visa, grew by, 10%)
    3. (Visa Direct, grew by, 41%)

    Input: {text}
    Output:
    """
)

# Step 3: Define the Chain
triplet_extraction_chain = LLMChain(llm=llm, prompt=triplet_extraction_prompt)

# Step 4: Extract Triplets
def extract_triplets(text):
    """
    Call the chain to extract triplets from the text.
    """
    response = triplet_extraction_chain.run(text=text)
    # Parse triplets from response
    triplets = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("(") and line.endswith(")"):
            triplets.append(eval(line))  # Convert string to tuple
    return triplets

# Step 5: Build Knowledge Graph
def build_knowledge_graph(triplets):
    """
    Create a NetworkX knowledge graph from triplets.
    """
    G = nx.DiGraph()
    for entity1, relation, entity2 in triplets:
        G.add_node(entity1, type="Entity")
        G.add_node(entity2, type="Entity")
        G.add_edge(entity1, entity2, relation=relation)
    return G

# Step 6: Visualize Knowledge Graph
def visualize_knowledge_graph(G):
    """
    Visualize the knowledge graph using matplotlib.
    """
    pos = nx.spring_layout(G)  # Generate positions for nodes
    edge_labels = nx.get_edge_attributes(G, "relation")
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=10, node_size=3000, edge_color="gray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Knowledge Graph Visualization")
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Replace with your document text
    document_text = """
    Visa achieved $8.9 billion in revenue in Q3 2024, growing by 10% year-over-year. 
    Visa partnered with ICICI Bank to enable cross-border payments in India. 
    The company launched AI-driven payment solutions and expanded to Asia.
    """

    # Extract triplets
    triplets = extract_triplets(document_text)
    print("Extracted Triplets:")
    for triplet in triplets:
        print(triplet)

    # Build and visualize the knowledge graph
    G = build_knowledge_graph(triplets)
    visualize_knowledge_graph(G)


