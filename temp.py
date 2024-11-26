
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from itertools import combinations

# Initialize MiniLM model and ChatOpenAI
model = SentenceTransformer("all-MiniLM-L6-v2")  # Embedding model
chat_model = ChatOpenAI(model="gpt-4")  # Replace with your OpenAI model

# Step 1: Sample Entities and Relationships
entities = [
    "Python is a programming language.",
    "Python is used for data analysis.",
    "Pandas is a library in Python.",
    "Pandas is used for data manipulation.",
    "Data analysis involves extracting insights from data.",
]

# Step 2: Generate Entity Embeddings
embeddings = model.encode(entities, convert_to_tensor=True)

# Step 3: Build the Knowledge Graph
def create_knowledge_graph(entities, embeddings, threshold=0.5):
    graph = nx.Graph()
    graph.add_nodes_from(entities)
    
    # Compute similarity between entity pairs
    for i, j in combinations(range(len(entities)), 2):
        similarity = util.pytorch_cos_sim(embeddings[i], embeddings[j]).item()
        if similarity > threshold:
            graph.add_edge(entities[i], entities[j], weight=similarity)
    return graph

graph = create_knowledge_graph(entities, embeddings)

# Step 4: Query the Knowledge Graph
def query_graph(graph, query):
    """
    Find nodes and edges relevant to the query.
    """
    query_embedding = model.encode(query, convert_to_tensor=True)
    nodes = list(graph.nodes)
    node_embeddings = model.encode(nodes, convert_to_tensor=True)
    
    # Find the most similar node(s) to the query
    similarities = util.pytorch_cos_sim(query_embedding, node_embeddings)
    most_similar_idx = similarities.argmax().item()
    relevant_node = nodes[most_similar_idx]
    
    # Fetch neighbors and their relationships
    neighbors = list(graph.neighbors(relevant_node))
    relationships = [(relevant_node, neighbor, graph[relevant_node][neighbor]["weight"]) for neighbor in neighbors]
    
    return relevant_node, relationships

query = "What is Python used for?"
main_node, relations = query_graph(graph, query)

# Step 5: Use ChatOpenAI for Reasoning
def generate_response(main_node, relations):
    """
    Use ChatOpenAI to explain relationships.
    """
    context = f"The main entity is '{main_node}'. The related entities and their relationships are as follows:\n"
    for relation in relations:
        context += f"- '{relation[1]}' (similarity: {relation[2]:.2f})\n"
    
    prompt = (
        f"{context}\n\nBased on this knowledge graph, answer the query: "
        f"'{query}'. Provide a detailed and accurate response."
    )
    
    response = chat_model([HumanMessage(content=prompt)])
    return response.content

response = generate_response(main_node, relations)
print("LARK Response:\n", response)
