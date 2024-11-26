












import networkx as nx
from sentence_transformers import SentenceTransformer, util
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from itertools import combinations

# Step 1: Initialize Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chat_model = ChatOpenAI(model="gpt-4")  # Use an LLM like GPT-4

# Step 2: Build the Knowledge Graph
def build_knowledge_graph(triplets):
    """
    Create a directed knowledge graph from triplets.
    """
    graph = nx.DiGraph()
    for head, relation, tail in triplets:
        graph.add_edge(head, tail, relation=relation)
    return graph

triplets = [
    ("Python", "is_a", "Programming Language"),
    ("Python", "used_for", "Data Analysis"),
    ("Pandas", "is_a", "Library"),
    ("Pandas", "used_for", "Data Manipulation"),
    ("Data Analysis", "involves", "Insights Extraction"),
]

kg = build_knowledge_graph(triplets)

# Step 3: Query Abstraction
def decompose_query(query):
    """
    Break a query into logical sub-operations.
    """
    operations = []
    if "used_for" in query:
        operations.append(("Projection", query.split()[0], "used_for"))
    return operations

query = "What is Python used for?"
decomposed_query = decompose_query(query)

# Step 4: Neighborhood Retrieval
def retrieve_neighborhood(graph, start_node, k=2):
    """
    Perform k-hop neighborhood retrieval to extract a subgraph.
    """
    neighborhood = nx.ego_graph(graph, start_node, radius=k, undirected=False)
    return neighborhood

start_entity = "Python"
k = 2
neighborhood = retrieve_neighborhood(kg, start_entity, k)

# Step 5: Logical Chain Operations
def perform_logical_operations(graph, operations):
    """
    Execute logical operations (Projection, Intersection, etc.) on the subgraph.
    """
    results = []
    for operation in operations:
        op_type, entity, relation = operation
        if op_type == "Projection":
            # Projection: Traverse the specified relation
            neighbors = [nbr for nbr in graph.neighbors(entity) if graph[entity][nbr]["relation"] == relation]
            results.append((op_type, neighbors))
    return results

logical_results = perform_logical_operations(neighborhood, decomposed_query)

# Step 6: Format LLM Prompt
def format_prompt(entity, neighborhood, logical_results, query):
    """
    Convert the graph and logical results into a structured prompt for LLM.
    """
    context = "The following is a subgraph of related entities and relations:\n"
    for edge in neighborhood.edges(data=True):
        context += f"{edge[0]} -[{edge[2]['relation']}]-> {edge[1]}\n"

    reasoning = "The logical results are as follows:\n"
    for result in logical_results:
        reasoning += f"{result[0]}: {result[1]}\n"

    prompt = (
        f"{context}\n\n"
        f"{reasoning}\n\n"
        f"Using this information, answer the query: '{query}'."
    )
    return prompt

prompt = format_prompt(start_entity, neighborhood, logical_results, query)

# Step 7: LLM Reasoning
def perform_reasoning(prompt):
    """
    Use the LLM to process the query based on the graph and logical results.
    """
    response = chat_model([HumanMessage(content=prompt)])
    return response.content

response = perform_reasoning(prompt)

# Final Output
print("Query:", query)
print("Response:", response)
