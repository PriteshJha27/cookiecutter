
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Dict, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import json
from numpy import dot
from numpy.linalg import norm

class Triple(BaseModel):
    subject: str = Field(description="Subject entity of the triple")
    predicate: str = Field(description="Relationship between subject and object")
    object: str = Field(description="Object entity of the triple")

class TripleList(BaseModel):
    triples: List[Triple] = Field(description="List of extracted triples")

def create_kg_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract subject-predicate-object triples from the text. Focus on key entities and their relationships."),
        ("human", "{text}")
    ])
    
    model = ChatOpenAI(temperature=0)
    parser = JsonOutputParser(pydantic_object=TripleList)
    
    return prompt | model | parser

def recursive_chunk_split(text: str, chunk_size: int = 1000) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    mid = text.rfind(". ", 0, chunk_size) + 1
    if mid == 0:
        mid = chunk_size
    
    first_half = text[:mid]
    second_half = text[mid:]
    
    return recursive_chunk_split(first_half, chunk_size) + recursive_chunk_split(second_half, chunk_size)

def extract_triples_from_chunks(chunks: List[str], chain) -> List[Triple]:
    all_triples = []
    for chunk in chunks:
        try:
            result = chain.invoke({"text": chunk})
            all_triples.extend(result.triples)
        except Exception as e:
            print(f"Error processing chunk: {e}")
    return all_triples

def create_knowledge_graph(triples: List[Triple]) -> nx.DiGraph:
    G = nx.DiGraph()
    
    # Add nodes and edges
    for triple in triples:
        G.add_node(triple.subject)
        G.add_node(triple.object)
        G.add_edge(triple.subject, triple.object, relation=triple.predicate)
    
    return G

def save_kg_to_json(G: nx.DiGraph, filename: str = "knowledge_graph.json"):
    kg_dict = {
        "nodes": [{"id": node, "label": node} for node in G.nodes()],
        "edges": [{"source": u, "target": v, "relation": d["relation"]} 
                 for u, v, d in G.edges(data=True)]
    }
    
    with open(filename, 'w') as f:
        json.dump(kg_dict, f, indent=2)
    
    return kg_dict

def visualize_kg(G: nx.DiGraph):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=1500, font_size=8, arrows=True)
    
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    plt.title("Knowledge Graph Visualization")
    plt.show()

def semantic_relevance_search(G: nx.DiGraph, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    
    node_scores = []
    for node in G.nodes():
        node_embedding = embeddings.embed_query(node)
        similarity = dot(query_embedding, node_embedding)/(norm(query_embedding)*norm(node_embedding))
        node_scores.append((node, similarity))
    
    return sorted(node_scores, key=lambda x: x[1], reverse=True)[:top_k]

def traverse_with_relevance(G: nx.DiGraph, start_node: str, query: str, 
                          threshold: float = 0.5, visited=None) -> List[Tuple[str, float]]:
    if visited is None:
        visited = set()
    
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    relevant_nodes = []
    
    def dfs(node, depth=0):
        if depth > 3 or node in visited:  # Limit depth to control traversal
            return
        
        visited.add(node)
        node_embedding = embeddings.embed_query(node)
        similarity = dot(query_embedding, node_embedding)/(norm(query_embedding)*norm(node_embedding))
        
        if similarity >= threshold:
            relevant_nodes.append((node, similarity))
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor, depth + 1)
    
    dfs(start_node)
    return sorted(relevant_nodes, key=lambda x: x[1], reverse=True)

# Example usage
def main():
    # Initialize the chain
    chain = create_kg_chain()
    
    # Sample text
    text = "Your input text here"
    
    # Process text
    chunks = recursive_chunk_split(text)
    triples = extract_triples_from_chunks(chunks, chain)
    
    # Create and save KG
    G = create_knowledge_graph(triples)
    kg_dict = save_kg_to_json(G)
    
    # Visualize
    visualize_kg(G)
    
    # Example semantic search
    query = "Your search query"
    relevant_nodes = semantic_relevance_search(G, query)
    print("Most relevant nodes:", relevant_nodes)
    
    # Example traversal from most relevant node
    if relevant_nodes:
        start_node = relevant_nodes[0][0]
        traversal_results = traverse_with_relevance(G, start_node, query)
        print("Traversal results:", traversal_results)

if __name__ == "__main__":
    main()
