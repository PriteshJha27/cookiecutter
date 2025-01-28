def precompute_node_embeddings(G, embeddings_model):
    # Store embeddings in node attributes to avoid recomputing
    node_embeddings = {}
    for node in G.nodes():
        node_embeddings[node] = embeddings_model.embed_query(node)
    nx.set_node_attributes(G, node_embeddings, 'embedding')
    return G

def faster_semantic_search(G, query_triplets, top_k=5):
    # Use pre-computed embeddings
    query_embedding = embeddings_model.embed_query(query_triplets[0].subject)  # Or any other entity
    
    scores = []
    for node in G.nodes():
        node_embedding = G.nodes[node]['embedding']
        similarity = cosine_similarity(query_embedding, node_embedding)
        scores.append((node, similarity))
    
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]






def match_query_patterns(G, query_triplets):
    matches = []
    for qt in query_triplets:
        # Find nodes with similar subject/object patterns
        for node in G.nodes():
            if similar_pattern(qt, node, G):
                matches.append(node)
    return matches

def similar_pattern(query_triple, node, G):
    # Check structural similarity
    node_neighbors = set(G.neighbors(node))
    query_pattern = set(get_triple_pattern(query_triple))
    return len(node_neighbors.intersection(query_pattern)) > 0







def hybrid_search(G, query_triplets, embeddings_model):
    # First filter by structure
    structural_matches = match_query_patterns(G, query_triplets)
    
    if len(structural_matches) == 0:
        # Fallback to semantic search on subset
        return semantic_search_subset(G, query_triplets, subset_size=10)
    
    # Then rank by semantic similarity
    query_embedding = embeddings_model.embed_query(query_triplets[0].subject)
    ranked_matches = []
    
    for node in structural_matches:
        node_embedding = G.nodes[node]['embedding']
        similarity = cosine_similarity(query_embedding, node_embedding)
        ranked_matches.append((node, similarity))
    
    return sorted(ranked_matches, key=lambda x: x[1], reverse=True)




def heuristic_traversal(G, start_nodes, query_triplets, max_depth=3):
    visited = set()
    matches = []
    
    def dfs_with_heuristic(node, depth=0):
        if depth >= max_depth or node in visited:
            return
        
        visited.add(node)
        
        # Quick heuristic check (can be customized)
        if matches_heuristic(node, query_triplets, G):
            matches.append(node)
        
        # Prioritize promising neighbors
        neighbors = sorted(G.neighbors(node), 
                         key=lambda n: neighbor_score(n, query_triplets, G),
                         reverse=True)
        
        for neighbor in neighbors[:5]:  # Limit branching
            if neighbor not in visited:
                dfs_with_heuristic(neighbor, depth + 1)
    
    for start in start_nodes:
        dfs_with_heuristic(start)
    
    return matches

def matches_heuristic(node, query_triplets, G):
    # Quick pattern matching without embeddings
    node_relations = set(d['relation'] for _, _, d in G.edges(node, data=True))
    query_relations = set(t.predicate for t in query_triplets)
    return len(node_relations.intersection(query_relations)) > 0

def neighbor_score(neighbor, query_triplets, G):
    # Simple scoring based on relation overlap
    neighbor_relations = set(d['relation'] for _, _, d in G.edges(neighbor, data=True))
    query_relations = set(t.predicate for t in query_triplets)
    return len(neighbor_relations.intersection(query_relations))












#----------------------------------------------------------------------------------------

from typing import List, Set, Dict, Tuple, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import networkx as nx
from collections import defaultdict
import json
import matplotlib.pyplot as plt
from difflib import SequenceMatcher

class Triple(BaseModel):
    subject: str = Field(description="Subject entity of the triple")
    predicate: str = Field(description="Relationship between subject and object")
    object: str = Field(description="Object entity of the triple")

class TripleList(BaseModel):
    triples: List[Triple] = Field(description="List of extracted triples")

class KGTraversal:
    def __init__(self, max_depth: int = 3, similarity_threshold: float = 0.6):
        self.max_depth = max_depth
        self.similarity_threshold = similarity_threshold
        self.seen_patterns: Dict[str, Set[str]] = defaultdict(set)
        
    def extract_query_triplets(self, query: str) -> List[Triple]:
        """Extract triplets from query using LangChain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract subject-predicate-object triples from the query. Focus on key entities and relationships."),
            ("human", "{text}")
        ])
        
        model = ChatOpenAI(temperature=0)
        parser = JsonOutputParser(pydantic_object=TripleList)
        chain = prompt | model | parser
        
        result = chain.invoke({"text": query})
        return result.triples

    def string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using SequenceMatcher."""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def get_node_pattern(self, node: str, G: nx.DiGraph) -> Dict[str, Set[str]]:
        """Extract node's local pattern (incoming and outgoing edges with relations)."""
        if node in self.seen_patterns:
            return self.seen_patterns[node]
            
        pattern = {
            'out_relations': set(),
            'in_relations': set(),
            'neighbors': set()
        }
        
        # Outgoing edges
        for _, target, data in G.edges(node, data=True):
            pattern['out_relations'].add(data['relation'])
            pattern['neighbors'].add(target)
            
        # Incoming edges
        for source, _, data in G.in_edges(node, data=True):
            pattern['in_relations'].add(data['relation'])
            pattern['neighbors'].add(source)
            
        self.seen_patterns[node] = pattern
        return pattern

    def calculate_relation_similarity(self, query_relations: Set[str], node_relations: Set[str]) -> float:
        """Calculate similarity between relation sets."""
        if not query_relations or not node_relations:
            return 0.0
            
        max_similarities = []
        for qr in query_relations:
            similarities = [self.string_similarity(qr, nr) for nr in node_relations]
            max_similarities.append(max(similarities) if similarities else 0)
            
        return sum(max_similarities) / len(max_similarities)

    def matches_heuristic(self, node: str, query_triplets: List[Triple], G: nx.DiGraph) -> Tuple[bool, float]:
        """Check if node matches query pattern using heuristics."""
        node_pattern = self.get_node_pattern(node, G)
        query_relations = {t.predicate for t in query_triplets}
        
        # Calculate relation similarity
        out_sim = self.calculate_relation_similarity(query_relations, node_pattern['out_relations'])
        in_sim = self.calculate_relation_similarity(query_relations, node_pattern['in_relations'])
        
        # Entity similarity
        query_entities = {t.subject for t in query_triplets} | {t.object for t in query_triplets}
        entity_similarities = [max(self.string_similarity(node, qe) for qe in query_entities)]
        entity_sim = max(entity_similarities) if entity_similarities else 0
        
        # Combined score
        total_score = (out_sim + in_sim + entity_sim) / 3
        return total_score >= self.similarity_threshold, total_score

    def neighbor_score(self, neighbor: str, query_triplets: List[Triple], G: nx.DiGraph) -> float:
        """Score neighbor nodes for traversal prioritization."""
        neighbor_pattern = self.get_node_pattern(neighbor, G)
        query_relations = {t.predicate for t in query_triplets}
        
        # Relation similarity
        relation_sim = self.calculate_relation_similarity(
            query_relations,
            neighbor_pattern['out_relations'] | neighbor_pattern['in_relations']
        )
        
        # Connectivity score (normalize by max_depth)
        connectivity = len(neighbor_pattern['neighbors']) / (self.max_depth * 2)
        
        # Combined score with weights
        return (relation_sim * 0.7) + (connectivity * 0.3)

    def heuristic_traversal(self, G: nx.DiGraph, query: str, start_nodes: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Main traversal function using heuristics."""
        query_triplets = self.extract_query_triplets(query)
        visited = set()
        matches = []
        
        if not start_nodes:
            # If no start nodes provided, use nodes with highest initial scores
            start_nodes = sorted(
                [(node, self.neighbor_score(node, query_triplets, G)) for node in G.nodes()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            start_nodes = [node for node, _ in start_nodes]
        
        def dfs_with_heuristic(node: str, depth: int = 0) -> None:
            if depth >= self.max_depth or node in visited:
                return
                
            visited.add(node)
            
            # Check if node matches query pattern
            is_match, score = self.matches_heuristic(node, query_triplets, G)
            if is_match:
                matches.append((node, score))
            
            # Get and sort neighbors by score
            neighbors = [(n, self.neighbor_score(n, query_triplets, G)) 
                        for n in G.neighbors(node) if n not in visited]
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            # Traverse top-scoring neighbors
            for neighbor, _ in neighbors[:5]:  # Limit branching
                dfs_with_heuristic(neighbor, depth + 1)
        
        # Start traversal from each start node
        for start in start_nodes:
            dfs_with_heuristic(start)
        
        # Sort matches by score
        return sorted(matches, key=lambda x: x[1], reverse=True)

def visualize_traversal_results(G: nx.DiGraph, matches: List[Tuple[str, float]], query: str):
    """Visualize the traversal results in the knowledge graph."""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Draw all nodes and edges
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    
    # Highlight matched nodes
    matched_nodes = [node for node, _ in matches]
    nx.draw_networkx_nodes(G, pos, nodelist=matched_nodes, 
                          node_color='red', node_size=800)
    
    # Add labels
    labels = {node: f"{node}\n({score:.2f})" if node in matched_nodes 
             else node for node, score in matches}
    nx.draw_networkx_labels(G, pos, labels)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(f"Traversal Results for Query: {query}")
    plt.axis('off')
    plt.show()

# Example usage
def main():
    # Create sample knowledge graph
    G = nx.DiGraph()
    # Add your nodes and edges here
    
    # Initialize traversal
    traversal = KGTraversal(max_depth=3, similarity_threshold=0.6)
    
    # Example query
    query = "What are the relationships between technology companies?"
    
    # Perform traversal
    matches = traversal.heuristic_traversal(G, query)
    
    # Print results
    print("\nTraversal Results:")
    for node, score in matches:
        print(f"Node: {node}, Relevance Score: {score:.3f}")
    
    # Visualize results
    visualize_traversal_results(G, matches, query)

if __name__ == "__main__":
    main()

# Initialize
traversal = KGTraversal(max_depth=3, similarity_threshold=0.6)

# Run traversal
matches = traversal.heuristic_traversal(your_graph, "your query")

# Visualize results
visualize_traversal_results(your_graph, matches, "your query")
