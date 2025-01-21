import networkx as nx
import numpy as np
from typing import List, Tuple, Set, NamedTuple
from dataclasses import dataclass
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt

@dataclass
class SearchResult:
    path: List[str]
    similarity_score: float
    connection_details: dict  # Added to store connection info

class PathState(NamedTuple):
    node: str
    path: List[str]
    score: float

class CosineGraphSearch:
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        self.graph = nx.DiGraph()
        self.triplets = triplets
        self._build_graph(triplets)
        self.node_vectors, self.feature_names = self._create_node_vectors()

    def _build_graph(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Build graph from triplets and store predicates"""
        for subj, pred, obj in triplets:
            self.graph.add_edge(subj, obj, predicate=pred)

    def _create_node_vectors(self) -> tuple:
        """
        Create enhanced vector representations for nodes based on their connections
        and predicates
        """
        # Collect all unique nodes and predicates
        nodes = list(self.graph.nodes())
        predicates = list(set(data['predicate'] for _, _, data in self.graph.edges(data=True)))
        
        # Create feature names for better interpretation
        feature_names = []
        for pred in predicates:
            feature_names.extend([f"out_{pred}", f"in_{pred}"])
        
        n_features = len(predicates) * 2
        vectors = defaultdict(lambda: np.zeros(n_features))

        # Create vectors based on both connections and predicates
        for node in nodes:
            # Outgoing edges
            for _, neighbor, data in self.graph.out_edges(node, data=True):
                pred_idx = predicates.index(data['predicate'])
                vectors[node][pred_idx * 2] += 1

            # Incoming edges
            for neighbor, _, data in self.graph.in_edges(node, data=True):
                pred_idx = predicates.index(data['predicate'])
                vectors[node][pred_idx * 2 + 1] += 1

        return vectors, feature_names

    def get_node_connections(self, node: str) -> dict:
        """Get detailed connection information for a node"""
        connections = {
            'outgoing': [],
            'incoming': []
        }
        
        # Outgoing connections
        for _, neighbor, data in self.graph.out_edges(node, data=True):
            connections['outgoing'].append({
                'node': neighbor,
                'predicate': data['predicate']
            })
        
        # Incoming connections
        for neighbor, _, data in self.graph.in_edges(node, data=True):
            connections['incoming'].append({
                'node': neighbor,
                'predicate': data['predicate']
            })
        
        return connections

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def find_similar_paths(self, 
                         query_node: str, 
                         similarity_threshold: float = 0.01,  # Lowered threshold
                         max_depth: int = 3,
                         max_results: int = 10) -> List[SearchResult]:
        """Find paths similar to the query node's connection pattern"""
        query_vector = self.node_vectors[query_node]
        query_connections = self.get_node_connections(query_node)
        
        all_results = []
        for start_node in self.graph.nodes():
            if start_node != query_node:
                similarity = self.cosine_similarity(
                    self.node_vectors[start_node],
                    query_vector
                )
                
                if similarity >= similarity_threshold:
                    connections = self.get_node_connections(start_node)
                    all_results.append(SearchResult(
                        path=[start_node],
                        similarity_score=similarity,
                        connection_details=connections
                    ))

        # Sort results by similarity score
        return sorted(all_results, key=lambda x: x.similarity_score, reverse=True)[:max_results]

    def print_vector_details(self, node: str):
        """Print detailed vector representation for a node"""
        vector = self.node_vectors[node]
        print(f"\nVector details for node '{node}':")
        for i, value in enumerate(vector):
            if value > 0:
                print(f"{self.feature_names[i]}: {value}")

    def print_comparison(self, node1: str, node2: str):
        """Print detailed comparison between two nodes"""
        vec1 = self.node_vectors[node1]
        vec2 = self.node_vectors[node2]
        similarity = self.cosine_similarity(vec1, vec2)
        
        print(f"\nComparison between '{node1}' and '{node2}':")
        print(f"Similarity score: {similarity:.3f}")
        
        print(f"\n{node1} connections:")
        connections1 = self.get_node_connections(node1)
        for direction in ['outgoing', 'incoming']:
            print(f"  {direction.capitalize()}:")
            for conn in connections1[direction]:
                print(f"    -> {conn['predicate']} -> {conn['node']}")
        
        print(f"\n{node2} connections:")
        connections2 = self.get_node_connections(node2)
        for direction in ['outgoing', 'incoming']:
            print(f"  {direction.capitalize()}:")
            for conn in connections2[direction]:
                print(f"    -> {conn['predicate']} -> {conn['node']}")

# Example usage with debugging information
if __name__ == "__main__":
    # Example triplets
    triplets = [
        ("John", "owns", "cat"),
        ("cat", "has_color", "brown"),
        ("cat", "sits_on", "mat"),
        ("mat", "located_in", "kitchen"),
        ("John", "lives_in", "house"),
        ("house", "has", "kitchen"),
        ("kitchen", "contains", "mat"),
        ("cat", "eats", "food"),
        ("food", "stored_in", "kitchen"),
        ("Mary", "owns", "dog"),
        ("dog", "has_color", "black"),
        ("dog", "plays_in", "garden"),
        ("Mary", "lives_in", "apartment")
    ]
    
    # Initialize search
    search = CosineGraphSearch(triplets)
    
    # Print all nodes and their vector representations
    print("Node vector details:")
    for node in search.graph.nodes():
        search.print_vector_details(node)
    
    # Find similar patterns to John's connections
    print("\nFinding patterns similar to John's connections...")
    results = search.find_similar_paths(
        query_node="John",
        similarity_threshold=0.01,  # Lowered threshold for testing
        max_depth=3,
        max_results=5
    )
    
    # Print detailed results
    print("\nSearch Results:")
    if not results:
        print("No results found. Try adjusting the similarity threshold.")
    else:
        for i, result in enumerate(results, 1):
            print(f"\nResult {i} (Similarity: {result.similarity_score:.3f}):")
            print("Node:", result.path[0])
            print("Connection details:")
            for direction in ['outgoing', 'incoming']:
                print(f"  {direction.capitalize()}:")
                for conn in result.connection_details[direction]:
                    print(f"    -> {conn['predicate']} -> {conn['node']}")
    
    # Print detailed comparison between John and Mary (who should have similar patterns)
    search.print_comparison("John", "Mary")
