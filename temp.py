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

class PathState(NamedTuple):
    node: str
    path: List[str]
    score: float

class CosineGraphSearch:
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        self.graph = nx.DiGraph()
        self._build_graph(triplets)
        self.node_vectors = self._create_node_vectors()

    def _build_graph(self, triplets: List[Tuple[str, str, str]]) -> None:
        """Build graph from triplets"""
        for subj, pred, obj in triplets:
            self.graph.add_edge(subj, obj, predicate=pred)

    def _create_node_vectors(self) -> dict:
        """
        Create vector representations for nodes based on their connections
        Using adjacency patterns as features
        """
        nodes = list(self.graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        vectors = defaultdict(lambda: np.zeros(n * 2))  # *2 for in and out connections

        # Create vectors based on connection patterns
        for node in nodes:
            idx = node_to_idx[node]
            # Incoming connections
            for pred in self.graph.predecessors(node):
                vectors[node][node_to_idx[pred]] = 1
            # Outgoing connections
            for succ in self.graph.successors(node):
                vectors[node][node_to_idx[succ] + n] = 1

        return vectors

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def dfs_with_cosine(self, 
                       start_node: str,
                       target_vector: np.ndarray,
                       similarity_threshold: float = 0.1,
                       max_depth: int = 5,
                       max_results: int = 10) -> List[SearchResult]:
        """
        Perform DFS using cosine similarity to guide the search
        """
        results = []
        visited_paths = set()
        paths_queue = []

        # Initialize with start node
        start_similarity = self.cosine_similarity(
            self.node_vectors[start_node],
            target_vector
        )
        
        heapq.heappush(paths_queue, PathState(
            node=start_node,
            path=[start_node],
            score=-start_similarity  # Negative for max heap
        ))

        while paths_queue and len(results) < max_results:
            current_state = heapq.heappop(paths_queue)
            current_node = current_state.node
            current_path = current_state.path

            path_key = tuple(current_path)
            if path_key in visited_paths or len(current_path) > max_depth:
                continue

            visited_paths.add(path_key)
            similarity = -current_state.score  # Convert back to positive

            if similarity >= similarity_threshold:
                results.append(SearchResult(
                    path=current_path,
                    similarity_score=similarity
                ))

            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in current_path:  # Avoid cycles
                    neighbor_vector = self.node_vectors[neighbor]
                    neighbor_similarity = self.cosine_similarity(
                        neighbor_vector,
                        target_vector
                    )
                    
                    if neighbor_similarity >= similarity_threshold:
                        heapq.heappush(paths_queue, PathState(
                            node=neighbor,
                            path=current_path + [neighbor],
                            score=-neighbor_similarity  # Negative for max heap
                        ))

        return sorted(results, key=lambda x: x.similarity_score, reverse=True)

    def find_similar_paths(self, 
                         query_node: str, 
                         similarity_threshold: float = 0.1,
                         max_depth: int = 5,
                         max_results: int = 10) -> List[SearchResult]:
        """
        Find paths similar to the query node's connection pattern
        """
        query_vector = self.node_vectors[query_node]
        
        all_results = []
        for start_node in self.graph.nodes():
            if start_node != query_node:
                results = self.dfs_with_cosine(
                    start_node=start_node,
                    target_vector=query_vector,
                    similarity_threshold=similarity_threshold,
                    max_depth=max_depth,
                    max_results=max_results
                )
                all_results.extend(results)

        # Sort and return top results
        return sorted(all_results, 
                     key=lambda x: x.similarity_score, 
                     reverse=True)[:max_results]

    def visualize_results(self, results: List[SearchResult]) -> None:
        """Visualize the search results"""
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(15, 10))
        
        # Draw all nodes and edges in light gray
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightgray', 
                             node_size=500, alpha=0.3)
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', alpha=0.3)
        nx.draw_networkx_labels(self.graph, pos, alpha=0.3)
        
        # Highlight paths from results
        colors = ['r', 'g', 'b', 'c', 'm']
        
        for idx, result in enumerate(results[:5]):
            color = colors[idx % len(colors)]
            path = result.path
            
            # Draw nodes in path
            nx.draw_networkx_nodes(
                self.graph, pos,
                nodelist=path,
                node_color=color,
                node_size=800,
                alpha=0.6
            )
            
            # Draw edges in path
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=path_edges,
                edge_color=color,
                width=2,
                alpha=0.6
            )
            
            # Add similarity score
            plt.annotate(
                f'Score: {result.similarity_score:.2f}',
                xy=pos[path[0]],
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(facecolor='white', edgecolor=color, alpha=0.6)
            )
        
        plt.title("Search Results")
        plt.axis('off')
        plt.show()

# Example usage
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
    
    # Find paths similar to John's connection pattern
    results = search.find_similar_paths(
        query_node="John",
        similarity_threshold=0.1,
        max_depth=4,
        max_results=5
    )
    
    # Print results
    print("\nSimilar paths to John's pattern:")
    for i, result in enumerate(results, 1):
        print(f"\nPath {i} (Similarity: {result.similarity_score:.3f}):")
        print(" -> ".join(result.path))
    
    # Visualize results
    search.visualize_results(results)
