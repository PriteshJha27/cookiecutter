import networkx as nx
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Set, Dict, NamedTuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import heapq

@dataclass
class SearchResult:
    path: List[str]
    path_score: float
    node_scores: List[float]
    predicates: List[str]

class PathState(NamedTuple):
    node: str
    path: List[str]
    predicates: List[str]
    score: float
    node_scores: List[float]

class SemanticGraphSearch:
    def __init__(self, triplets: List[Tuple[str, str, str]], model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
        self.predicate_embeddings = {}
        self._build_graph(triplets)

    def _build_graph(self, triplets: List[Tuple[str, str, str]]) -> None:
        """
        Build graph and compute embeddings for nodes and predicates
        """
        # Add edges and collect unique predicates
        unique_predicates = set()
        for subj, pred, obj in triplets:
            self.graph.add_edge(subj, obj, predicate=pred)
            unique_predicates.add(pred)

        # Compute embeddings for nodes
        all_nodes = list(self.graph.nodes())
        if all_nodes:
            node_embeddings = self.model.encode(all_nodes)
            self.node_embeddings = dict(zip(all_nodes, node_embeddings))

        # Compute embeddings for predicates
        pred_list = list(unique_predicates)
        if pred_list:
            pred_embeddings = self.model.encode(pred_list)
            self.predicate_embeddings = dict(zip(pred_list, pred_embeddings))

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def semantic_similarity_score(self, query: str, node: str, predicate: str = None) -> float:
        """
        Calculate combined semantic similarity score for a node and optional predicate
        """
        query_embedding = self.model.encode(query)
        node_similarity = self.cosine_similarity(query_embedding, self.node_embeddings[node])
        
        if predicate:
            pred_similarity = self.cosine_similarity(
                query_embedding, 
                self.predicate_embeddings[predicate]
            )
            return (node_similarity + pred_similarity) / 2
        
        return node_similarity

    def semantic_dfs_with_cosine(self, 
                                query: str,
                                start_node: str = None,
                                similarity_threshold: float = 0.3,
                                max_depth: int = 5,
                                max_results: int = 10) -> List[SearchResult]:
        """
        Perform semantic DFS using cosine similarity for both nodes and path traversal
        """
        query_embedding = self.model.encode(query)
        results = []
        visited_paths = set()

        # Priority queue to store paths based on their scores
        # Using negative score for max heap behavior
        paths_queue = []

        # Start nodes selection
        start_nodes = [start_node] if start_node else list(self.graph.nodes())
        
        # Initialize paths from start nodes
        for node in start_nodes:
            score = self.semantic_similarity_score(query, node)
            if score >= similarity_threshold:
                heapq.heappush(paths_queue, PathState(
                    node=node,
                    path=[node],
                    predicates=[],
                    score=-score,  # Negative for max heap
                    node_scores=[score]
                ))

        while paths_queue and len(results) < max_results:
            current_state = heapq.heappop(paths_queue)
            current_node = current_state.node
            current_path = current_state.path
            current_predicates = current_state.predicates
            current_scores = current_state.node_scores

            path_key = tuple(current_path)
            if path_key in visited_paths or len(current_path) > max_depth:
                continue

            visited_paths.add(path_key)

            # Add current path to results
            if len(current_path) > 1:
                avg_score = -current_state.score  # Convert back to positive
                results.append(SearchResult(
                    path=current_path,
                    path_score=avg_score,
                    node_scores=current_scores,
                    predicates=current_predicates
                ))

            # Explore neighbors
            for neighbor in self.graph.neighbors(current_node):
                if neighbor not in current_path:  # Avoid cycles
                    predicate = self.graph[current_node][neighbor]['predicate']
                    
                    # Calculate semantic similarity for the neighbor and predicate
                    neighbor_score = self.semantic_similarity_score(query, neighbor, predicate)
                    
                    if neighbor_score >= similarity_threshold:
                        # Calculate new path score as average of all node scores
                        new_scores = current_scores + [neighbor_score]
                        path_score = sum(new_scores) / len(new_scores)
                        
                        heapq.heappush(paths_queue, PathState(
                            node=neighbor,
                            path=current_path + [neighbor],
                            predicates=current_predicates + [predicate],
                            score=-path_score,  # Negative for max heap
                            node_scores=new_scores
                        ))

        return sorted(results, key=lambda x: x.path_score, reverse=True)

    def visualize_semantic_paths(self, results: List[SearchResult]) -> None:
        """
        Visualize the semantic search results in the graph
        """
        import matplotlib.pyplot as plt
        
        pos = nx.spring_layout(self.graph)
        plt.figure(figsize=(15, 10))
        
        # Draw all nodes and edges in light gray
        nx.draw_networkx_nodes(self.graph, pos, node_color='lightgray', 
                             node_size=500, alpha=0.3)
        nx.draw_networkx_edges(self.graph, pos, edge_color='lightgray', alpha=0.3)
        nx.draw_networkx_labels(self.graph, pos, alpha=0.3)
        
        # Highlight paths from results
        colors = ['r', 'g', 'b', 'c', 'm']  # Different colors for different paths
        
        for idx, result in enumerate(results[:5]):  # Show top 5 paths
            color = colors[idx % len(colors)]
            path = result.path
            
            # Draw nodes in path
            path_nodes = nx.draw_networkx_nodes(
                self.graph, pos, 
                nodelist=path,
                node_color=color,
                node_size=800,
                alpha=0.6
            )
            
            # Draw edges in path
            path_edges = []
            for i in range(len(path)-1):
                path_edges.append((path[i], path[i+1]))
            
            nx.draw_networkx_edges(
                self.graph, pos,
                edgelist=path_edges,
                edge_color=color,
                width=2,
                alpha=0.6
            )
            
            # Add score labels
            for node, score in zip(path, result.node_scores):
                plt.annotate(
                    f'{score:.2f}',
                    xy=pos[node],
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(facecolor='white', edgecolor=color, alpha=0.6)
                )
        
        plt.title("Semantic Search Results")
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
        ("food", "stored_in", "kitchen")
    ]
    
    # Initialize semantic search
    semantic_search = SemanticGraphSearch(triplets)
    
    # Example query
    query = "Where is the cat's food?"
    
    # Perform semantic se
