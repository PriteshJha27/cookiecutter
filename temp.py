
import networkx as nx
import fitz  # PyMuPDF for PDF text extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Extract Keywords
def extract_keywords(text, max_features=10):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()

# Step 3: Create Graph
def create_graph(keywords):
    graph = nx.Graph()
    graph.add_nodes_from(keywords)
    # Connect nodes based on similarity
    for node1, node2 in combinations(keywords, 2):
        # Example: Adding edges with a simple weight
        weight = cosine_similarity([[len(node1)]], [[len(node2)]])[0][0]
        graph.add_edge(node1, node2, weight=weight)
    return graph

# Step 4: Visualize the Graph
def visualize_graph(graph):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=2000, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()

# Main Execution
pdf_path = "example.pdf"  # Replace with your PDF file
text = extract_text_from_pdf(pdf_path)
keywords = extract_keywords(text)
graph = create_graph(keywords)

print("Nodes in Graph:", graph.nodes)
print("Edges in Graph:", graph.edges)
visualize_graph(graph)
