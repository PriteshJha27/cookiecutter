import os
from typing import List, Dict
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import faiss
import numpy as np
import pickle
import torch

def read_pdf(file_path: str) -> str:
    """
    Read a PDF file and extract its text content.
    
    Args:
        file_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

def create_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks using LangChain's text splitter.
    
    Args:
        text (str): Input text to be split
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_embeddings(chunks: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """
    Create embeddings for text chunks using MiniLM model.
    
    Args:
        chunks (List[str]): List of text chunks
        model_name (str): Name of the HuggingFace model to use
        
    Returns:
        np.ndarray: Matrix of embeddings
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    chunk_embeddings = embeddings.embed_documents(chunks)
    return np.array(chunk_embeddings)

def create_and_save_index(embeddings: np.ndarray, chunks: List[str], index_dir: str = "index") -> None:
    """
    Create and save FAISS index along with the original chunks.
    
    Args:
        embeddings (np.ndarray): Matrix of embeddings
        chunks (List[str]): List of original text chunks
        index_dir (str): Directory to save the index and chunks
    """
    # Create directory if it doesn't exist
    os.makedirs(index_dir, exist_ok=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and chunks
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

def load_index(index_dir: str = "index") -> tuple:
    """
    Load the FAISS index and chunks from disk.
    
    Args:
        index_dir (str): Directory containing the index and chunks
        
    Returns:
        tuple: (FAISS index, list of chunks)
    """
    index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "chunks.pkl"), "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def retrieve_similar_chunks(query: str, index, chunks: List[str], 
                          model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                          k: int = 3) -> List[Dict[str, any]]:
    """
    Retrieve similar chunks for a given query.
    
    Args:
        query (str): Search query
        index: FAISS index
        chunks (List[str]): List of original text chunks
        model_name (str): Name of the HuggingFace model to use
        k (int): Number of results to retrieve
        
    Returns:
        List[Dict]: List of dictionaries containing chunks and their distances
    """
    # Create embedding for the query
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    query_embedding = embeddings.embed_query(query)
    
    # Search the index
    distances, indices = index.search(np.array([query_embedding]), k)
    
    # Format results
    results = []
    for i in range(len(indices[0])):
        results.append({
            'chunk': chunks[indices[0][i]],
            'distance': distances[0][i]
        })
    
    return results

def llm_process_results(query: str, results: List[Dict], api_key: str) -> str:
    """
    Process retrieval results using an LLM.
    
    Args:
        query (str): Original query
        results (List[Dict]): Retrieved chunks and their distances
        api_key (str): API key for the LLM service
        
    Returns:
        str: LLM-generated response
    """
    # Combine chunks into context
    context = "\n\n".join([result['chunk'] for result in results])
    
    # Create prompt
    prompt = f"""Based on the following context, please answer the question.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    # Here you can integrate with your preferred LLM API
    # This is a placeholder - replace with actual LLM call
    response = "Placeholder for LLM response"
    
    return response

# Example usage
if __name__ == "__main__":
    # Read PDF
    pdf_text = read_pdf("sample.pdf")
    
    # Create chunks
    chunks = create_chunks(pdf_text)
    
    # Create embeddings
    embeddings_matrix = create_embeddings(chunks)
    
    # Create and save index
    create_and_save_index(embeddings_matrix, chunks)
    
    # Later, when you want to query:
    index, stored_chunks = load_index()
    
    # Retrieve similar chunks for a query
    query = "What is the main topic?"
    results = retrieve_similar_chunks(query, index, stored_chunks)
    
    # Process results with LLM
    api_key = "your-api-key"
    final_answer = llm_process_results(query, results, api_key)
    print(final_answer)
