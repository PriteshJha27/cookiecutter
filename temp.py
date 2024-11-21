```python
import pdfplumber
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import re
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import gc
from tqdm import tqdm

@dataclass
class PDFChunk:
    """Represents a chunk of content from PDF"""
    element_type: str
    content: str  # Always store as string for consistency
    metadata: Dict
    chunk_id: int

class LocalEmbeddingModel:
    def __init__(self, model_path: str):
        """Initialize local embedding model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return (sum_embeddings / sum_mask).cpu().numpy()

    def encode_batch(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """Encode texts to embeddings in batches"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.append(embeddings)

            # Clear some memory
            del encoded_input
            del model_output
            gc.collect()

        return np.vstack(all_embeddings)

class PDFRAG:
    def __init__(self, model_path: str):
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.chunks = []
        self.chunk_texts = []
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,;:?!()\'-]', '', text)
        return text

    def chunk_text(self, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into chunks"""
        text = self.preprocess_text(text)
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) > chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1  # +1 for space

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process_pdf(self, pdf_path: str, chunk_size: int = 512):
        """Process PDF and create searchable index"""
        try:
            all_text = []
            chunk_id = 0
            
            # Extract text from PDF
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        all_text.append({
                            'page': page_num,
                            'text': text
                        })

            # Create chunks
            self.chunks = []
            self.chunk_texts = []

            for text_obj in all_text:
                page_chunks = self.chunk_text(text_obj['text'], chunk_size)
                
                for chunk in page_chunks:
                    if len(chunk.strip()) > 50:  # Minimum chunk length
                        self.chunks.append(PDFChunk(
                            element_type='text',
                            content=chunk,
                            metadata={'page': text_obj['page']},
                            chunk_id=chunk_id
                        ))
                        self.chunk_texts.append(chunk)
                        chunk_id += 1

            # Create index in batches
            print("Creating embeddings and index...")
            embeddings = []
            batch_size = 8

            for i in tqdm(range(0, len(self.chunk_texts), batch_size)):
                batch_texts = self.chunk_texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode_batch(batch_texts, batch_size=batch_size)
                embeddings.append(batch_embeddings)

            all_embeddings = np.vstack(embeddings)

            # Create FAISS index
            dimension = all_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(all_embeddings.astype('float32'))

            # Clear memory
            del all_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Processed {len(self.chunks)} chunks")

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks with memory efficient approach"""
        try:
            if not self.index:
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode_batch([query])[0]

            # Search index
            distances, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                k
            )

            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'score': float(distance)
                    })

            # Clear some memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise

def format_results_for_llm(query: str, results: List[Dict]) -> str:
    """Format search results for LLM prompt"""
    context = f"Query: {query}\n\nRelevant content from PDF:\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"[{i}] Content from page {result['metadata']['page']}:\n"
        context += f"{result['content']}\n\n"
    
    prompt = f"""{context}
Based on the above content from the PDF, please answer the following question:
{query}

Answer:"""
    
    return prompt

# Example usage:
"""
# Initialize RAG
model_path = "path/to/local/minilm"
rag = PDFRAG(model_path)

# Process PDF
rag.process_pdf("document.pdf")

# Memory efficient search
query = "What was TCS's revenue in Q4 2023?"
try:
    results = rag.search(query)
    prompt = format_results_for_llm(query, results)
except Exception as e:
    print(f"Error: {str(e)}")
"""
```
