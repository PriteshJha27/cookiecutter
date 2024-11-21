```python
import os
# Set OpenMP environment variables before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# Optionally limit OpenMP threads if needed
os.environ['OMP_NUM_THREADS'] = '1'

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

class LocalEmbeddingModel:
    def __init__(self, model_path: str):
        """Initialize local embedding model with memory optimization"""
        # Set low memory mode for transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # Use float32 instead of float64
            low_cpu_mem_usage=True
        )
        
        # Move to CPU explicitly to avoid CUDA memory issues
        self.device = torch.device('cpu')  # Force CPU usage
        self.model.to(self.device)
        
        # Clear CUDA cache if it was used previously
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """Encode texts to embeddings with smaller batches"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Clear memory
            gc.collect()

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,  # Reduced from 512
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                attention_mask = encoded_input['attention_mask']
                
                # Mean pooling
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).numpy()
                
                all_embeddings.append(embeddings)

            # Clear memory
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

    def process_pdf(self, pdf_path: str, chunk_size: int = 256):  # Reduced chunk size
        """Process PDF with memory optimization"""
        try:
            chunks = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Split text into smaller chunks
                        sentences = re.split(r'[.!?]+', text)
                        current_chunk = []
                        current_length = 0
                        
                        for sentence in sentences:
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                                
                            if current_length + len(sentence) > chunk_size:
                                if current_chunk:
                                    chunk_text = ' '.join(current_chunk)
                                    chunks.append({
                                        'text': chunk_text,
                                        'page': page_num
                                    })
                                current_chunk = [sentence]
                                current_length = len(sentence)
                            else:
                                current_chunk.append(sentence)
                                current_length += len(sentence)
                        
                        if current_chunk:
                            chunk_text = ' '.join(current_chunk)
                            chunks.append({
                                'text': chunk_text,
                                'page': page_num
                            })

            # Store chunks
            self.chunks = chunks
            self.chunk_texts = [chunk['text'] for chunk in chunks]

            # Create embeddings in small batches
            print("Creating embeddings...")
            embeddings = []
            batch_size = 4  # Smaller batch size

            for i in tqdm(range(0, len(self.chunk_texts), batch_size)):
                batch_texts = self.chunk_texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode_batch(batch_texts, batch_size=batch_size)
                embeddings.append(batch_embeddings)
                gc.collect()  # Clear memory after each batch

            # Create FAISS index
            all_embeddings = np.vstack(embeddings)
            dimension = all_embeddings.shape[1]
            
            # Use CPU index
            self.index = faiss.IndexFlatL2(dimension)
            self.index = faiss.index_cpu_to_all_gpus(self.index)  # Use GPU if available
            self.index.add(all_embeddings.astype('float32'))

            print(f"Processed {len(self.chunks)} chunks")

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search with memory optimization"""
        try:
            if not self.index:
                return []

            # Generate query embedding
            query_embedding = self.embedding_model.encode_batch([query], batch_size=1)[0]
            query_embedding = query_embedding.reshape(1, -1).astype('float32')

            # Search
            distances, indices = self.index.search(query_embedding, k)

            # Format results
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunks):
                    chunk = self.chunks[idx]
                    results.append({
                        'content': chunk['text'],
                        'metadata': {'page': chunk['page']},
                        'score': float(distance)
                    })

            return results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise

# Example usage
"""
# Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Initialize RAG
model_path = "path/to/local/minilm"
rag = PDFRAG(model_path)

# Process PDF with smaller chunks
rag.process_pdf("document.pdf")

# Search
try:
    results = rag.search("What was redemption in 2014?", k=3)
    for result in results:
        print(f"Page {result['metadata']['page']}:")
        print(result['content'])
        print("---")
except Exception as e:
    print(f"Search failed: {str(e)}")
"""
```
