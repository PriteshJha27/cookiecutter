```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import pdfplumber
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Union
import re
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import gc
from tqdm import tqdm
import json
import pickle
from pathlib import Path

@dataclass
class ChunkMetadata:
    """Store chunk metadata"""
    text: str
    page: int
    chunk_id: int

class LocalEmbeddingModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def encode_batch(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            gc.collect()

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = (sum_embeddings / sum_mask).numpy()
                all_embeddings.append(embeddings)

            del encoded_input, model_output
            gc.collect()

        return np.vstack(all_embeddings)

class PDFRAG:
    def __init__(self, model_path: str):
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.chunks = []
        
    def save_index(self, save_dir: str):
        """Save FAISS index and metadata to disk"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        
        # Save chunks metadata
        chunks_data = [
            {
                'text': chunk['text'],
                'page': chunk['page']
            }
            for chunk in self.chunks
        ]
        
        with open(save_dir / "chunks.pkl", 'wb') as f:
            pickle.dump(chunks_data, f)
            
        print(f"Index and metadata saved to {save_dir}")
    
    def load_index(self, load_dir: str):
        """Load FAISS index and metadata from disk"""
        load_dir = Path(load_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_dir / "index.faiss"))
        
        # Load chunks metadata
        with open(load_dir / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
            
        print(f"Loaded {len(self.chunks)} chunks from {load_dir}")

    def process_pdf(self, pdf_path: str, chunk_size: int = 256, save_dir: str = None):
        """Process PDF and optionally save index"""
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    sentences = re.split(r'[.!?]+', text)
                    current_chunk = []
                    current_length = 0
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                            
                        if current_length + len(sentence) > chunk_size:
                            if current_chunk:
                                chunks.append({
                                    'text': ' '.join(current_chunk),
                                    'page': page_num
                                })
                            current_chunk = [sentence]
                            current_length = len(sentence)
                        else:
                            current_chunk.append(sentence)
                            current_length += len(sentence)
                    
                    if current_chunk:
                        chunks.append({
                            'text': ' '.join(current_chunk),
                            'page': page_num
                        })

        self.chunks = chunks
        chunk_texts = [chunk['text'] for chunk in chunks]

        print("Creating embeddings...")
        embeddings = []
        batch_size = 4

        for i in tqdm(range(0, len(chunk_texts), batch_size)):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode_batch(batch_texts, batch_size=batch_size)
            embeddings.append(batch_embeddings)
            gc.collect()

        all_embeddings = np.vstack(embeddings)
        dimension = all_embeddings.shape[1]
        
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(all_embeddings.astype('float32'))

        print(f"Processed {len(self.chunks)} chunks")
        
        # Save if directory specified
        if save_dir:
            self.save_index(save_dir)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.index:
            return []

        query_embedding = self.embedding_model.encode_batch([query], batch_size=1)[0]
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )

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

def format_results(results: List[Dict]) -> str:
    output = ""
    for r in results:
        output += f"Page {r['metadata']['page']}:\n{r['content']}\n---\n"
    return output

# Example usage:
"""
# First time processing:
model_path = "path/to/minilm"
rag = PDFRAG(model_path)
rag.process_pdf("document.pdf", save_dir="indexes/doc1")

# Later, to load existing index:
model_path = "path/to/minilm"
rag = PDFRAG(model_path)
rag.load_index("indexes/doc1")

# Search
results = rag.search("What was redemption in 2014?")
print(format_results(results))
"""

# Batch processing multiple PDFs:
def process_pdf_directory(pdf_dir: str, model_path: str, index_dir: str):
    """Process all PDFs in a directory and save their indexes"""
    pdf_dir = Path(pdf_dir)
    index_dir = Path(index_dir)
    
    rag = PDFRAG(model_path)
    
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        save_path = index_dir / pdf_file.stem
        rag.process_pdf(str(pdf_file), save_dir=save_path)

def load_specific_index(model_path: str, index_path: str) -> PDFRAG:
    """Load a specific index"""
    rag = PDFRAG(model_path)
    rag.load_index(index_path)
    return rag
```
