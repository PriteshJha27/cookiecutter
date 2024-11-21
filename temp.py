
import pdfplumber
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import re
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
from collections import defaultdict

@dataclass
class PDFElement:
    """Base class for PDF elements"""
    page_num: int
    content_type: str  # 'text', 'table', 'heading'
    content: Union[str, pd.DataFrame]
    bounding_box: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict = None

@dataclass
class PDFChunk:
    """Represents a chunk of content from PDF"""
    element_type: str
    content: Union[str, pd.DataFrame]
    metadata: Dict
    embedding: Optional[np.ndarray] = None

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
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
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
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)

class PDFProcessor:
    def __init__(self):
        self.heading_patterns = [
            r'^[A-Z][^.!?]*$',  # Uppercase starting lines
            r'^[0-9]+\.\s+[A-Z].*$',  # Numbered sections
        ]
    
    def extract_elements(self, pdf_path: str) -> List[PDFElement]:
        """Extract text and table elements from PDF"""
        elements = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table and any(any(cell for cell in row) for row in table):
                        df = pd.DataFrame(table[1:], columns=table[0])
                        elements.append(PDFElement(
                            page_num=page_num,
                            content_type='table',
                            content=df,
                            metadata={
                                'table_rows': len(df),
                                'table_cols': len(df.columns)
                            }
                        ))
                
                # Extract text
                text_content = page.extract_text()
                if text_content:
                    # Split into paragraphs
                    paragraphs = text_content.split('\n\n')
                    for para in paragraphs:
                        if para.strip():
                            element_type = 'heading' if any(re.match(pattern, para.strip()) for pattern in self.heading_patterns) else 'text'
                            elements.append(PDFElement(
                                page_num=page_num,
                                content_type=element_type,
                                content=para.strip(),
                                metadata={
                                    'length': len(para),
                                    'is_heading': element_type == 'heading'
                                }
                            ))
        
        return elements

class PDFChunker:
    def __init__(self, max_chunk_size: int = 512):
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text content"""
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if current_length + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence)
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def chunk_table(self, df: pd.DataFrame) -> List[str]:
        """Convert table chunks to textual representation"""
        chunks = []
        
        # Table summary
        summary = f"Table with {len(df)} rows and columns: {', '.join(df.columns)}"
        chunks.append(summary)
        
        # Chunk rows
        current_chunk = []
        current_length = 0
        
        for _, row in df.iterrows():
            row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
            if current_length + len(row_text) > self.max_chunk_size:
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                current_chunk = [row_text]
                current_length = len(row_text)
            else:
                current_chunk.append(row_text)
                current_length += len(row_text)
        
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks

class PDFRAG:
    def __init__(self, model_path: str):
        self.processor = PDFProcessor()
        self.chunker = PDFChunker()
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.chunks = []
        
    def process_pdf(self, pdf_path: str):
        """Process PDF and create searchable index"""
        # Extract elements
        elements = self.processor.extract_elements(pdf_path)
        
        # Create chunks
        chunks = []
        for element in elements:
            if element.content_type in ['text', 'heading']:
                text_chunks = self.chunker.chunk_text(element.content)
                for chunk in text_chunks:
                    chunks.append(PDFChunk(
                        element_type='text',
                        content=chunk,
                        metadata={
                            'page': element.page_num,
                            'type': element.content_type,
                            **element.metadata
                        }
                    ))
            elif element.content_type == 'table':
                table_chunks = self.chunker.chunk_table(element.content)
                for chunk in table_chunks:
                    chunks.append(PDFChunk(
                        element_type='table',
                        content=chunk,
                        metadata={
                            'page': element.page_num,
                            'type': 'table',
                            **element.metadata
                        }
                    ))
        
        # Create embeddings
        texts = [str(chunk.content) for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        # Store embeddings in chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        # Create FAISS index
        if len(chunks) > 0:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            self.chunks = chunks
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        if not self.index:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        # Format results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            results.append({
                'content': chunk.content,
                'metadata': chunk.metadata,
                'score': float(distance)
            })
        
        return results

def format_results_for_llm(query: str, results: List[Dict]) -> str:
    """Format search results for LLM prompt"""
    context = f"Query: {query}\n\nRelevant content from PDF:\n\n"
    
    for i, result in enumerate(results, 1):
        context += f"[{i}] "
        if result['metadata']['type'] == 'table':
            context += f"Table content (page {result['metadata']['page']}):\n"
        else:
            context += f"Text content (page {result['metadata']['page']}):\n"
        context += f"{result['content']}\n\n"
    
    prompt = f"""{context}
Based on the above content from the PDF, please answer the following question:
{query}

Answer:"""
    
    return prompt

# Example usage:
"""
# Initialize RAG system with local model path
model_path = "path/to/local/minilm/model"
rag = PDFRAG(model_path)

# Process PDF
rag.process_pdf("path/to/pdf")

# Query the system
query = "What are TCS's revenue figures for 2023?"
results = rag.search(query)

# Format for LLM
prompt = format_results_for_llm(query, results)

# Use prompt with your preferred LLM
# llm_response = call_llm(prompt)
"""
