import pandas as pd
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from transformers import AutoModel, AutoTokenizer
import torch

@dataclass
class TableSchema:
    name: str
    description: str
    columns: Dict[str, str]  # column_name: description
    relationships: Dict[str, str]  # column_name: related_table.column
    key_metrics: List[str]
    sample_queries: List[str]

class LocalEmbeddingModel:
    def __init__(self, model_path: str):
        """
        Initialize local embedding model
        Args:
            model_path: Path to directory containing model files
        """
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
        """
        Encode texts to embeddings
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
        Returns:
            Numpy array of embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize and move to device
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling
            embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        return np.vstack(all_embeddings)

class UnderwritingRAG:
    def __init__(self, model_path: str):
        """
        Initialize UnderwritingRAG with local model path
        Args:
            model_path: Path to directory containing model files
        """
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.schema_store = {}
        self.dataframes = {}
        
        # Initialize schema definitions
        self._initialize_schemas()

    def _initialize_schemas(self):
        """Initialize schema metadata for all tables"""
        # [Previous schema definitions remain the same]
        self.schema_store = {
            "risk_assessment": TableSchema(
                name="Risk Assessment",
                description="Contains risk assessment details and recommendations for borrowers",
                columns={
                    "assessment_id": "Unique identifier for risk assessment",
                    "borrower_id": "Foreign key to Borrower Profile",
                    "risk_score": "Overall risk score (0-100)",
                    "market_risk": "Market risk level (Low/Moderate/High)",
                    "credit_risk": "Credit risk level (Low/Moderate/High)",
                    "operational_risk": "Operational risk level (Low/Moderate/High)",
                    "regulatory_risk": "Regulatory risk level (Low/Moderate/High)",
                    "recommendation": "Final recommendation (Approve/Reject/Further Review)",
                    "assessment_date": "Date of risk assessment",
                    "reviewed_by": "Name of reviewing officer",
                    "next_review_date": "Next scheduled review date"
                },
                relationships={
                    "borrower_id": "borrower_profile.borrower_id"
                },
                key_metrics=[
                    "risk_score",
                    "recommendation",
                    "risk_levels"
                ],
                sample_queries=[
                    "What is the risk assessment for borrower TCS-1?",
                    "Show all high credit risk assessments",
                    "List rejected applications with risk scores above 50"
                ]
            ),
            # [Rest of the schema definitions remain the same...]
        }

    def create_schema_embedding(self, table_name: str) -> str:
        """Create rich schema description for embedding"""
        # [Previous implementation remains the same]
        schema = self.schema_store[table_name]
        
        schema_text = f"Table: {schema.name}\n"
        schema_text += f"Description: {schema.description}\n\n"
        
        # Add columns
        schema_text += "Columns:\n"
        for col, desc in schema.columns.items():
            schema_text += f"- {col}: {desc}\n"
        
        # Add relationships
        if schema.relationships:
            schema_text += "\nRelationships:\n"
            for col, rel in schema.relationships.items():
                schema_text += f"- {col} relates to {rel}\n"
        
        # Add key metrics
        schema_text += "\nKey Metrics:\n"
        for metric in schema.key_metrics:
            schema_text += f"- {metric}\n"
        
        # Add sample queries
        schema_text += "\nTypical Queries:\n"
        for query in schema.sample_queries:
            schema_text += f"- {query}\n"
            
        return schema_text

    def add_dataframe(self, df: pd.DataFrame, table_name: str) -> None:
        """Add a dataframe to the system"""
        self.dataframes[table_name] = df
        
        if self.index is None:
            # Create initial index with schema embedding
            schema_text = self.create_schema_embedding(table_name)
            embedding = self.embedding_model.encode([schema_text])[0]
            self.index = faiss.IndexFlatL2(embedding.shape[0])
            self.index.add(np.array([embedding]).astype('float32'))
        else:
            # Add schema embedding to existing index
            schema_text = self.create_schema_embedding(table_name)
            embedding = self.embedding_model.encode([schema_text])[0]
            self.index.add(np.array([embedding]).astype('float32'))

    def query(self, query: str, k: int = 2) -> List[Dict]:
        """Query the system and return relevant tables and data"""
        # Create query embedding
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        # Get relevant tables
        relevant_tables = []
        for idx, distance in enumerate(distances[0]):
            table_name = list(self.schema_store.keys())[indices[0][idx]]
            schema = self.schema_store[table_name]
            
            relevant_tables.append({
                'table_name': schema.name,
                'relevance_score': float(distance),
                'schema': schema,
                'sample_data': self.dataframes[table_name].head(3) if table_name in self.dataframes else None
            })
            
        return relevant_tables

def format_query_results(results: List[Dict]) -> str:
    """Format query results for LLM consumption"""
    context = "Relevant tables for your query:\n\n"
    
    for result in results:
        context += f"Table: {result['table_name']}\n"
        context += f"Relevance Score: {result['relevance_score']:.4f}\n"
        schema = result['schema']
        
        context += "Key Columns:\n"
        for col, desc in schema.columns.items():
            context += f"- {col}: {desc}\n"
        
        if result['sample_data'] is not None:
            context += "\nSample Data:\n"
            context += result['sample_data'].to_string()
        
        context += "\n\n"
    
    return context

# Example usage
def initialize_rag_system(model_path: str, risk_df, borrower_df, credit_df, ratios_df, statements_df):
    """
    Initialize RAG system with local model
    Args:
        model_path: Path to directory containing model files
        risk_df: Risk assessment DataFrame
        borrower_df: Borrower profile DataFrame
        credit_df: Credit history DataFrame
        ratios_df: Financial ratios DataFrame
        statements_df: Financial statements DataFrame
    Returns:
        Initialized UnderwritingRAG instance
    """
    rag = UnderwritingRAG(model_path)
    
    # Add all dataframes
    rag.add_dataframe(risk_df, "risk_assessment")
    rag.add_dataframe(borrower_df, "borrower_profile")
    rag.add_dataframe(credit_df, "credit_history")
    rag.add_dataframe(ratios_df, "financial_ratios")
    rag.add_dataframe(statements_df, "financial_statements")
    
    return rag

# Usage example:
"""
# Specify path to local model files
model_path = "path/to/local/minilm/model"

# Read CSV files
risk_df = pd.read_csv("Risk_Assessment.csv")
borrower_df = pd.read_csv("Borrower_Profile.csv")
credit_df = pd.read_csv("Credit_History.csv")
ratios_df = pd.read_csv("Financial_Ratios.csv")
statements_df = pd.read_csv("Financial_Statements.csv")

# Initialize RAG with local model
rag = initialize_rag_system(
    model_path,
    risk_df,
    borrower_df,
    credit_df,
    ratios_df,
    statements_df
)

# Query example
query = "Find high-risk borrowers with poor financial ratios"
results = rag.query(query)
context = format_query_results(results)
"""
