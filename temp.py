
```python
import pickle
import json
from pathlib import Path
import os
import faiss

class UnderwritingRAG:
    def __init__(self, model_path: str):
        self.embedding_model = LocalEmbeddingModel(model_path)
        self.index = None
        self.schema_store = {}
        self.dataframes = {}
        self._initialize_schemas()

    def save_vectorstore(self, directory: str):
        """
        Save FAISS index, schemas, and dataframes to disk
        Args:
            directory: Directory to save the vectorstore
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            faiss.write_index(self.index, str(directory / "index.faiss"))

        # Save schema store
        schema_data = {
            name: {
                "name": schema.name,
                "description": schema.description,
                "columns": schema.columns,
                "relationships": schema.relationships,
                "key_metrics": schema.key_metrics,
                "sample_queries": schema.sample_queries
            }
            for name, schema in self.schema_store.items()
        }
        with open(directory / "schema_store.json", "w") as f:
            json.dump(schema_data, f, indent=2)

        # Save dataframes
        with open(directory / "dataframes.pkl", "wb") as f:
            pickle.dump(self.dataframes, f)

        print(f"Vectorstore saved to {directory}")

    @classmethod
    def load_vectorstore(cls, model_path: str, directory: str):
        """
        Load FAISS index, schemas, and dataframes from disk
        Args:
            model_path: Path to the embedding model
            directory: Directory containing the vectorstore
        Returns:
            UnderwritingRAG instance
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory {directory} does not exist")

        # Create instance
        instance = cls(model_path)

        # Load FAISS index
        index_path = directory / "index.faiss"
        if index_path.exists():
            instance.index = faiss.read_index(str(index_path))

        # Load schema store
        schema_path = directory / "schema_store.json"
        if schema_path.exists():
            with open(schema_path) as f:
                schema_data = json.load(f)
            
            instance.schema_store = {
                name: TableSchema(
                    name=data["name"],
                    description=data["description"],
                    columns=data["columns"],
                    relationships=data["relationships"],
                    key_metrics=data["key_metrics"],
                    sample_queries=data["sample_queries"]
                )
                for name, data in schema_data.items()
            }

        # Load dataframes
        df_path = directory / "dataframes.pkl"
        if df_path.exists():
            with open(df_path, "rb") as f:
                instance.dataframes = pickle.load(f)

        print(f"Loaded vectorstore from {directory}")
        return instance

    def process_and_save(self, directory: str, risk_df, borrower_df, credit_df, ratios_df, statements_df):
        """
        Process all dataframes and save the vectorstore
        Args:
            directory: Directory to save the vectorstore
            risk_df: Risk assessment DataFrame
            borrower_df: Borrower profile DataFrame
            credit_df: Credit history DataFrame
            ratios_df: Financial ratios DataFrame
            statements_df: Financial statements DataFrame
        """
        # Add all dataframes
        self.add_dataframe(risk_df, "risk_assessment")
        self.add_dataframe(borrower_df, "borrower_profile")
        self.add_dataframe(credit_df, "credit_history")
        self.add_dataframe(ratios_df, "financial_ratios")
        self.add_dataframe(statements_df, "financial_statements")
        
        # Save vectorstore
        self.save_vectorstore(directory)

# Example usage:
"""
# First time processing and saving:
model_path = "path/to/minilm"
rag = UnderwritingRAG(model_path)

# Read CSV files
risk_df = pd.read_csv("Risk_Assessment.csv")
borrower_df = pd.read_csv("Borrower_Profile.csv")
credit_df = pd.read_csv("Credit_History.csv")
ratios_df = pd.read_csv("Financial_Ratios.csv")
statements_df = pd.read_csv("Financial_Statements.csv")

# Process and save
rag.process_and_save(
    "vectorstores/underwriting",
    risk_df,
    borrower_df,
    credit_df,
    ratios_df,
    statements_df
)

# Later, to load the saved vectorstore:
rag = UnderwritingRAG.load_vectorstore(
    model_path="path/to/minilm",
    directory="vectorstores/underwriting"
)

# Query as before
results = rag.query("Find high-risk borrowers with poor financial ratios")
context = format_query_results(results)
"""

# Additional helper functions

def verify_vectorstore(directory: str) -> bool:
    """
    Verify that a vectorstore directory contains all required files
    Args:
        directory: Directory to verify
    Returns:
        bool: True if valid, False otherwise
    """
    directory = Path(directory)
    required_files = ["index.faiss", "schema_store.json", "dataframes.pkl"]
    return all((directory / file).exists() for file in required_files)

def list_vectorstores(base_directory: str) -> List[str]:
    """
    List all available vectorstores in a directory
    Args:
        base_directory: Base directory containing vectorstores
    Returns:
        List of vectorstore names
    """
    base_directory = Path(base_directory)
    vectorstores = []
    
    for directory in base_directory.iterdir():
        if directory.is_dir() and verify_vectorstore(directory):
            vectorstores.append(directory.name)
    
    return vectorstores
```
