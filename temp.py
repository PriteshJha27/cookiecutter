import pandas as pd


class SchemaLinker:
    def __init__(self):
        pass

    def create_schema_documents(self, df):
        """
        Creates schema metadata documents from a given DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing schema metadata.
        
        Returns:
            List[dict]: List of schema documents.
        """
        print("Creating schema documents...")
        documents = []
        for _, row in df.iterrows():
            document = {
                "metadata": {
                    "dataset": row["dataset"],
                    "table_name": row["table_name"],
                    "table_description": row["table_description"],
                    "business_name": row["business_name"],
                    "column_name": row["column_name"],
                    "column_description": row["column_description"],
                    "key_cols": row["key_cols"],
                    "partition_cols": row["partition_cols"],
                }
            }
            documents.append(document)
        return documents

    def split_large_documents(self, documents, max_length=1100):
        """
        Splits documents that exceed a certain length into smaller chunks.
        
        Args:
            documents (List[dict]): List of schema documents.
            max_length (int): Maximum length of each document.
        
        Returns:
            List[dict]: List of smaller schema documents.
        """
        print("Splitting large schema documents...")
        short_docs = []
        for doc in documents:
            if len(doc) > max_length:
                # Truncate to max_length if needed (example logic for simplicity)
                short_docs.append(doc[:max_length])
            else:
                short_docs.append(doc)
        return short_docs
