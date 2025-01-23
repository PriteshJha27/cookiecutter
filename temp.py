
from db_connector import VectorDBConnector
from typing import List, Dict

class RAGRetriever:
    def __init__(self):
        self.db = VectorDBConnector()
    
    def store_document(self, content: str, metadata: Dict) -> None:
        usecase_id = metadata.get('id', '1001001001')
        return self.db.create_index(content, metadata, usecase_id)
    
    def query(self, question: str, limit: int = 3) -> List[Dict]:
        # Get relevant documents
        docs = self.db.read_index(question, limit=limit)
        return [
            {
                'content': doc['bodyBlob'],
                'metadata': doc['metadata'],
                'score': doc['score']
            }
            for doc in docs
        ]

# Usage example
def main():
    rag = RAGRetriever()
    
    # Store a document
    doc = {
        'author': 'Tech COE',
        'usecase': 'TestUseCase1'
    }
    rag.store_document("Mastercard Q4 2023 Earnings Call Transcript", doc)
    
    # Query
    results = rag.query("What were Mastercard's earnings?")
    for result in results:
        print(f"Score: {result['score']}")
        print(f"Content: {result['content']}\n")

if __name__ == "__main__":
    main()
