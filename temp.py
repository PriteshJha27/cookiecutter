from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
from minilm import embeddings  # Assuming this is your embedding model
import logging

@dataclass
class VectorDBConnector:
    user_name: str = os.getenv("AIDA_API_USER")
    api_url: str = os.getenv("AIDA_API_URL")
    token: Optional[str] = None

    def __post_init__(self):
        if not self.token:
            self.token = self.get_okta_token()

    def get_okta_token(self) -> str:
        data = f'grant_type=password&username={self.user_name}&password={os.getenv("AIDA_API_SECRET")}&scope=openid'
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic <your_auth_token>"
        }
        response = requests.post(
            f"{os.getenv('OKTA_URL')}/oauth2/aus1u191g9Pek5suX0H7/v1/token",
            data=data,
            headers=headers,
            verify=False
        )
        if response.status_code == 200:
            return response.json()['access_token']
        raise Exception(f"Failed to get token: {response.status_code}")

    def create_index(self, content: str, metadata: Dict[str, Any], usecase_id: str) -> Dict:
        vector = embeddings.encode_batch([content], batch_size=1)[0].tolist()
        
        payload = {
            "bodyBlob": content,
            "metadata": metadata,
            "summaryBlob": f"Summary of {metadata.get('usecase', 'content')}",
            "useCaseId": usecase_id,
            "useCaseName": metadata.get("usecase", "default"),
            "vector": vector
        }
        
        return self._make_api_call(
            f"{self.api_url}/createAiDaFeatureOpsVector.v1",
            payload
        )

    def read_index(self, query: str, limit: int = 3, threshold: float = 0.72) -> List[Dict]:
        query_vector = embeddings.encode_batch([query], batch_size=1)[0].tolist()
        
        payload = {
            "useCaseId": "default",
            "limit": limit,
            "threshold": threshold,
            "vector": query_vector
        }
        
        response = self._make_api_call(
            f"{self.api_url}/readAiDaFeatureOpsVector.v1",
            payload
        )
        return response.get('items', [])

    def _make_api_call(self, url: str, data: Dict) -> Dict:
        headers = {
            "Content-Type": "application/json",
            "one-data-correlation-id": str(uuid.uuid4()),
            "Authorization": f"Claims impersonation_id={self.user_name};app_authorization={self.token}"
        }
        
        response = requests.post(url, json=data, headers=headers, verify=False)
        if response.status_code == 200:
            return response.json()
        raise Exception(f"API call failed: {response.status_code}")




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
