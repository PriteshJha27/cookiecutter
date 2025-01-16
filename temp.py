### Template for RAG Code Modular Structure

# Directory Structure
# app/
#   - main.py
#   - router/
#       - router.py
#   - config/
#       - config.yml
#   - src/
#       - indexing/
#           - data_loader.py
#           - chunking.py
#           - preprocessing.py
#           - save_index_vectorDB.py
#       - query_serving/
#           - query_rewrite.py
#           - prompts.py
#           - llm_call.py
#           - validation.py
#   - utils/
#       - data_check.py
#       - embedding/
#           - document_embedding.py
#           - query_embedding.py
#           - llm_service.py
#   - exceptions/
#       - custom_exceptions.py
#   - logger/
#       - logger.py
#   - tests/
#   - requirements.txt

# app/main.py
from router.router import router

if __name__ == "__main__":
    router()

# app/router/router.py
def router():
    from src.indexing.data_loader import DataLoader
    from src.indexing.chunking import Chunker
    from src.indexing.preprocessing import Preprocessor
    from src.indexing.save_index_vectorDB import IndexSaver
    from src.query_serving.query_rewrite import QueryRewriter
    from src.query_serving.prompts import PromptGenerator
    from src.query_serving.llm_call import LLMCaller
    from src.query_serving.validation import Validator

    # Example flow
    print("Starting Indexing Pipeline...")
    loader = DataLoader()
    data = loader.load_documents()
    chunker = Chunker()
    chunks = chunker.chunk_documents(data)
    preprocessor = Preprocessor()
    processed_data = preprocessor.preprocess(chunks)
    saver = IndexSaver()
    saver.save(processed_data)

    print("Starting Query Serving Pipeline...")
    rewriter = QueryRewriter()
    query = rewriter.rewrite("user_query")
    llm_caller = LLMCaller()
    response = llm_caller.call_llm(query)
    validator = Validator()
    validation_result = validator.validate_response(response)
    print(validation_result)

# app/config/config.yml
paths:
  data: "path/to/data"
  vectorstore: "path/to/vectorstore"

# app/src/indexing/data_loader.py
class DataLoader:
    def load_documents(self):
        print("Loading documents...")
        return ["Document 1", "Document 2"]

# app/src/indexing/chunking.py
class Chunker:
    def chunk_documents(self, documents):
        print("Chunking documents...")
        return ["Chunk 1", "Chunk 2"]

# app/src/indexing/preprocessing.py
class Preprocessor:
    def preprocess(self, chunks):
        print("Preprocessing chunks...")
        return ["Processed Chunk 1", "Processed Chunk 2"]

# app/src/indexing/save_index_vectorDB.py
class IndexSaver:
    def save(self, processed_data):
        print("Saving index to vector database...")

# app/src/query_serving/query_rewrite.py
class QueryRewriter:
    def rewrite(self, query):
        print("Rewriting query...")
        return "Rewritten Query"

# app/src/query_serving/prompts.py
class PromptGenerator:
    def generate(self):
        print("Generating prompts...")

# app/src/query_serving/llm_call.py
class LLMCaller:
    def call_llm(self, query):
        print("Calling LLM with query...")
        return "LLM Response"

# app/src/query_serving/validation.py
class Validator:
    def validate_response(self, response):
        print("Validating response...")
        return "Validation Result"

# app/utils/data_check.py
class DataChecker:
    def check_data(self, data):
        print("Checking data integrity...")

# app/utils/embedding/document_embedding.py
class DocumentEmbedder:
    def embed(self, document):
        print("Embedding document...")
        return "Document Embedding"

# app/utils/embedding/query_embedding.py
class QueryEmbedder:
    def embed(self, query):
        print("Embedding query...")
        return "Query Embedding"

# app/utils/embedding/llm_service.py
class LLMService:
    def serve(self):
        print("Serving LLM...")

# app/exceptions/custom_exceptions.py
class CustomException(Exception):
    pass

# app/logger/logger.py
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def get_logger(name):
    return logging.getLogger(name)

# app/requirements.txt
# Include your dependencies here, e.g.,
# langchain
# llama-index
# FAISS

### Instructions to Fill Logic
1. Start by completing `config.yml` with your specific paths.
2. Fill the `DataLoader` class with logic to load your documents.
3. Implement chunking in `Chunker` based on your desired strategy.
4. Add preprocessing logic in `Preprocessor`.
5. Implement saving of processed data to your vector database in `IndexSaver`.
6. Move to query serving:
   - Implement query rewriting in `QueryRewriter`.
   - Complete `PromptGenerator` with prompt-generation logic.
   - Fill LLM call logic in `LLMCaller`.
   - Add response validation logic in `Validator`.
7. Utilize `utils` for any reusable functionalities, such as embedding methods or data checks.
8. Use `exceptions` for handling errors and `logger` for logging.
9. Write tests in the `tests` folder to ensure smooth execution.

Let me know if you need further assistance or additional customization!

