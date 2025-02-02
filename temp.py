# llm/base.py
import logging
from typing import Dict, Any
from safechain.lcel import model
from utils.config_loader import load_config
from safechain.utils import get_token_from_env

logger = logging.getLogger(__name__)

class LLMService:
    """Base class for LLM interactions."""
    
    def __init__(self):
        self.config = load_config()
        self.llm_dict = {"gpt": "3", "llama": "1"}
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM model based on configuration."""
        try:
            model_id = self.llm_dict[self.config['llm']['model']]
            auth_token = get_token_from_env("1")
            return model(model_id)
        except KeyError as e:
            logger.error(f"Invalid model configuration: {e}")
            raise ValueError(f"Unsupported model type: {self.config['llm']['model']}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise

    def invoke_chain(self, messages: list) -> Dict[str, Any]:
        """Execute the LLM chain with given messages."""
        try:
            chain = messages | self.llm
            result = chain.invoke()
            return result
        except Exception as e:
            logger.error(f"Error invoking LLM chain: {e}")
            raise



########################################################

# llm/entity_extraction.py
import logging
from typing import Dict, Any
from safechain.prompts import ChatPromptTemplate
from llm.base import LLMService

logger = logging.getLogger(__name__)

class EntityExtractor(LLMService):
    """Handles entity extraction from text."""
    
    def extract_ner_entities(self, input_text: str) -> Dict[str, Any]:
        """Extract named entities from input text."""
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an advanced language model designed to extract structured information from text.
                Below is a transcript of an earnings call, with multiple speakers discussing various topics.
                Each speaker's statements are labeled with their name and role.
                Extract Named Entities such as Organizations, People, Financial Metrics, Dates, Products/Services, Locations, and other entities.
                Organize your output in a structured format.
                If no relevant information or speaker information is there in input text dont add just return NA."""),
                ("human", f"text: {input_text}"),
            ])
            
            response = self.invoke_chain(prompt_template)
            return response.content
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            raise

    def extract_metadata_entities(self, query: str) -> tuple:
        """Extract metadata entities from query text."""
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You need to find entities from input ie which is company_name, date,
                and period in key values pairs and return in json format. ***Return json only nothing else required***"""),
                ("human", f"{query}")
            ])
            
            result = self.invoke_chain(prompt_template)
            res = result.content
            
            # Process JSON response
            json_str = self._extract_json_content(res)
            data = json.loads(json_str)
            
            company_name = data.get('company_name', "")
            date = data.get('date', "")
            period = data.get('period', "")
            
            return period, date, company_name
        except Exception as e:
            logger.error(f"Error in metadata extraction: {e}")
            raise

    def _extract_json_content(self, content: str) -> str:
        """Helper method to extract JSON content from response."""
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            return content[json_start:json_end]
        except Exception as e:
            logger.error(f"Error extracting JSON content: {e}")
            raise ValueError("Invalid JSON content in response")



########################################################################################################################


# llm/summarization.py
import logging
from typing import Dict, Any
from safechain.prompts import ChatPromptTemplate
from llm.base import LLMService

logger = logging.getLogger(__name__)

class TextSummarizer(LLMService):
    """Handles text summarization tasks."""
    
    def summarize_earnings_call(self, input_text: str) -> str:
        """Summarize earnings call text."""
        try:
            earnings_call_prompts = """ "Key themes and topics": ( "Identify the main themes and topics discussed, including financial performance, operational updates, and future outlook" )"""
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Write a concise summary in points of the following text delimited"""),
                ("human", f"text: {input_text}")
            ])
            
            summary_chain = prompt_template | self.llm
            result = summary_chain.invoke({"input": input_text})
            return result.content
        except Exception as e:
            logger.error(f"Error in earnings call summarization: {e}")
            return ""

    def summarize_chunks(self, input_text: str, query: str) -> str:
        """Summarize document chunks with respect to a query."""
        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """Write a concise summary for dummy documents in points of the following"""),
                ("user", f"input: {input_text}\n query:{query}")
            ])
            
            chain = prompt_template | self.llm
            result = chain.invoke({"input": input_text, "query": query})
            return result.content
        except Exception as e:
            logger.error(f"Error in chunk summarization: {e}")
            return ""



################################################################################################################################

# llm/hyde.py
import logging
from typing import List, Dict, Any
from safechain.prompts import ChatPromptTemplate
from llm.base import LLMService

logger = logging.getLogger(__name__)

class HyDEGenerator(LLMService):
    """Handles Hypothetical Document Embeddings (HyDE) generation."""
    
    def generate_hyde_questions(self, content: str) -> List[Dict[str, str]]:
        """Generate hypothetical questions and answers for content."""
        if not content.strip():
            return []

        try:
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an AI assistant that generates hypothetical questions.
                - Create a maximum of 5-6 questions if the content is important; otherwise, return empty.
                - Questions should be related to the content and specific to the details provided.
                - Provide answers based only on the content.
                Return your output as a list of dict having question ans answer:"""),
                ("human", f"content: {content}")
            ])
            
            chain = prompt_template | self.llm
            result = chain.invoke({"content": content})
            response = result.content
            
            if not response:
                logger.warning("No questions generated for the content")
                return []
                
            return response
        except Exception as e:
            logger.error(f"Error generating HyDE questions: {e}")
            print(f"Error generating questions: {e}")
            return []




############################################################################################################

# utils/config_management.py
import os
import yaml
import logging
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validates configuration settings."""
    
    @staticmethod
    def validate_paths(config: Dict[str, Any]) -> None:
        """Validate all paths in configuration exist."""
        required_paths = [
            config['vectorstore']['vectorstore_path'],
            config['vectorstore']['index_save_path'],
            config['data']['data_folder']
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                raise ValueError(f"Required path does not exist: {path}")

    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        valid_models = ['gpt', 'llama']
        if config['llm']['model'] not in valid_models:
            raise ValueError(f"Invalid model type. Must be one of {valid_models}")

    @staticmethod
    def validate_vectorstore_config(config: Dict[str, Any]) -> None:
        """Validate vector store configuration."""
        valid_stores = ['faiss', 'datastax']
        if not all(store in valid_stores for store in config['vectorstore']['storename']):
            raise ValueError(f"Invalid vector store. Must be one of {valid_stores}")

class ConfigLoader:
    """Handles configuration loading and environment setup."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.getenv('CONFIG_PATH', 'config/config.yaml')
        self.validator = ConfigValidator()
        self._load_environment()

    def _load_environment(self) -> None:
        """Load environment variables."""
        env_file = os.getenv('ENV_FILE', '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")

    def _read_yaml(self) -> Dict[str, Any]:
        """Read and parse YAML configuration file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Successfully loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise

    def load_and_validate(self) -> Dict[str, Any]:
        """Load and validate configuration."""
        config = self._read_yaml()
        
        try:
            self.validator.validate_paths(config)
            self.validator.validate_model_config(config)
            self.validator.validate_vectorstore_config(config)
            return config
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

def load_config() -> Dict[str, Any]:
    """Utility function to load configuration."""
    loader = ConfigLoader()
    return loader.load_and_validate()

############################################################################################################


# orchestrators/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class BaseOrchestrator(ABC):
    """Base class for all orchestrators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate orchestrator-specific configuration."""
        pass

    @abstractmethod
    def execute(self) -> Any:
        """Execute the orchestrator's main function."""
        pass

# orchestrators/data_loader.py
from typing import Dict, Any
from src.indexing.data_loader import DataLoader
import logging

logger = logging.getLogger(__name__)

class DataLoadOrchestrator(BaseOrchestrator):
    """Orchestrates data loading operations."""
    
    def _validate_config(self) -> None:
        if not self.config.get('data', {}).get('data_folder'):
            raise ValueError("Data folder not configured")

    def execute(self) -> Dict[str, Any]:
        """Load documents from configured source."""
        try:
            loader = DataLoader()
            documents = loader.load_document()
            logger.info(f"Successfully loaded documents from {self.config['data']['data_folder']}")
            return documents
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

# orchestrators/chunking.py
from typing import Dict, Any, Tuple, List
from src.indexing.lumos_chunking import Chunking
import logging

logger = logging.getLogger(__name__)

class ChunkingOrchestrator(BaseOrchestrator):
    """Orchestrates document chunking operations."""
    
    def _validate_config(self) -> None:
        preprocessing = self.config.get('preprocessing', {})
        if not isinstance(preprocessing.get('simple_ratio_threshold'), (int, float)):
            raise ValueError("Invalid chunking threshold configuration")

    def execute(self, documents: Dict[str, Any]) -> Tuple[List[Any], List[Any]]:
        """Process documents into chunks."""
        try:
            chunking = Chunking()
            datastax_chunks, faiss_chunks = chunking.create_chunks(data=documents)
            logger.info(f"Created {len(faiss_chunks)} chunks for FAISS and {len(datastax_chunks)} chunks for Datastax")
            return datastax_chunks, faiss_chunks
        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            raise

# orchestrators/embedding.py
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class EmbeddingOrchestrator(BaseOrchestrator):
    """Orchestrates embedding generation."""
    
    def _validate_config(self) -> None:
        if not self.config.get('embedding_model', {}).get('model_name'):
            raise ValueError("Embedding model not configured")

    def execute(self, chunks: List[Any]) -> List[Any]:
        """Generate embeddings for chunks."""
        try:
            model = SentenceTransformer(self.config['embedding_model']['embed_model'])
            embeddings = model.encode(chunks)
            logger.info(f"Generated embeddings for {len(chunks)} chunks")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

# orchestrators/indexing.py
from typing import Dict, Any, List
from src.indexing.document import DocumentManager
import logging

logger = logging.getLogger(__name__)

class IndexingOrchestrator(BaseOrchestrator):
    """Orchestrates vector store indexing operations."""
    
    def _validate_config(self) -> None:
        if not self.config.get('vectorstore', {}).get('storename'):
            raise ValueError("Vector store not configured")

    def execute(self, chunks: List[Any], embeddings: List[Any]) -> None:
        """Index chunks and embeddings in vector store."""
        try:
            doc_manager = DocumentManager(self.config)
            doc_manager.add_docs(chunks, embeddings)
            
            if self.config['vectorstore']['storename'] == 'faiss':
                doc_manager.save_to_file(self.config['vectorstore']['index_save_path'])
                
            logger.info("Successfully indexed documents")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            raise

############################################################################################################


# router/indexing_router.py
from typing import Dict, Any
import logging
from orchestrators.data_loader import DataLoadOrchestrator
from orchestrators.chunking import ChunkingOrchestrator
from orchestrators.embedding import EmbeddingOrchestrator
from orchestrators.indexing import IndexingOrchestrator

logger = logging.getLogger(__name__)

class IndexingRouter:
    """Routes and coordinates the indexing pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_orchestrator = DataLoadOrchestrator(config)
        self.chunking_orchestrator = ChunkingOrchestrator(config)
        self.embedding_orchestrator = EmbeddingOrchestrator(config)
        self.indexing_orchestrator = IndexingOrchestrator(config)

    def execute_pipeline(self) -> None:
        """Execute the complete indexing pipeline."""
        try:
            # Load documents
            logger.info("Starting document loading")
            documents = self.data_orchestrator.execute()

            # Create chunks
            logger.info("Starting document chunking")
            datastax_chunks, faiss_chunks = self.chunking_orchestrator.execute(documents)

            # Generate embeddings
            logger.info("Generating embeddings")
            embeddings = self.embedding_orchestrator.execute(
                faiss_chunks if self.config['vectorstore']['storename'] == 'faiss' else datastax_chunks
            )

            # Index documents
            logger.info("Indexing documents")
            self.indexing_orchestrator.execute(
                faiss_chunks if self.config['vectorstore']['storename'] == 'faiss' else datastax_chunks,
                embeddings
            )

            logger.info("Indexing pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in indexing pipeline: {e}")
            raise

# main.py
import logging
import sys
from utils.config_management import load_config
from router.indexing_router import IndexingRouter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('indexing.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Load and validate configuration
        config = load_config()
        
        # Initialize and execute indexing pipeline
        indexing_router = IndexingRouter(config)
        indexing_router.execute_pipeline()
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


############################################################################################################


# src/indexing/document.py
import logging
import faiss
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from src.indexing.datastax import DatastaxConnector
from src.exceptions.index_exceptions import IndexCreationError, IndexLoadError

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document vectorization and storage."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embed_model = config['embedding_model']['embed_model']
        self.embeddings = SentenceTransformer(self.embed_model)
        self.file_path = config['data']['data_folder']
        self.vector_db = None
        self.docstore = None
        self.RETRIEVER_TYPE = ['Vanilla', 'Auto_merging'][1]
        self._initialize_vector_store()

    def _initialize_vector_store(self) -> None:
        """Initialize the vector store based on configuration."""
        try:
            # Initialize for FAISS
            embed_test = self.embeddings.encode("hello world")
            self.index = faiss.IndexFlatL2(len(embed_test))
            self.vector_db = FAISS(
                embedding_function=self.embeddings.encode,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={}
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise IndexCreationError("Failed to initialize vector store")

    def add_docs(self, docs: List[Dict[str, Any]], embeddings: Optional[List[Any]] = None) -> None:
        """Add documents to the vector store."""
        try:
            store_type = self.config['vectorstore']['storename']
            if store_type == 'faiss':
                self._add_to_faiss(docs, embeddings)
            elif store_type == 'datastax':
                self._add_to_datastax(docs)
            logger.info(f"Successfully added {len(docs)} documents to {store_type}")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise

    def _add_to_faiss(self, docs: List[Dict[str, Any]], embeddings: List[Any]) -> None:
        """Add documents to FAISS vector store."""
        try:
            self.vector_db.add_texts(docs, embeddings=embeddings)
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise

    def _add_to_datastax(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents to Datastax vector store."""
        try:
            datastax = DatastaxConnector()
            create_payload_data = datastax.build_data([doc['content'] for doc in docs],
                                                    [doc['metadata'] for doc in docs])
            datastax.create_vector(data=create_payload_data)
        except Exception as e:
            logger.error(f"Error adding documents to Datastax: {e}")
            raise

    def retrieve(self, query: str, filter: Dict[str, str]) -> List[Tuple[Any, float]]:
        """Retrieve relevant documents for a query."""
        try:
            if self.RETRIEVER_TYPE == 'Vanilla':
                return self.vector_db.similarity_search_with_score(query, k=self.config['vectorstore']['k'])
            else:
                from src.indexing.retriever.automerger import AutoMergingRetriever
                retriever = AutoMergingRetriever(
                    vector_db=self.vector_db,
                    docstore=self.docstore,
                    simple_ratio_threshold=self.config['preprocessing']['simple_ratio_threshold']
                )
                return retriever.get_relevant_documents(query, filter)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def save_to_file(self, path: str) -> None:
        """Save the vector store to disk."""
        try:
            os.makedirs(path, exist_ok=True)
            self.vector_db.save_local(os.path.join(path, 'faiss.db'))
            
            metadata = {
                'file_path': self.file_path,
                'RETRIEVER_TYPE': self.RETRIEVER_TYPE
            }
            
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4)
                
            if self.docstore:
                with open(os.path.join(path, 'docstore.pkl'), 'wb') as f:
                    pickle.dump(self.docstore, f)
                    
            logger.info(f"Vector store saved successfully to {path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
            raise

    @classmethod
    def from_file(cls, path: str, config: Dict[str, Any]):
        """Load a vector store from disk."""
        try:
            instance = cls(config)
            
            # Load metadata
            with open(os.path.join(path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
                
            instance.file_path = metadata['file_path']
            instance.RETRIEVER_TYPE = metadata['RETRIEVER_TYPE']
            
            # Load vector store
            instance.vector_db = FAISS.load_local(
                os.path.join(path, 'faiss.db'),
                embeddings=instance.embeddings
            )
            
            # Load docstore if exists
            docstore_path = os.path.join(path, 'docstore.pkl')
            if os.path.exists(docstore_path):
                with open(docstore_path, 'rb') as f:
                    instance.docstore = pickle.load(f)
                    
            logger.info(f"Vector store loaded successfully from {path}")
            return instance
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise IndexLoadError(f"Failed to load vector store from {path}")


############################################################################################################

# src/indexing/chunking/base_chunker.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class BaseChunker(ABC):
    """Base class for document chunking strategies."""
    
    @abstractmethod
    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from the input text."""
        pass

# src/indexing/chunking/recursive_chunker.py
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.indexing.chunking.base_chunker import BaseChunker

class RecursiveChunker(BaseChunker):
    """Implements recursive text splitting strategy."""
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into chunks using recursive character splitting."""
        try:
            chunks = self.splitter.split_text(text)
            return [
                {
                    'content': chunk,
                    'metadata': {**metadata, 'chunk_size': self.chunk_size}
                }
                for chunk in chunks
            ]
        except Exception as e:
            logger.error(f"Error in recursive chunking: {e}")
            raise

# src/indexing/chunking/auto_merger.py
from typing import List, Dict, Any
from src.indexing.chunking.base_chunker import BaseChunker

class AutoMergeChunker(BaseChunker):
    """Implements auto-merging chunking strategy."""
    
    def __init__(self, chunk_sizes: List[int] = [4096, 1024, 256]):
        self.chunk_sizes = chunk_sizes

    def create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create hierarchical chunks with different sizes."""
        try:
            chunks = []
            for size in self.chunk_sizes:
                size_chunks = self._create_size_chunks(text, size)
                chunks.extend([
                    {
                        'content': chunk,
                        'metadata': {
                            **metadata,
                            'chunk_size': size,
                            'level': self.chunk_sizes.index(size)
                        }
                    }
                    for chunk in size_chunks
                ])
            return chunks
        except Exception as e:
            logger.error(f"Error in auto-merge chunking: {e}")
            raise

    def _create_size_chunks(self, text: str, size: int) -> List[str]:
        """Create chunks of specified size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

# src/indexing/lumos_chunking.py
from typing import List, Dict, Any, Tuple
import logging
from src.indexing.chunking.recursive_chunker import RecursiveChunker
from src.indexing.chunking.auto_merger import AutoMergeChunker
from utils.config_loader import load_config
from llm.hyde import HyDEGenerator

logger = logging.getLogger(__name__)

class Chunking:
    """Main chunking coordinator."""
    
    def __init__(self):
        self.config = load_config()
        self.recursive_chunker = RecursiveChunker()
        self.auto_merger = AutoMergeChunker()
        self.all_doc_chunks_datastax = []
        self.all_doc_chunks_faiss = []

    def create_chunks(self, data: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Process documents and create chunks."""
        try:
            for filename, pages in data.items():
                chunks = self._process_document(pages, filename)
                
                if self.config['preprocessing']['hyde_chunks']:
                    hyde_chunks = self._generate_hyde_chunks(chunks)
                    chunks.extend(hyde_chunks)
                    
                self.all_doc_chunks_faiss.extend(chunks)
                self.all_doc_chunks_datastax.extend(self._prepare_datastax_chunks(chunks))
                
            logger.info(f"Created {len(self.all_doc_chunks_faiss)} chunks for FAISS and {len(self.all_doc_chunks_datastax)} for Datastax")
            return self.all_doc_chunks_datastax, self.all_doc_chunks_faiss
        except Exception as e:
            logger.error(f"Error in chunk creation: {e}")
            raise

    def _process_document(self, pages: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        """Process a single document."""
        chunks = []
        for page in pages:
            if self.config['preprocessing']['auto_merging']:
                page_chunks = self.auto_merger.create_chunks(page.page_content, page.metadata)
            else:
                page_chunks = self.recursive_chunker.create_chunks(page.page_content, page.metadata)
            chunks.extend(page_chunks)
        return chunks

    def _generate_hyde_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate HyDE chunks if enabled."""
        hyde_generator = HyDEGenerator()
        hyde_chunks = []
        for chunk in chunks:
            questions = hyde_generator.generate_hyde_questions(chunk['content'])
            for q in questions:
                hyde_chunks.append({
                    'content': f"Q: {q['question']}\nA: {q['answer']}",
                    'metadata': {**chunk['metadata'], 'type': 'hyde'}
                })
        return hyde_chunks

    def _prepare_datastax_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare chunks for Datastax format."""
        return [
            {
                'content': chunk['content'],
                'metadata': {
                    k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in chunk['metadata'].items()
                }
            }
            for chunk in chunks
        ]



############################################################################################################
# src/indexing/data_loader.py
import os
import logging
from typing import Dict, List, Any
from langchain_community.document_loaders import PyPDFLoader
from utils.config_loader import load_config
from src.exceptions.loader_exceptions import DataLoaderError

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles document loading from various sources."""
    
    def __init__(self):
        self.config = load_config()
        self.data_folder = self.config['data']['data_folder']

    def load_document(self) -> Dict[str, List[Any]]:
        """Load documents from the configured data folder."""
        if not self.data_folder:
            logger.error("Data folder is not specified in the configuration file.")
            raise DataLoaderError("Data folder is not defined in the configuration.")
            
        logger.info(f"Data folder set to: {self.data_folder}")
        
        data = {}
        try:
            for filename in sorted(os.listdir(self.data_folder), reverse=True):
                if filename.endswith(".pdf"):
                    data[filename] = self._load_pdf(filename)
            return data
        except Exception as e:
            logger.error(f"Error in document loading: {e}")
            raise DataLoaderError(f"Failed to load documents: {str(e)}")

    def _load_pdf(self, filename: str) -> List[Any]:
        """Load a PDF file and its pages."""
        try:
            logger.info(f"Processing file: {filename}")
            filepath = os.path.join(self.data_folder, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            
            # Enrich metadata
            for page in pages:
                page.metadata.update({
                    'source': filename,
                    'file_path': filepath,
                })
            
            return pages
        except Exception as e:
            logger.error(f"Error loading PDF {filename}: {e}")
            raise DataLoaderError(f"Failed to load PDF {filename}: {str(e)}")

    @staticmethod
    def validate_document(document: Dict[str, Any]) -> bool:
        """Validate loaded document structure."""
        try:
            if not isinstance(document, dict):
                return False
            
            required_metadata = ['source', 'file_path']
            for pages in document.values():
                for page in pages:
                    if not hasattr(page, 'metadata'):
                        return False
                    if not all(key in page.metadata for key in required_metadata):
                        return False
            return True
        except Exception:
            return False




############################################################################################################

# src/query_serving/querying.py
import logging
from typing import Dict, Any, List
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from src.exceptions.query_exceptions import QueryError
from utils.config_loader import load_config
from src.indexing.datastax import DatastaxConnector

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    content: str
    metadata: Dict[str, Any]
    score: float
    source_document: str

class QueryProcessor:
    """Handles query processing and retrieval."""
    
    def __init__(self):
        self.config = load_config()
        self.embedding = SentenceTransformer(self.config['embedding_model']['embed_model'])

    def fetch_data(self, query: str, file: str = None) -> List[QueryResult]:
        """Fetch relevant data for a query."""
        try:
            # Prepare query payload
            read_payload_data = {
                "usecaseId": self.config['project']['usecase_id'],
                "metadata": {
                    "author": self.config['project']['author'],
                    "usecase": self.config['project']['usecase'],
                    "document": self.config['project']['document_name'],
                    "source": file
                },
                "limit": self.config['preprocessing']['k'],
                "threshold": self.config['project']['threshold'],
                "vector": self.embedding.encode(query).tolist()
            }
            
            # Query the vector store
            conn = DatastaxConnector()
            results = conn.read_vector(read_payload_data)
            
            # Process results
            processed_results = []
            for result in results.get('items', []):
                processed_results.append(
                    QueryResult(
                        content=result.get('bodyBlob', ''),
                        metadata=result.get('metadata', {}),
                        score=result.get('score', 0.0),
                        source_document=file or 'unknown'
                    )
                )
            
            return processed_results
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            raise QueryError(f"Failed to process query: {str(e)}")

class QueryRouter:
    """Routes queries to appropriate processors based on type."""
    
    def __init__(self):
        self.processor = QueryProcessor()
        self.config = load_config()

    def process_query(self, query: str, query_type: str, file: str = None) -> List[QueryResult]:
        """Process a query based on its type."""
        try:
            if query_type == "document":
                return self.processor.fetch_data(query, file)
            elif query_type == "hyde":
                return self._process_hyde_query(query, file)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            raise QueryError(f"Query routing failed: {str(e)}")

    def _process_hyde_query(self, query: str, file: str) -> List[QueryResult]:
        """Process a HyDE-type query."""
        # Implementation for HyDE query processing
        pass

# src/query_serving/response_formatter.py
from typing import List, Dict, Any

class ResponseFormatter:
    """Formats query results for presentation."""
    
    @staticmethod
    def format_results(results: List[QueryResult]) -> Dict[str, Any]:
        """Format query results into a structured response."""
        return {
            "results": [
                {
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "source": result.source_document
                }
                for result in results
            ],
            "total_results": len(results),
            "summary": {
                "avg_score": sum(r.score for r in results) / len(results) if results else 0,
                "sources": list(set(r.source_document for r in results))
            }
        }



############################################################################################################

# src/indexing/retriever/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseRetriever(ABC):
    """Base class for document retrievers."""
    
    @abstractmethod
    def get_relevant_documents(self, query: str, filter_dict: Dict[str, str] = None) -> List[Any]:
        """Retrieve relevant documents for a query."""
        pass

# src/indexing/retriever/automerger.py
import logging
from typing import List, Dict, Any, Optional
from src.indexing.retriever.base import BaseRetriever

logger = logging.getLogger(__name__)

class AutoMergingRetriever(BaseRetriever):
    """Implements auto-merging retrieval strategy."""
    
    def __init__(self, vector_db, docstore, simple_ratio_threshold: float = 0.3):
        self.vector_db = vector_db
        self.docstore = docstore
        self.simple_ratio_threshold = simple_ratio_threshold

    def get_relevant_documents(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """Get relevant documents with auto-merging strategy."""
        try:
            # Initial retrieval
            base_docs = self.vector_db.similarity_search_with_score(
                query,
                filter=filter_dict
            )

            # Process and merge documents
            merged_docs = self._merge_documents(base_docs)
            
            # Sort by relevance score
            merged_docs.sort(key=lambda x: x[1], reverse=True)
            
            return merged_docs
        except Exception as e:
            logger.error(f"Error in auto-merging retrieval: {e}")
            raise

    def _merge_documents(self, docs: List[Any]) -> List[Any]:
        """Merge similar or overlapping documents."""
        try:
            merged = []
            used = set()
            
            for i, (doc1, score1) in enumerate(docs):
                if i in used:
                    continue
                    
                current_content = doc1.page_content
                current_metadata = doc1.metadata.copy()
                current_score = score1
                
                # Look for similar documents to merge
                for j, (doc2, score2) in enumerate(docs[i+1:], i+1):
                    if j in used:
                        continue
                        
                    if self._should_merge(doc1, doc2):
                        current_content += f"\n{doc2.page_content}"
                        current_metadata.update(doc2.metadata)
                        current_score = max(current_score, score2)
                        used.add(j)
                
                merged.append((
                    self._create_merged_document(current_content, current_metadata),
                    current_score
                ))
            
            return merged
        except Exception as e:
            logger.error(f"Error merging documents: {e}")
            raise

    def _should_merge(self, doc1: Any, doc2: Any) -> bool:
        """Determine if two documents should be merged."""
        # Simple overlap ratio calculation
        words1 = set(doc1.page_content.split())
        words2 = set(doc2.page_content.split())
        overlap = len(words1.intersection(words2)) / len(words1.union(words2))
        
        # Check metadata similarity
        same_source = doc1.metadata.get('source') == doc2.metadata.get('source')
        sequential_pages = self._are_sequential_pages(doc1.metadata, doc2.metadata)
        
        return (overlap > self.simple_ratio_threshold and same_source) or sequential_pages

    def _are_sequential_pages(self, meta1: Dict[str, Any], meta2: Dict[str, Any]) -> bool:
        """Check if documents are from sequential pages."""
        try:
            page1 = meta1.get('page', -1)
            page2 = meta2.get('page', -1)
            return abs(page1 - page2) == 1
        except:
            return False

    def _create_merged_document(self, content: str, metadata: Dict[str, Any]) -> Any:
        """Create a new document from merged content and metadata."""
        from dataclasses import dataclass
        
        @dataclass
        class Document:
            page_content: str
            metadata: Dict[str, Any]
        
        return Document(
            page_content=content,
            metadata=metadata
        )

# src/indexing/retriever/vanilla.py
from typing import List, Dict, Any, Optional
from src.indexing.retriever.base import BaseRetriever

class VanillaRetriever(BaseRetriever):
    """Implements simple vector similarity search."""
    
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def get_relevant_documents(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, str]] = None
    ) -> List[Any]:
        """Get relevant documents using simple similarity search."""
        return self.vector_db.similarity_search_with_score(
            query,
            filter=filter_dict
        )




############################################################################################################


# src/exceptions/base.py
class LumosError(Exception):
    """Base exception for Lumos application."""
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

# src/exceptions/loader_exceptions.py
class DataLoaderError(LumosError):
    """Raised when there's an error loading data."""
    pass

class FileNotSupportedError(DataLoaderError):
    """Raised when file type is not supported."""
    pass

class FileReadError(DataLoaderError):
    """Raised when there's an error reading a file."""
    pass

class DirectoryNotFoundError(DataLoaderError):
    """Raised when the specified directory doesn't exist."""
    pass

# src/exceptions/index_exceptions.py
class IndexError(LumosError):
    """Base class for indexing errors."""
    pass

class IndexCreationError(IndexError):
    """Raised when there's an error creating the index."""
    pass

class IndexLoadError(IndexError):
    """Raised when there's an error loading the index."""
    pass

class IndexUpdateError(IndexError):
    """Raised when there's an error updating the index."""
    pass

class IndexSaveError(IndexError):
    """Raised when there's an error saving the index."""
    pass

class IndexSearchError(IndexError):
    """Raised when there's an error searching the index."""
    pass

# src/exceptions/query_exceptions.py
class QueryError(LumosError):
    """Base class for query-related errors."""
    pass

class InvalidQueryError(QueryError):
    """Raised when the query is invalid."""
    pass

class QueryProcessingError(QueryError):
    """Raised when there's an error processing the query."""
    pass

class QueryTimeoutError(QueryError):
    """Raised when a query operation times out."""
    pass

class QueryFilterError(QueryError):
    """Raised when there's an error with query filters."""
    pass

# src/exceptions/config_exceptions.py
class ConfigError(LumosError):
    """Base class for configuration errors."""
    pass

class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""
    pass

class ConfigLoadError(ConfigError):
    """Raised when configuration loading fails."""
    pass

class ConfigKeyError(ConfigError):
    """Raised when a required configuration key is missing."""
    pass

class ConfigTypeError(ConfigError):
    """Raised when a configuration value has an incorrect type."""
    pass

# src/exceptions/chunking_exceptions.py
class ChunkingError(LumosError):
    """Base class for chunking-related errors."""
    pass

class ChunkSizeError(ChunkingError):
    """Raised when there's an issue with chunk sizing."""
    pass

class ChunkProcessingError(ChunkingError):
    """Raised when there's an error processing chunks."""
    pass

class ChunkMergeError(ChunkingError):
    """Raised when there's an error merging chunks."""
    pass

class ChunkValidationError(ChunkingError):
    """Raised when chunk validation fails."""
    pass

# src/exceptions/embedding_exceptions.py
class EmbeddingError(LumosError):
    """Base class for embedding-related errors."""
    pass

class ModelLoadError(EmbeddingError):
    """Raised when there's an error loading the embedding model."""
    pass

class EmbeddingGenerationError(EmbeddingError):
    """Raised when there's an error generating embeddings."""
    pass

class EmbeddingDimensionError(EmbeddingError):
    """Raised when there's a mismatch in embedding dimensions."""
    pass

class ModelNotFoundError(EmbeddingError):
    """Raised when the specified model cannot be found."""
    pass

# src/exceptions/vectorstore_exceptions.py
class VectorStoreError(LumosError):
    """Base class for vector store related errors."""
    pass

class VectorStoreConnectionError(VectorStoreError):
    """Raised when there's an error connecting to the vector store."""
    pass

class VectorStoreQueryError(VectorStoreError):
    """Raised when there's an error querying the vector store."""
    pass

class VectorStoreInsertError(VectorStoreError):
    """Raised when there's an error inserting into the vector store."""
    pass

class VectorStoreDeleteError(VectorStoreError):
    """Raised when there's an error deleting from the vector store."""
    pass

# src/exceptions/llm_exceptions.py
class LLMError(LumosError):
    """Base class for LLM-related errors."""
    pass

class LLMConnectionError(LLMError):
    """Raised when there's an error connecting to the LLM service."""
    pass

class LLMResponseError(LLMError):
    """Raised when there's an error in LLM response."""
    pass

class LLMTokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass

class LLMTimeoutError(LLMError):
    """Raised when LLM request times out."""
    pass

# src/exceptions/auth_exceptions.py
class AuthError(LumosError):
    """Base class for authentication-related errors."""
    pass

class TokenError(AuthError):
    """Raised when there's an error with authentication tokens."""
    pass

class CredentialsError(AuthError):
    """Raised when there's an error with credentials."""
    pass

class PermissionError(AuthError):
    """Raised when there's an error with permissions."""
    pass

# src/exceptions/preprocessing_exceptions.py
class PreprocessingError(LumosError):
    """Base class for preprocessing-related errors."""
    pass

class TextCleaningError(PreprocessingError):
    """Raised when there's an error cleaning text."""
    pass

class MetadataExtractionError(PreprocessingError):
    """Raised when there's an error extracting metadata."""
    pass

class DocumentParsingError(PreprocessingError):
    """Raised when there's an error parsing documents."""
    pass

# src/exceptions/validation_exceptions.py
class ValidationError(LumosError):
    """Base class for validation-related errors."""
    pass

class InputValidationError(ValidationError):
    """Raised when input validation fails."""
    pass

class OutputValidationError(ValidationError):
    """Raised when output validation fails."""
    pass

class SchemaValidationError(ValidationError):
    """Raised when schema validation fails."""
    pass



############################################################################################################



# src/indexing/datastax.py
import requests
import logging
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from utils.config_loader import load_config
from src.exceptions.vectorstore_exceptions import (
    VectorStoreConnectionError,
    VectorStoreQueryError,
    VectorStoreInsertError
)
from src.exceptions.auth_exceptions import TokenError

logger = logging.getLogger(__name__)

@dataclass
class AidaConfig:
    """Configuration for AIDA API."""
    user_name: str
    api_url: str
    secret: str
    certificate_path: str

class DatastaxConnector:
    """Handles connection and operations with Datastax through AIDA."""
    
    def __init__(self):
        self.config = load_config()
        self.aida_config = self._load_aida_config()
        self.token: Optional[str] = None
        self._initialize_connection()

    def _load_aida_config(self) -> AidaConfig:
        """Load AIDA configuration from environment."""
        try:
            return AidaConfig(
                user_name=os.getenv("AIDA_API_USER"),
                api_url=os.getenv("AIDA_API_URL"),
                secret=os.getenv("AIDA_API_SECRET"),
                certificate_path=self.config['auth']['certificate_path']
            )
        except Exception as e:
            logger.error(f"Error loading AIDA config: {e}")
            raise ConfigLoadError("Failed to load AIDA configuration")

    def _initialize_connection(self) -> None:
        """Initialize connection and get authentication token."""
        if not self.token:
            self.token = self._get_okta_token()

    def _get_okta_token(self) -> str:
        """Get authentication token from Okta."""
        try:
            data = f'grant_type=password&username={self.aida_config.user_name}&password={self.aida_config.secret}&scope=openid'
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": self.config['auth']['basic']
            }
            proxies = {
                'http': self.config['proxies']['http'],
                'https': self.config['proxies']['https']
            }
            
            response = requests.post(
                self.config['auth']['auth_url'],
                verify=False,  # In production, should use proper certificate verification
                data=data,
                headers=headers,
                proxies=proxies
            )
            
            if response.status_code == 200:
                return response.json()['access_token']
            else:
                raise TokenError(f"Failed to get token. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error getting Okta token: {e}")
            raise TokenError(f"Token acquisition failed: {str(e)}")

    def _get_request_header(self) -> Dict[str, str]:
        """Get headers for AIDA API requests."""
        return {
            "Content-Type": "application/json",
            "one-data-correlation-id": str(uuid.uuid4()),
            "Authorization": f"claims impersonation_id={self.aida_config.user_name};app_authorization={self.token}"
        }

    def build_data(self, texts: List[str], metas: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build data payload for vector store operations."""
        return {
            "usecaseId": self.config['project']['usecase_id'],
            "metadata": {
                "author": self.config['project']['author'],
                "usecase": self.config['project']['usecase'],
                "document": self.config['project']['document_name']
            },
            "data": [
                {
                    "bodyBlob": text,
                    "metadata": meta
                }
                for text, meta in zip(texts, metas)
            ]
        }

    def create_vector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create vector embeddings in Datastax."""
        try:
            response = self._make_api_call(
                f"{self.aida_config.api_url}/createGenAiKnowledgeStoreVector.v1",
                data
            )
            return response
        except Exception as e:
            logger.error(f"Error creating vector: {e}")
            raise VectorStoreInsertError(f"Failed to create vector: {str(e)}")

    def read_vector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Read vectors from Datastax."""
        try:
            response = self._make_api_call(
                f"{self.aida_config.api_url}/readAiDaFeatureOpsVector.v1",
                data
            )
            return response
        except Exception as e:
            logger.error(f"Error reading vector: {e}")
            raise VectorStoreQueryError(f"Failed to read vector: {str(e)}")

    def _make_api_call(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make API call to AIDA endpoint."""
        try:
            session = requests.Session()
            session.trust_env = False
            
            response = session.post(
                url=url,
                json=data,
                headers=self._get_request_header(),
                verify=False,  # In production, should use proper certificate verification
                proxies={
                    'http': self.config['proxies']['http'],
                    'https': self.config['proxies']['https']
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                raise VectorStoreConnectionError(
                    f"API call failed. Status: {response.status_code}, Response: {response.text}"
                )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise VectorStoreConnectionError(f"API call failed: {str(e)}")



############################################################################################################

# src/auth/auth_manager.py
import os
import logging
from typing import Dict, Any, Optional
import requests
from dataclasses import dataclass
from src.exceptions.auth_exceptions import (
    TokenError,
    CredentialsError,
    PermissionError
)
from utils.config_loader import load_config

logger = logging.getLogger(__name__)

@dataclass
class AuthCredentials:
    """Authentication credentials."""
    username: str
    secret: str
    scope: str = 'openid'

@dataclass
class AuthConfig:
    """Authentication configuration."""
    auth_url: str
    basic_auth: str
    certificate_path: str
    proxy_config: Dict[str, str]

class AuthenticationManager:
    """Manages authentication and token lifecycle."""
    
    def __init__(self):
        self.config = load_config()
        self.auth_config = self._load_auth_config()
        self.credentials = self._load_credentials()
        self.token: Optional[str] = None
        self._validate_configuration()

    def _load_auth_config(self) -> AuthConfig:
        """Load authentication configuration."""
        try:
            return AuthConfig(
                auth_url=self.config['auth']['auth_url'],
                basic_auth=self.config['auth']['basic'],
                certificate_path=self.config['auth']['certificate_path'],
                proxy_config=self.config['proxies']
            )
        except KeyError as e:
            logger.error(f"Missing auth configuration: {e}")
            raise ConfigLoadError(f"Missing required auth configuration: {str(e)}")

    def _load_credentials(self) -> AuthCredentials:
        """Load credentials from environment."""
        username = os.getenv("AIDA_API_USER")
        secret = os.getenv("AIDA_API_SECRET")
        
        if not username or not secret:
            raise CredentialsError("Missing required credentials")
            
        return AuthCredentials(username=username, secret=secret)

    def _validate_configuration(self) -> None:
        """Validate authentication configuration."""
        if not os.path.exists(self.auth_config.certificate_path):
            raise ConfigValidationError(f"Certificate not found: {self.auth_config.certificate_path}")

    def get_token(self, force_refresh: bool = False) -> str:
        """Get authentication token."""
        if self.token is None or force_refresh:
            self.token = self._fetch_new_token()
        return self.token

    def _fetch_new_token(self) -> str:
        """Fetch new authentication token."""
        try:
            data = {
                'grant_type': 'password',
                'username': self.credentials.username,
                'password': self.credentials.secret,
                'scope': self.credentials.scope
            }
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": self.auth_config.basic_auth
            }
            
            response = requests.post(
                self.auth_config.auth_url,
                data=data,
                headers=headers,
                verify=self.auth_config.certificate_path,
                proxies=self.auth_config.proxy_config
            )
            
            if response.status_code == 200:
                token_data = response.json()
                return token_data['access_token']
            elif response.status_code == 401:
                raise PermissionError("Authentication failed: Invalid credentials")
            else:
                raise TokenError(f"Failed to obtain token. Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during authentication: {e}")
            raise TokenError(f"Network error during authentication: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during authentication: {e}")
            raise TokenError(f"Authentication failed: {str(e)}")

# src/auth/certificate_manager.py
import os
import logging
from typing import Optional
from datetime import datetime
from cryptography import x509
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)

class CertificateManager:
    """Manages SSL certificates."""
    
    def __init__(self, cert_path: str):
        self.cert_path = cert_path
        self._validate_certificate()

    def _validate_certificate(self) -> None:
        """Validate certificate existence and expiration."""
        try:
            if not os.path.exists(self.cert_path):
                raise ValueError(f"Certificate not found: {self.cert_path}")
                
            with open(self.cert_path, 'rb') as cert_file:
                cert_data = cert_file.read()
                cert = x509.load_pem_x509_certificate(cert_data, default_backend())
                
            if datetime.utcnow() > cert.not_valid_after:
                raise ValueError(f"Certificate expired on {cert.not_valid_after}")
                
            logger.info(f"Certificate valid until {cert.not_valid_after}")
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            raise

# src/auth/proxy_manager.py
import logging
from typing import Dict, Optional
import requests

logger = logging.getLogger(__name__)

class ProxyManager:
    """Manages proxy configuration."""
    
    def __init__(self, proxy_config: Dict[str, str]):
        self.proxy_config = proxy_config
        self._validate_proxy_config()

    def _validate_proxy_config(self) -> None:
        """Validate proxy configuration."""
        required_keys = ['http', 'https']
        for key in required_keys:
            if key not in self.proxy_config:
                raise ValueError(f"Missing required proxy configuration: {key}")

    def get_proxy_dict(self) -> Dict[str, str]:
        """Get proxy configuration dictionary."""
        return self.proxy_config

    def test_proxy_connection(self) -> bool:
        """Test proxy connection."""
        try:
            response = requests.get(
                'https://www.google.com',
                proxies=self.proxy_config,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Proxy connection test failed: {e}")
            return False




############################################################################################################

# src/utils/logging_utils.py
import logging
import sys
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_record = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
            
        return json.dumps(log_record)

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """Setup application logging."""
    
    # Create formatters
    json_formatter = CustomJsonFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(json_formatter)
        root_logger.addHandler(file_handler)

# src/utils/metric_collector.py
import time
from typing import Dict, Any, List
from dataclasses import data



############################################################################################################






############################################################################################################


