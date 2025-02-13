
# src/query_serving/models.py
from typing import TypedDict, List, Dict, Union, Optional
from pydantic import BaseModel

class QueryState(TypedDict):
    """State management for query processing."""
    question: str
    theme: Dict[str, Union[str, bool]]
    rewritten_question: str
    retrieved_results: List[Dict[str, str]]
    context: str
    reranked_documents: List[Dict[str, str]]
    llm_result: str
    validation_results: Dict[str, Union[List[str], float, str, bool]]
    suggestions: List[str]
    corrected_prompt: str
    correction_attempt: int
    continue_loop: bool
    question_counter: int

class ValidationResult(BaseModel):
    """Validation results for LLM response."""
    evaluation_score: float
    is_satisfactory: bool
    reasoning: str
    improvement_suggestions: List[str]

class RetrievalResult(BaseModel):
    """Results from vector store retrieval."""
    content: str
    metadata: Dict[str, any]
    score: float
    source: str

# src/query_serving/config.py
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class QueryConfig:
    """Configuration for query serving pipeline."""
    max_correction_attempts: int = 2
    validation_threshold: float = 0.8
    retrieval_k: int = 4
    reranking_threshold: float = 0.7
    max_context_length: int = 4096
    model_name: str = "gpt"

# src/query_serving/exceptions.py
from src.exceptions.base import LumosError

class QueryProcessingError(LumosError):
    """Base class for query processing errors."""
    pass

class ThemeClassificationError(QueryProcessingError):
    """Error in query theme classification."""
    pass

class RetrievalError(QueryProcessingError):
    """Error in document retrieval."""
    pass

class ValidationError(QueryProcessingError):
    """Error in response validation."""
    pass

class CorrectionError(QueryProcessingError):
    """Error in response correction."""
    pass







___________________________________________________________________________________________________________________________


# src/query_serving/orchestrator.py
import logging
from typing import Dict, Any, Optional
from .models import QueryState, QueryConfig
from .nodes import (
    UserInputNode,
    QueryRewriteNode,
    RetrievalNode,
    RerankerNode,
    LLMChainNode,
    ValidationNode,
    CorrectionNode
)

logger = logging.getLogger(__name__)

class QueryOrchestrator:
    """Orchestrates the query serving pipeline."""
    
    def __init__(self, config: QueryConfig):
        self.config = config
        self._initialize_nodes()
        self.state = self._initialize_state()

    def _initialize_nodes(self) -> None:
        """Initialize all pipeline nodes."""
        self.input_node = UserInputNode()
        self.rewrite_node = QueryRewriteNode()
        self.retrieval_node = RetrievalNode()
        self.reranker_node = RerankerNode()
        self.llm_node = LLMChainNode()
        self.validation_node = ValidationNode()
        self.correction_node = CorrectionNode()

    def _initialize_state(self) -> QueryState:
        """Initialize query state."""
        return QueryState(
            question="",
            theme={},
            rewritten_question="",
            retrieved_results=[],
            context="",
            reranked_documents=[],
            llm_result="",
            validation_results={},
            suggestions=[],
            corrected_prompt="",
            correction_attempt=1,
            continue_loop=True,
            question_counter=1
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query through the pipeline."""
        try:
            self.state['question'] = query
            
            while self.state['continue_loop']:
                # Execute pipeline nodes
                self.state = self.rewrite_node.execute(self.state)
                self.state = self.retrieval_node.execute(self.state)
                self.state = self.reranker_node.execute(self.state)
                self.state = self.llm_node.execute(self.state)
                self.state = self.validation_node.execute(self.state)

                # Check if correction is needed
                if self._needs_correction():
                    if self.state['correction_attempt'] < self.config.max_correction_attempts:
                        self.state = self.correction_node.execute(self.state)
                        self.state['correction_attempt'] += 1
                    else:
                        self.state['continue_loop'] = False
                else:
                    self.state['continue_loop'] = False

            return self._prepare_response()

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise QueryProcessingError(f"Failed to process query: {str(e)}")

    def _needs_correction(self) -> bool:
        """Determine if response needs correction."""
        score = self.state['validation_results'].get('evaluation_score', 0.0)
        return score < self.config.validation_threshold

    def _prepare_response(self) -> Dict[str, Any]:
        """Prepare final response."""
        return {
            'answer': self.state['llm_result'],
            'validation_score': self.state['validation_results'].get('evaluation_score', 0.0),
            'context_used': self.state['context'],
            'improvements_made': self.state['correction_attempt'] - 1
        }

    def get_state(self) -> QueryState:
        """Get current state."""
        return self.state

    def reset_state(self) -> None:
        """Reset state for new query."""
        self.state = self._initialize_state()
___________________________________________________________________________________________________________________________

# src/query_serving/nodes/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..models import QueryState

class BaseNode(ABC):
    """Base class for all pipeline nodes."""
    
    @abstractmethod
    def execute(self, state: QueryState) -> QueryState:
        """Execute node's main functionality."""
        pass

# src/query_serving/nodes/query_rewrite.py
import logging
from typing import Dict, Any
from .base import BaseNode
from ..models import QueryState
from src.llm.theme_classifier import ClassifyTheme
from src.exceptions.query_exceptions import ThemeClassificationError

logger = logging.getLogger(__name__)

class QueryRewriteNode(BaseNode):
    """Handles query analysis and rewriting."""
    
    def __init__(self):
        self.theme_classifier = ClassifyTheme()

    def execute(self, state: QueryState) -> QueryState:
        """Execute query rewriting."""
        try:
            logger.info(f"Rewriting query: {state['question']}")
            
            # Classify query theme
            theme_results, rewritten_query = self.theme_classifier.process_query(
                state['question']
            )
            
            state['theme'] = theme_results
            state['rewritten_question'] = rewritten_query
            
            return state
        except Exception as e:
            logger.error(f"Error in query rewriting: {e}")
            raise ThemeClassificationError(f"Failed to rewrite query: {str(e)}")

# src/query_serving/nodes/retrieval.py
from .base import BaseNode
from ..models import QueryState
from src.indexing.datastax import DatastaxConnector
from src.exceptions.query_exceptions import RetrievalError

class RetrievalNode(BaseNode):
    """Handles document retrieval from vector store."""
    
    def __init__(self):
        self.datastax = DatastaxConnector()

    def execute(self, state: QueryState) -> QueryState:
        """Execute document retrieval."""
        try:
            retrieved_results, context = self.datastax.retrieve(
                query=state['rewritten_question'],
                filter={"type": "doc"}
            )
            
            state['retrieved_results'] = retrieved_results
            state['context'] = context
            
            return state
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {str(e)}")

# src/query_serving/nodes/reranker.py
from .base import BaseNode
from ..models import QueryState
from src.llm.reranker import Reranked

class RerankerNode(BaseNode):
    """Handles semantic reranking of retrieved documents."""
    
    def execute(self, state: QueryState) -> QueryState:
        """Execute document reranking."""
        reranker = Reranked()
        state['reranked_documents'] = reranker.process(
            state['rewritten_question'],
            state['retrieved_results']
        )
        return state

# src/query_serving/nodes/llm_chain.py
from .base import BaseNode
from ..models import QueryState
from src.llm.chain import LLMChain
from src.utils.context_builder import get_retrieval_context

class LLMChainNode(BaseNode):
    """Handles LLM response generation."""
    
    def execute(self, state: QueryState) -> QueryState:
        """Execute LLM chain."""
        context = get_retrieval_context(state['reranked_documents'])
        chain = LLMChain()
        state['llm_result'] = chain.run(
            question=state['rewritten_question'],
            context=context
        )
        return state

# src/query_serving/nodes/validation.py
from .base import BaseNode
from ..models import QueryState
from src.llm.validator import ResponseValidator

class ValidationNode(BaseNode):
    """Handles response validation."""
    
    def execute(self, state: QueryState) -> QueryState:
        """Execute response validation."""
        validator = ResponseValidator()
        validation_results = validator.validate(
            query=state['rewritten_question'],
            response=state['llm_result']
        )
        state['validation_results'] = validation_results
        return state

# src/query_serving/nodes/correction.py
from .base import BaseNode
from ..models import QueryState
from src.llm.corrector import ResponseCorrector

class CorrectionNode(BaseNode):
    """Handles response correction."""
    
    def execute(self, state: QueryState) -> QueryState:
        """Execute response correction."""
        corrector = ResponseCorrector()
        corrected_response = corrector.improve_response(
            query=state['rewritten_question'],
            response=state['llm_result'],
            suggestions=state['validation_results']['improvement_suggestions']
        )
        state['corrected_prompt'] = corrected_response
        return state

___________________________________________________________________________________________________________________________


# src/main_query.py
import logging
from typing import Dict, Any
from src.query_serving.orchestrator import QueryOrchestrator
from src.query_serving.models import QueryConfig
from src.utils.config_loader import load_config
from src.exceptions.query_exceptions import QueryProcessingError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QueryRouter:
    """Routes and manages query processing."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)
        self.query_config = self._load_query_config()
        self.orchestrator = QueryOrchestrator(self.query_config)

    def _load_query_config(self) -> QueryConfig:
        """Load query configuration."""
        return QueryConfig(
            max_correction_attempts=self.config.get('query', {}).get('max_correction_attempts', 2),
            validation_threshold=self.config.get('query', {}).get('validation_threshold', 0.8),
            retrieval_k=self.config.get('query', {}).get('retrieval_k', 4),
            reranking_threshold=self.config.get('query', {}).get('reranking_threshold', 0.7),
            max_context_length=self.config.get('query', {}).get('max_context_length', 4096),
            model_name=self.config.get('llm', {}).get('model', 'gpt')
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a single query."""
        try:
            logger.info(f"Processing query: {query}")
            response = self.orchestrator.process_query(query)
            logger.info("Query processing completed successfully")
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise QueryProcessingError(f"Failed to process query: {str(e)}")

    def reset(self) -> None:
        """Reset the orchestrator state."""
        self.orchestrator.reset_state()

def main():
    """Main entry point for query processing."""
    try:
        router = QueryRouter()
        
        while True:
            query = input("Enter your question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
                
            try:
                response = router.process_query(query)
                print("\nAnswer:", response['answer'])
                print(f"Confidence Score: {response['validation_score']:.2f}")
                print(f"Improvements Made: {response['improvements_made']}")
            except Exception as e:
                print(f"Error processing query: {e}")
            
            router.reset()
            print("\n---")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()

___________________________________________________________________________________________________________________________




___________________________________________________________________________________________________________________________
