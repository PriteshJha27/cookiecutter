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




