from typing import Any, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import BaseOutputParser
from langchain.schema.language_model import BaseLanguageModel

class LCELChainBuilder:
    def build_chain(
        self,
        prompt: ChatPromptTemplate,
        model: BaseLanguageModel,
        output_parser: Optional[BaseOutputParser] = None
    ) -> Any:
        """
        Build an LCEL chain with the given components
        """
        if output_parser:
            return prompt | model | output_parser
        return prompt | model