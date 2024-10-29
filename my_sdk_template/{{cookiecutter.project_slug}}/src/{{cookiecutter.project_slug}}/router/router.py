from typing import Any, Dict
from .services.lcel import LCELChainBuilder
from .model.models import get_model
from .prompts.prompt_generator import create_prompt_template
from .outputparser.output_parser import create_output_parser

class Router:
    def __init__(self):
        self.model = get_model()
        self.chain_builder = LCELChainBuilder()
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Create prompt template
        prompt = create_prompt_template(input_data.get("prompt_template", ""))
        
        # Create output parser
        output_parser = create_output_parser(input_data.get("output_format", "json"))
        
        # Build and run chain
        chain = self.chain_builder.build_chain(
            prompt=prompt,
            model=self.model,
            output_parser=output_parser
        )
        
        return chain.invoke(input_data)