
from typing import List, Tuple
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

class KGTriplet(BaseModel):
    subject: str = Field(description="Subject of the triplet")
    predicate: str = Field(description="Predicate/relation of the triplet")
    object: str = Field(description="Object of the triplet")
    
    def to_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

class TextToKGProcessor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=KGTriplet)
        
        self.prompt = PromptTemplate(
            template="""Extract a knowledge graph triplet from the following text chunk.
            The output should be in the format (subject, predicate, object).
            
            Text chunk: {text}
            
            {format_instructions}
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
    def process_chunk(self, text: str) -> KGTriplet:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({"text": text})
    
    def process_chunks(self, chunks: List[str]) -> List[Tuple[str, str, str]]:
        triplets = []
        for chunk in chunks:
            try:
                kg_triplet = self.process_chunk(chunk)
                triplets.append(kg_triplet.to_tuple())
            except Exception as e:
                print(f"Error processing chunk: {e}")
        return triplets

# Example usage:
if __name__ == "__main__":
    api_key = "your-openai-api-key"
    processor = TextToKGProcessor(api_key)
    
    text_chunks = [
        "The cat sits on the mat.",
        "Paris is the capital of France.",
        "Einstein developed the theory of relativity."
    ]
    
    kg_triplets = processor.process_chunks(text_chunks)
    for triplet in kg_triplets:
        print(f"Subject: {triplet[0]}, Predicate: {triplet[1]}, Object: {triplet[2]}")
