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

class KGTripletList(BaseModel):
    triplets: List[KGTriplet] = Field(description="List of knowledge graph triplets")

class TextToKGProcessor:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            api_key=api_key
        )
        self.parser = PydanticOutputParser(pydantic_object=KGTripletList)
        
        self.prompt = PromptTemplate(
            template="""Extract ALL possible knowledge graph triplets from the following text chunk.
            Generate as many meaningful triplets as you can find in the text.
            Each triplet should be in the format (subject, predicate, object).
            
            Text chunk: {text}
            
            {format_instructions}
            
            Remember to:
            1. Break down complex sentences into multiple triplets
            2. Include implicit relationships
            3. Extract both direct and indirect relationships
            4. Consider attributes and properties as separate triplets
            """,
            input_variables=["text"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
    def process_chunk(self, text: str) -> List[Tuple[str, str, str]]:
        chain = self.prompt | self.llm | self.parser
        try:
            result = chain.invoke({"text": text})
            return [triplet.to_tuple() for triplet in result.triplets]
        except Exception as e:
            print(f"Error processing chunk: {e}")
            return []
    
    def process_chunks(self, chunks: List[str]) -> List[Tuple[str, str, str]]:
        all_triplets = []
        for chunk in chunks:
            chunk_triplets = self.process_chunk(chunk)
            all_triplets.extend(chunk_triplets)
        return all_triplets

# Example usage:
if __name__ == "__main__":
    api_key = "your-openai-api-key"
    processor = TextToKGProcessor(api_key)
    
    text_chunks = [
        """The large brown cat sits on the comfortable mat in the kitchen. 
        The cat belongs to John, who bought it from a local shelter last year.""",
        "Paris, the beautiful capital of France, attracts millions of tourists annually.",
    ]
    
    kg_triplets = processor.process_chunks(text_chunks)
    print("Generated Knowledge Graph Triplets:")
    for triplet in kg_triplets:
        print(f"({triplet[0]}, {triplet[1]}, {triplet[2]})")
