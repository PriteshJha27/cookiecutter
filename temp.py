from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

def create_correction_chain(llm: ChatOpenAI = ChatOpenAI()):
    # Create prompt template for generating improved response
    prompt = ChatPromptTemplate.from_template("""
    As an expert assistant, improve the given response based on the evaluation feedback.
    
    Original Query: {query}
    
    Original Response: {response}
    
    Improvement Suggestions:
    {suggestions}
    
    Create an improved response that:
    1. Addresses all improvement suggestions
    2. Maintains accuracy and relevance
    3. Improves clarity and completeness
    4. Preserves any correct information from the original response
    
    Return a JSON object with:
    - improved_response: the enhanced response
    - changes_made: list of specific improvements implemented
    - confidence_score: number between 0 and 1 indicating confidence in improvements
    
    Focus on meaningful improvements while maintaining natural language flow.
    """)
    
    # Create JSON output parser
    output_parser = JsonOutputParser()
    
    # Create the chain
    chain = (
        {
            "query": RunnablePassthrough(),
            "response": lambda x: str(x["response"]),
            "suggestions": lambda x: str(x["suggestions"])
        }
        | prompt
        | llm
        | output_parser
    )
    
    return chain

def create_combined_correction_chain(llm: ChatOpenAI = ChatOpenAI()):
    """Creates a chain that combines evaluation and correction"""
    from your_evaluation_file import evaluate_response  # Import the evaluation function
    
    async def process_evaluation(inputs: Dict) -> Dict:
        """Process evaluation results and add to inputs"""
        evaluation = evaluate_response(inputs["query"], inputs["response"])
        inputs["suggestions"] = evaluation["improvement_suggestions"]
        return inputs
    
    correction_chain = create_correction_chain(llm)
    
    # Combine evaluation and correction
    combined_chain = (
        RunnablePassthrough.assign(
            evaluation_results=process_evaluation
        )
        | correction_chain
    )
    
    return combined_chain

def correct_response(
    query: str,
    response: str,
    suggestions: List[str] = None
) -> Dict:
    """
    Generate an improved response based on evaluation feedback.
    
    Args:
        query: Original user query
        response: Original response
        suggestions: Optional list of improvement suggestions. 
                   If None, will run evaluation to get suggestions.
    
    Returns:
        Dict containing improved response and metadata
    """
    # Initialize the chain
    if suggestions:
        # Use provided suggestions
        correction_chain = create_correction_chain()
        corrected = correction_chain.invoke({
            "query": query,
            "response": response,
            "suggestions": suggestions
        })
    else:
        # Run full evaluation + correction
        combined_chain = create_combined_correction_chain()
        corrected = combined_chain.invoke({
            "query": query,
            "response": response
        })
    
    return corrected

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_query = "What causes climate change?"
    sample_response = "Climate change is caused by greenhouse gases."
    sample_suggestions = [
        "Add specific examples of greenhouse gases",
        "Explain the greenhouse effect mechanism",
        "Include human activities' impact",
        "Add scientific consensus"
    ]
    
    # Get improved response
    improved = correct_response(
        query=sample_query,
        response=sample_response,
        suggestions=sample_suggestions
    )
    
    print("Improved response:", improved["improved_response"])
    print("\nChanges made:", improved["changes_made"])
    print("\nConfidence score:", improved["confidence_score"])
