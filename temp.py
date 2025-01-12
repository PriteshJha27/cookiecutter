
from typing import Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

def create_evaluation_chain(llm: ChatOpenAI = ChatOpenAI()):
    # Create prompt template for evaluation
    prompt = ChatPromptTemplate.from_template("""
    You are an expert evaluator. Assess if the response adequately answers the user's query.
    Consider these aspects in your evaluation:
    1. Relevance: Does the response directly address the query?
    2. Completeness: Does it cover all aspects of the question?
    3. Accuracy: Is the information provided correct and well-supported?
    4. Clarity: Is the response clear and well-structured?
    
    User Query: {query}
    
    Response to Evaluate: {response}
    
    Provide your evaluation in JSON format with the following structure:
    - evaluation_score: number between 0 and 1
    - is_satisfactory: boolean
    - reasoning: brief explanation of your evaluation
    - improvement_suggestions: list of specific suggestions
    
    Focus on being objective and constructive in your assessment.
    """)
    
    # Create JSON output parser with expected schema
    output_parser = JsonOutputParser()
    
    # Create the chain
    chain = (
        {
            "query": RunnablePassthrough(),
            "response": lambda x: str(x["response"])
        }
        | prompt
        | llm
        | output_parser
    )
    
    return chain

def evaluate_response(query: str, response: str) -> Dict:
    """
    Evaluate the quality and relevance of a response to a user query.
    
    Args:
        query: The original user query
        response: The response to evaluate
        
    Returns:
        Dict: Evaluation results with score, reasoning, and suggestions
    """
    # Initialize the chain
    evaluation_chain = create_evaluation_chain()
    
    # Run evaluation
    evaluation_result = evaluation_chain.invoke({
        "query": query,
        "response": response
    })
    
    return evaluation_result

# Example usage
if __name__ == "__main__":
    # Sample query and response
    sample_query = "What are the main differences between Python 2 and Python 3?"
    sample_response = """
    Python 3 introduced several major changes:
    1. Print is now a function rather than a statement
    2. Unicode strings are default
    3. Division of integers returns float by default
    """
    
    # Run evaluation
    evaluation = evaluate_response(sample_query, sample_response)
    print("Evaluation results:", evaluation)
