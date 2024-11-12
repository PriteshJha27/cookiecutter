from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from chat_amex_llama import ChatAmexLlama  # Your custom LLM
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Example 1: Basic LCEL Chain
def create_basic_chain():
    """Create a basic LCEL chain for simple text transformation."""
    
    # Initialize your custom LLM
    llm = ChatAmexLlama()
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant that transforms text to {style} style."
        ),
        HumanMessagePromptTemplate.from_template("{input_text}")
    ])
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    return chain

# Example 2: Parallel Processing Chain
def create_parallel_chain():
    """Create a chain that processes input in parallel branches."""
    
    llm = ChatAmexLlama()
    
    # Create different prompts for each branch
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Create a brief summary of the following text."),
        ("human", "{input_text}")
    ])
    
    keywords_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key topics from the text as a comma-separated list."),
        ("human", "{input_text}")
    ])
    
    # Create parallel branches
    chain = RunnableParallel(
        summary=summary_prompt | llm | StrOutputParser(),
        keywords=keywords_prompt | llm | CommaSeparatedListOutputParser()
    )
    
    return chain

# Example 3: Sequential Chain with Memory
def create_sequential_chain():
    """Create a chain that processes input sequentially with context."""
    
    llm = ChatAmexLlama()
    
    # Create prompts for each step
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following text and identify main points."),
        ("human", "{input_text}")
    ])
    
    conclusion_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the analysis, provide key conclusions."),
        ("human", "Analysis: {analysis}\nProvide conclusions:")
    ])
    
    # Create the sequential chain
    chain = (
        RunnableParallel({
            "analysis": analysis_prompt | llm | StrOutputParser(),
            "input_text": RunnablePassthrough()
        })
        | conclusion_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

# Example usage
async def main():
    # Test basic chain
    basic_chain = create_basic_chain()
    result1 = await basic_chain.ainvoke({
        "style": "professional",
        "input_text": "Hey there! How's it going?"
    })
    print("Basic Chain Result:", result1)
    
    # Test parallel chain
    parallel_chain = create_parallel_chain()
    result2 = await parallel_chain.ainvoke({
        "input_text": "Artificial Intelligence is transforming various industries. "
                     "Machine learning models are becoming more sophisticated, "
                     "while neural networks continue to advance."
    })
    print("\nParallel Chain Result:")
    print("Summary:", result2["summary"])
    print("Keywords:", result2["keywords"])
    
    # Test sequential chain
    sequential_chain = create_sequential_chain()
    result3 = await sequential_chain.ainvoke({
        "input_text": "The global climate is changing rapidly. "
                     "Temperatures are rising, and extreme weather events "
                     "are becoming more frequent."
    })
    print("\nSequential Chain Result:", result3)

if __name__ == "__main__":
    asyncio.run(main())
