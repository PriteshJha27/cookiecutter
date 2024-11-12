from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
from typing import Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_basic_chain():
    """Create a basic LLMChain."""
    
    # Initialize LLM
    llm = ChatAmexLlama()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessagePromptTemplate.from_template("{input_text}")
    ])
    
    # Create chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="result"
    )
    
    return chain

def create_sequential_chain():
    """Create a sequential chain for multi-step processing."""
    
    llm = ChatAmexLlama()
    
    # First chain for analysis
    analysis_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Analyze the following text and identify key points."),
        HumanMessagePromptTemplate.from_template("{input_text}")
    ])
    
    analysis_chain = LLMChain(
        llm=llm,
        prompt=analysis_prompt,
        output_key="analysis"
    )
    
    # Second chain for summary
    summary_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Summarize the following analysis."),
        HumanMessagePromptTemplate.from_template("{analysis}")
    ])
    
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )
    
    # Combine chains
    combined_chain = SimpleSequentialChain(
        chains=[analysis_chain, summary_chain],
        verbose=True
    )
    
    return combined_chain

def main():
    """Test the chain implementations."""
    try:
        # Test basic chain
        print("Testing basic chain...")
        basic_chain = create_basic_chain()
        result = basic_chain.run(
            input_text="What are three key benefits of machine learning?"
        )
        print(f"\nBasic Chain Result: {result}")
        
        # Test sequential chain
        print("\nTesting sequential chain...")
        sequential_chain = create_sequential_chain()
        result = sequential_chain.run(
            "Artificial Intelligence is transforming industries through "
            "automation, data analysis, and predictive capabilities."
        )
        print(f"\nSequential Chain Result: {result}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
