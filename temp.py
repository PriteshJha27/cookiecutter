from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import StrOutputParser
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from chat_amex_llama import ChatAmexLlama
from dotenv import load_dotenv
import asyncio

# Load environment variables
load_dotenv()

async def create_basic_chain():
    """Create a basic chain compatible with LangChain v0.1.x"""
    
    # Initialize the LLM
    llm = ChatAmexLlama()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a helpful assistant. Answer the following question."
        ),
        HumanMessagePromptTemplate.from_template("{input_text}")
    ])
    
    # Create chain using LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_key="result"
    )
    
    return chain

async def create_sequential_chain():
    """Create a sequential chain with multiple steps"""
    
    llm = ChatAmexLlama()
    
    # First chain for analysis
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the following text and identify key points."),
        ("human", "{input_text}")
    ])
    
    analysis_chain = LLMChain(
        llm=llm,
        prompt=analysis_prompt,
        output_key="analysis"
    )
    
    # Second chain for summary
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Based on the analysis, provide a concise summary."),
        ("human", "{analysis}")
    ])
    
    summary_chain = LLMChain(
        llm=llm,
        prompt=summary_prompt,
        output_key="summary"
    )
    
    # Combine chains
    combined_chain = SequentialChain(
        chains=[analysis_chain, summary_chain],
        input_variables=["input_text"],
        output_variables=["analysis", "summary"]
    )
    
    return combined_chain

async def main():
    """Test the chains"""
    try:
        # Test basic chain
        print("Testing basic chain...")
        basic_chain = await create_basic_chain()
        result1 = await basic_chain.arun(
            input_text="What are the three main types of machine learning?"
        )
        print("\nBasic Chain Result:", result1)
        
        # Test sequential chain
        print("\nTesting sequential chain...")
        sequential_chain = await create_sequential_chain()
        results = await sequential_chain.arun(
            input_text="Artificial Intelligence is transforming industries through automation, "
                      "data analysis, and predictive capabilities."
        )
        print("\nSequential Chain Results:", results)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
