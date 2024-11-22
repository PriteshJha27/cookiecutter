
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent
from langchain.chains import LLMChain
from typing import List, Dict
import re

def example_function(input_data: str) -> str:
    """Example function that processes input"""
    return f"Processed: {input_data}"

def another_function(numbers: List[int]) -> float:
    """Example function that works with numbers"""
    return sum(numbers)

# Method 1: Using initialize_agent with custom prompt
def create_restricted_react_agent_1():
    # Define tools
    tools = [
        Tool(
            name="ProcessData",
            func=example_function,
            description="Use this tool to process input data. Input should be a string."
        ),
        Tool(
            name="SumNumbers",
            func=another_function,
            description="Use this tool to sum numbers. Input should be a list of integers."
        )
    ]

    # Create a custom prompt that enforces tool usage
    custom_prefix = """You are an AI assistant that can ONLY use the provided tools to answer questions. 
    You cannot use any external knowledge or provide information from any other source.
    If you cannot answer using the provided tools, say "I can only help with tasks using my available tools."
    
    You have access to the following tools:"""

    custom_suffix = """You must ALWAYS use one of the provided tools to answer.
    Do not try to answer questions directly without using a tool.

    Question: {input}
    Thought: Let me approach this step by step:
    1. First, I'll identify which tool I need
    2. Then, I'll use only that tool to get the answer
    {agent_scratchpad}"""

    # Initialize the agent with custom prompt
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=custom_prefix,
        suffix=custom_suffix,
        input_variables=["input", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]
    
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=2  # Limit the number of tool uses
    )

# Method 2: Using a custom agent class with stricter control
class RestrictedReactAgent:
    def __init__(self, tools: List[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        
        # Strict prompt template for action extraction
        self.action_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Given the following tools:
            {tool_descriptions}
            
            Select ONLY ONE tool to use for this query: {query}
            
            Respond in the format:
            Tool: <tool_name>
            Input: <tool_input>
            
            If you cannot process the query with available tools, respond with:
            Tool: None
            Input: Cannot process this query with available tools."""
        )
        
    def _get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions"""
        return "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
        
    def _extract_action(self, response: str) -> tuple:
        """Extract tool and input from response"""
        tool_match = re.search(r"Tool: (.+)", response)
        input_match = re.search(r"Input: (.+)", response)
        
        if not tool_match or not input_match:
            return None, None
            
        return tool_match.group(1).strip(), input_match.group(1).strip()
        
    def run(self, query: str) -> str:
        """Run the agent with strict tool usage"""
        # Get tool selection from LLM
        prompt = self.action_prompt.format(
            tool_descriptions=self._get_tool_descriptions(),
            query=query
        )
        
        response = self.llm.predict(prompt)
        tool_name, tool_input = self._extract_action(response)
        
        # Handle cases where no tool can be used
        if tool_name == "None" or tool_name not in self.tools:
            return "I can only help with tasks using my available tools."
            
        # Execute the selected tool
        try:
            result = self.tools[tool_name].func(eval(tool_input))
            return f"Used {tool_name} tool. Result: {result}"
        except Exception as e:
            return f"Error executing tool: {str(e)}"

# Example usage
def demonstrate_restricted_agents():
    # Define test tools
    tools = [
        Tool(
            name="ProcessData",
            func=example_function,
            description="Process a string input and return result"
        ),
        Tool(
            name="SumNumbers",
            func=another_function,
            description="Sum a list of numbers"
        )
    ]
    
    # Method 1: Using modified initialize_agent
    print("\nMethod 1: Modified initialize_agent")
    agent1 = create_restricted_react_agent_1()
    
    # Method 2: Using custom restricted agent
    print("\nMethod 2: Custom restricted agent")
    agent2 = RestrictedReactAgent(tools)
    
    # Test queries
    test_queries = [
        "Process this text: 'hello world'",
        "What is the sum of [1, 2, 3, 4, 5]?",
        "What is the weather today?",  # Should reject
        "Tell me about history"  # Should reject
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("Method 1 response:", agent1.run(query))
        print("Method 2 response:", agent2.run(query))

if __name__ == "__main__":
    demonstrate_restricted_agents()
