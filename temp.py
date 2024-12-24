
from langchain.agents import AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory

# Define your tools
tools = [...]  # Your list of tools

# Define the agent prompt template
prefix = """You are an AI assistant. You have access to the following tools:"""
suffix = """Begin!"

Question: {input}
Thought: {agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)

# Create the agent
llm_chain = LLMChain(llm=your_llm, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=ConversationBufferMemory()
)


from langchain.agents import initialize_agent, AgentType

agent = initialize_agent(
    tools=tools,
    llm=your_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # or other agent types
    verbose=True
)



from langchain.agents import Agent, AgentOutputParser, Tool
from langchain.schema import AgentAction, AgentFinish

class CustomAgent(Agent):
    @property
    def _agent_type(self) -> str:
        return "custom-agent"
    
    @property
    def observation_prefix(self) -> str:
        return "Observation: "
    
    @property
    def llm_prefix(self) -> str:
        return "Assistant: "
    
    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        # Your agent's planning logic here
        pass

# Create and use your custom agent
custom_agent = CustomAgent(...)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=custom_agent,
    tools=tools,
    verbose=True
)






from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"],
    template=REACT_TEMPLATE
)

chain = LLMChain(
    llm=your_llm,
    prompt=prompt,
    verbose=True
)
