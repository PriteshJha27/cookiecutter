from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from typing import List, Dict
import os

# Initialize the base LLM
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# Individual tools
def load_pdf(pdf_path: str) -> List:
    """Load PDF document and return pages"""
    loader = PyPDFLoader(pdf_path)
    return loader.load()

def chunk_documents(documents: List) -> List:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def create_embeddings(chunks: List) -> FAISS:
    """Create embeddings and store in vector database"""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def retrieve_docs(vectorstore: FAISS, query: str) -> List:
    """Retrieve relevant documents based on query"""
    return vectorstore.similarity_search(query, k=3)

def query_llm(vectorstore: FAISS, query: str) -> str:
    """Get answer from LLM using retrieved context"""
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain.run(query)

# Initialize individual agents
def create_loader_agent():
    tools = [
        Tool(
            name="LoadPDF",
            func=load_pdf,
            description="Load a PDF document. Input should be the path to the PDF file."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

def create_chunking_agent():
    tools = [
        Tool(
            name="ChunkDocuments",
            func=chunk_documents,
            description="Split documents into smaller chunks. Input should be a list of documents."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

def create_embedding_agent():
    tools = [
        Tool(
            name="CreateEmbeddings",
            func=create_embeddings,
            description="Create embeddings for document chunks and store in vector database. Input should be a list of document chunks."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

def create_retrieval_agent():
    tools = [
        Tool(
            name="RetrieveDocs",
            func=retrieve_docs,
            description="Retrieve relevant documents based on a query. Input should be a dictionary with 'vectorstore' and 'query' keys."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

def create_llm_agent():
    tools = [
        Tool(
            name="QueryLLM",
            func=query_llm,
            description="Get answer from LLM using retrieved context. Input should be a dictionary with 'vectorstore' and 'query' keys."
        )
    ]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.REACT_DOCSTORE,
        verbose=True
    )

class SupervisorAgent:
    def __init__(self):
        # Initialize all sub-agents
        self.loader_agent = create_loader_agent()
        self.chunking_agent = create_chunking_agent()
        self.embedding_agent = create_embedding_agent()
        self.retrieval_agent = create_retrieval_agent()
        self.llm_agent = create_llm_agent()
        
        # Create supervisor tool
        self.tools = [
            Tool(
                name="LoaderAgent",
                func=self.loader_agent.run,
                description="Agent for loading PDF documents"
            ),
            Tool(
                name="ChunkingAgent",
                func=self.chunking_agent.run,
                description="Agent for chunking documents"
            ),
            Tool(
                name="EmbeddingAgent",
                func=self.embedding_agent.run,
                description="Agent for creating embeddings"
            ),
            Tool(
                name="RetrievalAgent",
                func=self.retrieval_agent.run,
                description="Agent for retrieving relevant documents"
            ),
            Tool(
                name="LLMAgent",
                func=self.llm_agent.run,
                description="Agent for querying LLM with context"
            )
        ]
        
        # Initialize supervisor agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=llm,
            agent=AgentType.REACT_DOCSTORE,
            verbose=True
        )
    
    def process_query(self, pdf_path: str, query: str) -> str:
        """Orchestrate the complete RAG pipeline"""
        try:
            # Step 1: Load PDF
            documents = self.loader_agent.run(f"Load PDF from path: {pdf_path}")
            
            # Step 2: Chunk documents
            chunks = self.chunking_agent.run(f"Chunk these documents: {documents}")
            
            # Step 3: Create embeddings
            vectorstore = self.embedding_agent.run(f"Create embeddings for chunks: {chunks}")
            
            # Step 4: Retrieve relevant documents
            retrieval_input = {"vectorstore": vectorstore, "query": query}
            relevant_docs = self.retrieval_agent.run(f"Retrieve documents for query: {retrieval_input}")
            
            # Step 5: Get answer from LLM
            llm_input = {"vectorstore": vectorstore, "query": query}
            final_answer = self.llm_agent.run(f"Get answer for query: {llm_input}")
            
            return final_answer
            
        except Exception as e:
            return f"Error in processing: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Initialize supervisor
    supervisor = SupervisorAgent()
    
    # Process a query
    pdf_path = "path/to/your/document.pdf"
    query = "What is the main topic of the document?"
    
    response = supervisor.process_query(pdf_path, query)
    print(f"Final Answer: {response}")
