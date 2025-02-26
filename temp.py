
# api.py
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
from contextlib import asynccontextmanager

# Import your existing components
from orchestrators.langgraph import LanggraphOrchestrator
from utils.config_management import load_config

# Store active sessions
active_sessions = {}

class SessionRequest(BaseModel):
    """Request model for creating a new session"""
    user_id: Optional[str] = None

class QueryRequest(BaseModel):
    """Request model for user queries"""
    session_id: str
    query: str
    query_type: str = "doc"  # Default query type
    filename: Optional[str] = None

class SessionResponse(BaseModel):
    """Response model for session creation"""
    session_id: str
    status: str

class QueryResponse(BaseModel):
    """Response model for query results"""
    result: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load configuration on startup
    config = load_config()
    yield
    # Clean up sessions on shutdown
    active_sessions.clear()

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/initialize", response_model=SessionResponse)
async def initialize_session(request: SessionRequest):
    """
    Initialize a new session with LanggraphOrchestrator
    This replaces the initialization part of router.py
    """
    try:
        # Generate a unique session ID
        session_id = str(uuid.uuid4())
        
        # Load configuration
        config = load_config()
        
        # Initialize orchestrator for this session (without user input yet)
        orchestrator = LanggraphOrchestrator()
        
        # Store in active sessions
        active_sessions[session_id] = {
            "orchestrator": orchestrator,
            "user_id": request.user_id,
            "state": "initialized"
        }
        
        return SessionResponse(
            session_id=session_id,
            status="initialized"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session initialization failed: {str(e)}")

@app.post("/execute", response_model=QueryResponse)
async def execute_query(request: QueryRequest):
    """
    Execute a query using an established session
    This replaces the execute_pipeline part of router.py
    """
    # Check if session exists
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[request.session_id]
    orchestrator = session["orchestrator"]
    
    try:
        # Update the orchestrator with user query instead of using input()
        orchestrator.initial_state.question = request.query
        
        if request.filename:
            orchestrator.initial_state.filename = request.filename
            
        orchestrator.initial_state.query_type = request.query_type
        
        # Execute the workflow
        result = orchestrator.execute()
        
        # Return results
        return QueryResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the status of a session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": active_sessions[session_id]["state"]}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """End a session and clean up resources"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up the session
    del active_sessions[session_id]
    return {"status": "deleted"}

# Run with: uvicorn api:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



#------------------------------------------------------------------------------------------------------------------------------------------


# orchestrators/langgraph.py
from typing import Dict, Any, Optional
import logging
from orchestrators.base import BaseOrchestrator
from utils.config_management import load_config
from langchain_core.runnables import RunnableConfig
from services.nodes.state import AgentState
from langgraph.graph import StateGraph, Graph, END, START
from services.nodes.query_decomposition import QueryDecompositionNode
from services.nodes.rewrite import RewriteNode
from services.nodes.retrieval import RetrieverNode
from services.nodes.reranker import RerankerNode
from services.nodes.generate import LLMChainNode
from services.nodes.validation import ValidationNode
from services.nodes.correction import CorrectionNode
from services.nodes.get_user_input import GetUserInputNode
from services.nodes.query_complexity import QueryComplexityNode
from services.nodes.pre_summary_node import PreSummaryNode
from services.nodes.final_output import FinalOutputNode

logger = logging.getLogger(__name__)

class LanggraphOrchestrator(BaseOrchestrator):
    """Orchestrates Langgraph operations."""

    def _validate_config(self) -> None:
        if not self.config.get('query_serving', {}).get('files'):
            raise ValueError("Query Serving setup not configured")

    def __init__(self):
        self.config = load_config()
        self.initial_state = AgentState(
            question="",
            theme=None,
            query_complexity=None,
            rewritten_question="",
            retrieved_results=None,
            context="",
            reranked_documents=None,
            llm_result="",
            validation_results=None,
            suggestions=None,
            corrected_prompt="",
            continue_loop=True,
            question_counter=1,
            correction_attempt=1,
            filename="",  # Will be set by the API
            query_type='doc',  # Default, can be overridden by API
            bm25_result="",
            final_output="",
        )
        
        # Initialize nodes
        self.get_user_input = GetUserInputNode()
        self.query_complexity = QueryComplexityNode()
        self.query_rewrite_node = QueryDecompositionNode()
        self.retrieval_node = RetrieverNode()
        self.reranker_node = RerankerNode()
        self.llm_chain_node = LLMChainNode()
        self.validation_node = ValidationNode()
        self.correction_node = CorrectionNode()
        self.presummary_node = PreSummaryNode()
        self.finaloutput_node = FinalOutputNode()
        self.workflow = self.workflow()
        self.config_runnable = RunnableConfig()

    def query_rewriter_router(self, state: AgentState) -> str:
        """Route to correction node or get_input based on satisfaction flag"""
        if state['query_complexity'] == "High":
            return "rewrite_query"
        else:
            return "retrieval"

    def correction_router(self, state: AgentState) -> str:
        """Route to correction node or get_input based on satisfaction flag"""
        if state['continue_loop'] == False:
            return "final_output"
        
        if state['correction_attempt'] <= 2:
            if float(state['validation_results'].get('evaluation_score')) < 0.5:
                return "correction"
            else:
                return "final_output"
        else:
            return "final_output"

    def workflow(self):
        workflow = StateGraph(AgentState)
        
        # Add nodes to the workflow
        workflow.add_node("get_input", self.get_user_input.execute)
        workflow.add_node("query_complexity", self.query_complexity.execute)
        workflow.add_node("rewrite_query", self.query_rewrite_node.execute)
        workflow.add_node("retrieval", self.retrieval_node.execute)
        workflow.add_node("reranker", self.reranker_node.execute)
        workflow.add_node("llm_chain", self.llm_chain_node.execute)
        workflow.add_node("validation", self.validation_node.execute)
        workflow.add_node("correction", self.correction_node.execute)
        workflow.add_node("pre_summ", self.presummary_node.execute)
        workflow.add_node("final_output", self.finaloutput_node.execute)
        
        # Define edges for the workflow
        workflow.add_edge(START, "pre_summ")
        workflow.add_edge("pre_summ", "get_input")
        workflow.add_edge("get_input", "query_complexity")
        workflow.add_edge("rewrite_query", "retrieval")
        workflow.add_edge("retrieval", "reranker")
        workflow.add_edge("reranker", "llm_chain")
        workflow.add_edge("llm_chain", "validation")
        # workflow.add_edge("validation", "correction")
        workflow.add_edge("correction", "retrieval")
        workflow.add_edge("final_output", END)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validation",
            self.correction_router,
            {
                "final_output": "final_output",
                "correction": "correction"
            }
        )
        
        workflow.add_conditional_edges(
            "query_complexity",
            self.query_rewriter_router,
            {
                "rewrite_query": "rewrite_query",
                "retrieval": "retrieval"
            }
        )
        
        return workflow

    def execute(self) -> Dict[str, Any]:
        """Load documents from configured source."""
        try:
            graph = self.workflow.compile()
            
            # Process the stream once
            result = None
            for output in graph.stream(self.initial_state, self.config_runnable):
                if not output.get('continue_loop', True):
                    result = output
                    break
                else:
                    result = output
            
            logger.info(f"Successfully completed query response")
            
            # Return the final state for the API response
            return {
                "question": result.get("question", ""),
                "rewritten_question": result.get("rewritten_question", ""),
                "final_output": result.get("final_output", ""),
                "llm_result": result.get("llm_result", ""),
                "validation_score": result.get("validation_results", {}).get("evaluation_score", 0),
                "query_complexity": result.get("query_complexity", "Low")
            }
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise



#------------------------------------------------------------------------------------------------------------------------------------------

# services/nodes/get_user_input.py
import logging
from typing import Dict, Any
from services.nodes.state import AgentState

logger = logging.getLogger(__name__)

class GetUserInputNode:
    """
    Node to handle user input.
    In API mode, this node simply passes through the query that
    was already set in the state by the API endpoint.
    """
    
    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Process user input or validate input from API
        
        Args:
            state: The current state
            
        Returns:
            Updated state with user input
        """
        # Log the current query
        if state.question:
            logger.info(f"Processing query: {state.question}")
            
            # Validate the query
            if len(state.question.strip()) == 0:
                logger.warning("Empty query received")
                # In API context, we'll return an error message
                return {
                    **state,
                    "continue_loop": False,
                    "final_output": "Error: Empty query received"
                }
                
            # If filename is required but not provided
            if not state.filename:
                logger.warning("No filename provided")
                return {
                    **state,
                    "continue_loop": False,
                    "final_output": "Error: No filename provided for the query"
                }
                
            # Return unchanged state since we already have the input from the API
            return state
        else:
            # This should not happen in API mode, but just in case
            logger.error("No query provided in state")
            return {
                **state,
                "continue_loop": False,
                "final_output": "Error: No query provided"
            }






#------------------------------------------------------------------------------------------------------------------------------------------

# main.py
import logging
import sys
import os
from utils.config_management import load_config
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('query_serving.log')
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # Load and validate configuration
        config = load_config()
        
        # Get current directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        print(curr_dir)
        sys.path.insert(0, curr_dir)
        
        # Add safechains directory to path if needed
        if config['llm']['safechains_directory'] not in sys.path:
            sys.path.append(config['llm']['safechains_directory'])
        
        # Import and run the API
        import uvicorn
        from api import app
        
        # Run the FastAPI application
        uvicorn.run(app, host="0.0.0.0", port=8000)
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()







#------------------------------------------------------------------------------------------------------------------------------------------






#------------------------------------------------------------------------------------------------------------------------------------------
