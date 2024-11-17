
from langchain.tools import Tool
from dotenv import load_dotenv
from chat_amex_llama import ChatAmexLlama
from chat_amex_llama_agent import create_amex_agent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
import os
import json

def load_pdf(file_path: str) -> str:
    """Loads the PDF document and returns text content"""
    try:
        # Properly extract text from PDF
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # Return as JSON string with the extracted text
        return json.dumps({
            "status": "success", 
            "document_text": text
        })
    except Exception as e:
        return json.dumps({
            "status": "error", 
            "message": str(e)
        })

def create_text_chunks(text_input: str) -> str:
    """Creates chunks using recursive character splitting"""
    try:
        # Parse input JSON
        input_data = json.loads(text_input)
        text = input_data.get("document_text", "")
        
        if not text:
            raise ValueError("No document text provided")

        # Create text chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=100)
        chunks = splitter.create_documents([text])  # Change split_documents to create_documents
        text_chunks = [chunk.page_content for chunk in chunks]
        
        # Return as JSON string
        return json.dumps({
            "status": "success",
            "chunks": text_chunks
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

class DocumentProcessor:
    def __init__(self):
        self.document_text = None
        self.text_chunks = None

    def process_document(self, file_path: str):
        """Process document and store results for further use"""
        # Initialize LLM
        llm = ChatAmexLlama(
            base_url=os.getenv("LLAMA_API_URL"),
            auth_url=os.getenv("LLAMA_AUTH_URL"),
            user_id=os.getenv("LLAMA_USER_ID"),
            pwd=os.getenv("LLAMA_PASSWORD"),
            cert_path=os.getenv("CERT_PATH")
        )

        # Create tools
        tools = [
            Tool(
                name="load_document",
                func=load_pdf,
                description="Load a PDF document from the given file path and extract its text content. Input should be a file path string. Returns JSON with document text."
            ),
            Tool(
                name="create_chunks",
                func=create_text_chunks,
                description="Create text chunks from the document content using recursive character splitting. Input should be JSON with document_text field. Returns JSON with chunks."
            )
        ]

        # Create agent executor
        agent_executor = create_amex_agent(
            llm=llm,
            tools=tools,
            verbose=True
        )

        try:
            # Run agent
            result = agent_executor.invoke({
                "input": f"Please load the document from {file_path}, extract its text content, and then create text chunks from it. Return the JSON output from the create_chunks tool."
            })

            # Parse the final answer to extract JSON
            try:
                # Look for JSON in the intermediate steps
                for step in result["intermediate_steps"]:
                    if isinstance(step[1], str) and "chunks" in step[1]:
                        chunks_result = json.loads(step[1])
                        if chunks_result["status"] == "success":
                            self.text_chunks = chunks_result["chunks"]
                            return self.text_chunks
                
                raise ValueError("Could not find valid chunks in the result")
                
            except Exception as e:
                print(f"Error parsing result: {str(e)}")
                print("Raw result:", result)
                return None

        except Exception as e:
            print(f"Error in agent execution: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

def main():
    load_dotenv()
    
    # Create processor instance
    processor = DocumentProcessor()
    
    # Process document and get chunks
    text_chunks = processor.process_document("./data/sample.pdf")
    
    if text_chunks is not None:
        print("\nText Chunks Retrieved Successfully:")
        print(f"Number of chunks: {len(text_chunks)}")
        print("\nFirst chunk preview:")
        print(text_chunks[0][:200] + "...")  # Preview of first chunk
        
        # Now you can use text_chunks for further processing
        # For example:
        # do_further_processing(text_chunks)
    else:
        print("Failed to process document")

if __name__ == "__main__":
    main()
