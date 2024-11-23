import streamlit as st
from pathlib import Path
from dependencies import *
import yaml
import os

def initialize_session_state():
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'executor' not in st.session_state:
        st.session_state.executor = None
    if 'pdf_retrieval_engine' not in st.session_state:
        st.session_state.pdf_retrieval_engine = None
    if 'csv_retrieval_engine' not in st.session_state:
        st.session_state.csv_retrieval_engine = None

def navigate_to(page):
    st.session_state.page = page

def main():
    initialize_session_state()
    
    if st.session_state.page == 'home':
        show_home()
    elif st.session_state.page == 'load_data':
        show_load_data()
    elif st.session_state.page == 'load_vectorstore':
        show_load_vectorstore()
    elif st.session_state.page == 'qa':
        show_qa()

def show_home():
    st.title("RAG System")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Data", use_container_width=True):
            navigate_to('load_data')
    with col2:
        if st.button("Load Vectorstore", use_container_width=True):
            navigate_to('load_vectorstore')

def show_load_data():
    st.title("Load Data")
    
    # Sidebar parameters
    with st.sidebar:
        st.header("Processing Parameters")
        chunk_size = st.slider("Chunk Size", 100, 1000, 256)

    # File upload sections
    st.subheader("Upload Files")
    
    # PDF upload
    pdf_files = st.file_uploader("Upload PDF Files", type=['pdf'], accept_multiple_files=True)
    
    # CSV upload
    st.divider()
    csv_files = st.file_uploader("Upload CSV Files", type=['csv'], accept_multiple_files=True)
    
    # Config upload
    st.divider()
    config = st.file_uploader("Upload Config File", type=['yml', 'yaml'])

    if st.button("Process and Store", type="primary", use_container_width=True):
        if pdf_files and csv_files:
            with st.spinner("Processing files..."):
                try:
                    # Save uploaded files
                    pdf_path = Path("pdf_uploads")
                    csv_path = Path("csv_uploads")
                    pdf_path.mkdir(exist_ok=True)
                    csv_path.mkdir(exist_ok=True)

                    # Save PDFs
                    pdf_dir = str(pdf_path)
                    for pdf in pdf_files:
                        with open(pdf_path / pdf.name, "wb") as f:
                            f.write(pdf.getvalue())

                    # Save CSVs
                    for csv in csv_files:
                        with open(csv_path / csv.name, "wb") as f:
                            f.write(csv.getvalue())

                    # Initialize model and process files
                    model = ChatAmexLlama(
                        base_url=os.getenv("LLAMA_API_URL"),
                        auth_url=os.getenv("LLAMA_AUTH_URL"),
                        user_id=os.getenv("LLAMA_USER_ID"),
                        pwd=os.getenv("LLAMA_PASSWORD"),
                        cert_path=os.getenv("CERT_PATH")
                    )
                    st.session_state.model = model

                    # Process CSV files
                    csv_paths = [str(csv_path / f.name) for f in csv_files]
                    dataframes = load_csv_files()  # Your function
                    executor = SQLiteQueryExecutor()
                    executor.load_dataframes(dataframes)
                    st.session_state.executor = executor

                    # Initialize RAG engines
                    pdf_retrieval_engine = PDFRAG(model_path)
                    pdf_retrieval_engine.process_pdf_directory(
                        pdf_dir=pdf_dir,
                        model_path=model_path,
                        index_dir="vectorstore",
                        chunk_size=chunk_size
                    )
                    st.session_state.pdf_retrieval_engine = pdf_retrieval_engine

                    csv_retrieval_engine = initialize_rag_system(
                        model_path,
                        risk_df,
                        borrower_df,
                        credit_df,
                        ratios_df,
                        statements_df
                    )
                    st.session_state.csv_retrieval_engine = csv_retrieval_engine

                    st.success("Processing complete!")
                    
                    if st.button("Go to Q&A", type="primary"):
                        navigate_to('qa')
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        else:
            st.warning("Please upload required files")

def show_load_vectorstore():
    st.title("Load Existing Vectorstore")
    
    vectorstore_path = st.text_input("Enter path to existing vectorstore:")
    
    if st.button("Load Vectorstore", type="primary"):
        if vectorstore_path:
            try:
                # Load existing vectorstore
                pdf_retrieval_engine = PDFRAG(model_path)
                pdf_retrieval_engine.load_index(vectorstore_path)
                st.session_state.pdf_retrieval_engine = pdf_retrieval_engine
                
                # Initialize model
                model = ChatAmexLlama(
                    base_url=os.getenv("LLAMA_API_URL"),
                    auth_url=os.getenv("LLAMA_AUTH_URL"),
                    user_id=os.getenv("LLAMA_USER_ID"),
                    pwd=os.getenv("LLAMA_PASSWORD"),
                    cert_path=os.getenv("CERT_PATH")
                )
                st.session_state.model = model
                
                st.success("Vectorstore loaded successfully!")
                
                if st.button("Go to Q&A", type="primary"):
                    navigate_to('qa')
            except Exception as e:
                st.error(f"Error loading vectorstore: {str(e)}")
        else:
            st.warning("Please enter a vectorstore path")

def show_qa():
    st.title("Q&A Interface")

    # Display chat history
    for q, a in st.session_state.chat_history:
        st.text_area("Question:", value=q, height=100, disabled=True)
        st.text_area("Answer:", value=a, height=200, disabled=True)
        st.divider()

    # Query input
    query = st.text_area("Enter your question:", height=100)
    
    if st.button("Submit", type="primary"):
        if query:
            if st.session_state.pdf_retrieval_engine and st.session_state.model:
                with st.spinner("Generating response..."):
                    try:
                        answer = retrieval_llm(
                            pdf_rag=st.session_state.pdf_retrieval_engine,
                            csv_rag=st.session_state.csv_retrieval_engine,
                            query=query
                        )
                        
                        # If answer contains SQL query, execute it
                        sql_query = st.session_state.executor.extract_sql_query(answer)
                        if sql_query:
                            final_answer = st.session_state.executor.execute_query(sql_query)
                            answer = final_answer[0] if isinstance(final_answer, list) else final_answer
                        
                        st.session_state.chat_history.append((query, answer))
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
            else:
                st.error("No vectorstore or model loaded. Please load data first.")
        else:
            st.warning("Please enter a question")

    # Clear chat button
    if st.button("Clear Chat", type="secondary"):
        st.session_state.chat_history = []
        st.experimental_rerun()

if __name__ == "__main__":
    main()
